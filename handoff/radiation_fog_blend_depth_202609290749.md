# 山谷辐射雾订正：定稿算法 blend_depth

> 生成时间：2026-09-29 07:49（北京时）
> 前序文档：`handoff/radiation_fog_valley.md`（方案设计）、`handoff/radiation_fog_delivery.md`（首轮交付与各模式对比）
> 状态：填充模式定为 `blend_depth`，已设为代码和示例配置的默认值；尚未提交 git（基线提交 `5ef37bf`）

---

## 1. 结论

- 三个 2026 年 9 月个例比较了四种填充模式：`weight`、`blend_linear`、`blend`、`blend_depth`。最终选定 `blend_depth`。
- `blend_depth` 的效果：
  - 雾区按山谷形状填满，边缘与山谷边界吻合（继承自 `blend`）；
  - 谷内能见度随深度渐变，谷底最低，向谷缘和雾顶升到 `visThresholdM`（1000 m）。
- 其余三种模式保留为可选项（`fillMode`），用于个例对照和回退。原算法（未订正）的输出仍一并写入 NetCDF。

---

## 2. 算法流程（每个时次、每条参考路径分别执行）

两条参考路径分别判别：`national` 只用国家站的能见度；`national_and_regional` 同时用国家站和区域站。

### 2.1 前处理（与首轮交付相同）

1. **山谷产品**：使用 `fusion_t20_near_m0` 融合山谷产品（`data/assets/dem/valley_fusion_t20_near_m0.nc`），重采样到 IDW 用的 0.01° DEM 网格。合并后山谷的雾顶海拔取各组成单元的面积加权平均。
2. **站点归属**：每个站取最近格点所在的山谷。站点海拔高于该谷雾顶时视为山顶站，不算谷内站。
3. **实测雾站**：谷内站，且能见度 < `visThresholdM`（1000 m），且 1 小时降水 `pre_1h`（V13019）≤ `precipThresholdMm`。降水缺测默认按无降水处理。
4. **虚拟雾站**：满足以下全部条件的山谷，取一个湿度站作为虚拟雾站：
   - 谷内没有能见度观测；
   - 有湿度站 RH ≥ 95% 且无降水；
   - 50 km 内至少有一个实测雾站。

   虚拟雾站的能见度取这些实测雾站能见度的中位数。
5. **影响域**：
   - 在谷内用 Voronoi 划分，只保留离雾站更近的部分；
   - 影响域外的衰减系数 g = exp(−Δz/σz) · exp(−d_out/σd)，其中 Δz 为高出雾顶的高度，d_out 为到影响域的水平距离；
   - σz = 50 m，σd = 2 km，g < 0.05 时截断为 0。
6. **站点估算**：区域站估算时，雾站的权重乘以 g。取最近 8 个候选站，其中 g > 0 的前 4 个参与估算。估算结果 < 阈值且落在影响域内的站，记为该影响域的成员（`fog_domain_id`）。

### 2.2 格点合成：blend_depth（本次定稿）

对每个影响域：

1. **基底场 V_base**：去掉全部雾域站（实测雾站、虚拟雾站、影响域成员）后，用原各向异性 IDW 插值得到。
2. **雾场 V_fog**：只用本影响域内的实测雾站和虚拟雾站，做各向异性 IDW。格点与站点重合时，取最近站的值。
3. **深度渐变**，由函数 `_depth_profile` 实现：
   - 谷底海拔 z_floor 取影响域内 DEM 海拔的 5% 分位数。雾顶与谷底的高差 H = 雾顶 − z_floor，下限为 10 m。
   - 格点相对深度 r = clip((雾顶 − z) / H, 0, 1)，其中 z 为格点的 DEM 海拔。影响域外，以及域内高出雾顶的格点，r = 0。
   - 雾站相对深度 r_s 用**站点所在格点的 DEM 海拔**计算，不用站点海拔（原因见第 5 节）。r_s 用 IDW 插值到格点得到 r̄，下限为 0.1。
   - 雾区能见度（对数空间内沿深度线性变化）：

     \[
     V_\text{fog}' = V_\text{edge}\cdot\left(\frac{V_\text{fog}}{V_\text{edge}}\right)^{\min(r/\bar r,\;2)},\qquad V_\text{edge}=\max(\text{visThresholdM},\,V_\text{fog})
     \]

   - 这个公式的含义：
     - 在雾站所在深度（r = r̄），等于雾站插值；
     - 在雾顶和谷缘（r = 0），升到 1000 m；
     - 比雾站更深的地方继续降低，指数上限为 2，即谷底相对雾站最多再降低一个同样的对数幅度。
4. **对数混合**：V = V_fog'^g · V_base^(1 − g)，能见度下限为 10 m。
   - 影响域内 g = 1，不做垂直衰减，使雾区与山谷边界一致；
   - 影响域外沿用 g 的衰减。影响域边界处的值约为 1000 m，向外很快过渡到基底场，所以边缘清晰。
   - 格点受多个影响域覆盖时，取 g 最大的影响域。

`blend_depth` 中的固定常数（`src/business/radiation_fog.py` 顶部）：

| 常数 | 值 | 作用 |
|---|---|---|
| `_MIN_BLEND_VIS_M` | 10 m | 能见度下限 |
| `_MIN_DEPTH_RANGE_M` | 10 m | 雾顶与谷底高差的下限，避免浅谷中相对深度被放大 |
| `_MIN_SEED_DEPTH` | 0.1 | 雾站相对深度下限，避免雾站贴近雾顶时指数失控 |
| `_MAX_DEPTH_EXPONENT` | 2 | 谷底相对雾站的最大降幅（对数意义上） |

### 2.3 其余可选模式（保留，用于对照）

| `fillMode` | 说明 |
|---|---|
| `blend` | 与 `blend_depth` 相同，但不做深度渐变，整个影响域填成 V_fog |
| `blend_linear` | 线性混合 g·V_fog + (1 − g)·V_base，域内保留垂直衰减 |
| `weight` | 最初方案：IDW 中格点对雾站的权重乘以 g，谷外好站仍会压制谷内雾区 |

---

## 3. 部署、启用与回滚

1. **生成山谷产品**（每台机器一次，约 15 秒；`data/` 不入库，也可直接拷贝）：

   ```bash
   uv run python -m src.valley_boundary --product
   # 输出 data/assets/dem/valley_fusion_t20_near_m0.nc（广东境内 85 个山谷，全部 114 个）
   ```

2. **启用**：在个人配置（`local.config.json` / `server.config.json`）中加入 `radiationFog` 段，内容可直接从示例配置复制。目前个人配置中**没有**这一段，订正处于关闭状态。

   ```json
   "radiationFog": {
     "enabled": true,
     "valleyPath": "data/assets/dem/valley_fusion_t20_near_m0.nc",
     "visThresholdM": 1000,
     "rhThresholdPct": 95,
     "precipThresholdMm": 0,
     "precipMissingAsDry": true,
     "sigmaZM": 50,
     "sigmaDKm": 2,
     "gCutoff": 0.05,
     "inferRadiusKm": 50,
     "fillMode": "blend_depth",
     "ncRoot": "idw_nc_radiation_fog",
     "imgRoot": "vis_img_radiation_fog"
   }
   ```

   `ncRoot`、`imgRoot` 可省略，相对路径按 `dataRoot` 解析，默认值即上面的值。

   Docker 配置使用 `/data/...` 路径，见 `src/config/business.docker.config.example.json`。省略 `fillMode` 时默认为 `blend_depth`。

3. **关闭与回滚**：
   - 设 `"enabled": false`，或删除整个 `radiationFog` 段，即不再生成订正产品；
   - 设 `"fillMode": "blend"`，退回到不带深度渐变的版本。

   原算法产品在任何情况下都不受影响（见第 4 节）。

---

## 4. 输出：原算法与订正产品并行发布（2026-09-29 更新）

为了做一段时间的对比测试，业务流水线（`python -m src.business`）改为两套产品并行发布：

- **原算法产品保持不变**：CSV、NetCDF、PNG 的路径、文件名和内容，都与引入订正之前完全一致。启用订正也不会改动它们，单元测试会逐格点核对。
- **订正产品单独发布**：在原算法产品发布之后才生成，由 `pipeline._build_radiation_fog_outputs` 负责。
  - 订正过程中出错时（包括山谷产品缺失），只在日志中记录，并在 timings 中标记 `radiation_fog_error`；
  - 原算法产品照常发布，该时次仍记为成功。

订正产品路径（`<YYYY>/<MM>/<DD>` 为 UTC 日期，`<stamp>` 为 UTC 时次 `YYYYMMDDHHmm`）：

| 产品 | 路径 |
|---|---|
| 格点 NetCDF | `<dataRoot>/idw_nc_radiation_fog/<路径>/<YYYY>/<MM>/<DD>/visibility_radiation_fog_<stamp>.nc` |
| 站点 CSV | `<dataRoot>/idw_nc_radiation_fog/<路径>/<YYYY>/<MM>/<DD>/station_vis_radiation_fog_<stamp>.csv` |
| 图片 | `<dataRoot>/vis_img_radiation_fog/<YYYY>/<MM>/<DD>/visibility_radiation_fog_<路径>_<stamp>.png` |

- `<路径>` 为 `national` 或 `national_and_regional`。
- 图片与原算法图用同一个绘图函数（`plot_visibility`），色标和范围一致，可以直接并排比较。标题为"广东省能见度（辐射雾订正 blend_depth）- <路径> - <北京时>"。
- 订正图上标出识别到的雾站，符号与 `fog_compare` 一致：实测雾站为红色圆点，虚拟雾站为紫色三角，右下角图例注明数量。站点来自同时次的订正站点 CSV（通过 `plot_visibility(..., stations_path=...)` 传入）。原算法图不加标记。
- 用 09-22 07 时的真实数据完整跑过一遍 `_build_outputs`，生成了 6 个原算法文件和 6 个订正文件。订正部分额外耗时约 15 秒，整个时次约 30 秒。

订正产品的内容：

- **格点 NetCDF**：
  - `visibility`：订正后能见度（m）；
  - `visibility_original`：原算法能见度（m）；
  - `fog_influence`：g 的最大值，域内为 1。
  - 全局属性含 `fill_mode = "blend_depth"`；`visibility` 的属性 `radiation_fog_correction` 记录了合成方法。
- **站点 CSV**：新增以下列：
  - `valley_id`；
  - `is_radiation_fog`：0 为非雾站，1 为实测雾站，2 为虚拟雾站；
  - `fog_domain_id`；
  - `pre_1h`；
  - `vis_original`。
- **日志**：新增 `radiation_fog` 耗时，并记录降水缺测站数。

---

## 5. 定稿前的一处修正：雾站深度改用 DEM 海拔

`blend_depth` 最初按站点海拔计算雾站深度，结果 09-21 07 时影响域内能见度中位数只有 40 m（雾站观测为 100–200 m），谷底被过度外推。

原因是站点海拔与所在 0.01° 格点的 DEM 海拔可相差上百米：

| 站点 | 站点海拔 | 所在格点 DEM | 按站点海拔算的 r_s | 按 DEM 算的 r_s |
|---|---|---|---|---|
| 德庆 59269 | 129 m | 27 m | 0.16 | 0.92 |
| 龙川 59107 | 180 m | 139 m | 0.07 | 0.37 |
| 阳山 59075 | 155 m | 85 m | 0.38 | 0.94 |

格点深度用的是 DEM，雾站深度也必须用同一套海拔，否则雾站会显得偏浅。改用 DEM 后，分布恢复合理（见第 6 节）。

注意：IDW 本身的各向异性距离仍使用站点海拔，这部分与原算法一致，没有改动。

---

## 6. 验证

**单元测试**：55 个全部通过：

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

新增的 `test_blend_depth_lowest_at_valley_floor` 构造了一个自西向东加深的合成山谷，检查以下几点：
- 雾站处等于观测值；
- 能见度随深度单调降低；
- 谷缘值在 500–1000 m 之间；
- 平底谷中与 `blend` 相同；
- 谷外与 `blend` 相同。

**个例**（`visThresholdM = 1000`，其余参数为默认值；输入已缓存在 `output/fog_compare/<UTC>/input_*.csv`）。

广东境内能见度 < 1 km 的面积（km²）：

| 北京时 | UTC 目录 | 路径 | 原算法 | blend | blend_depth |
|---|---|---|---|---|---|
| 09-21 07 时 | `202609202300` | national | 326 | 3467 | 2847 |
| | | national_and_regional | 391 | 3576 | 2938 |
| 09-22 06 时 | `202609212200` | national | 82 | 1979 | 1810 |
| | | national_and_regional | 160 | 2054 | 1886 |
| 09-22 07 时 | `202609212300` | national | 67 | 3168 | 2402 |
| | | national_and_regional | 79 | 3251 | 2402 |

影响域内能见度分位数（m，national_and_regional 路径）：

| 北京时 | 模式 | 5% | 25% | 50% | 75% | 95% |
|---|---|---|---|---|---|---|
| 09-21 07 时 | blend | 100 | 100 | 200 | 200 | 200 |
| | blend_depth | 40 | 96 | 265 | 691 | 1000 |
| 09-22 06 时 | blend | 200 | 200 | 400 | 400 | 400 |
| | blend_depth | 173 | 312 | 547 | 1000 | 1000 |
| 09-22 07 时 | blend | 10 | 10 | 100 | 100 | 100 |
| | blend_depth | 10 | 59 | 162 | 849 | 1000 |

**图件**：

| 内容 | 路径 |
|---|---|
| 单个时次的三联图（原算法 / 订正 / g 场），全省图和放大图 | `output/fog_compare/<UTC>/blend_depth_v1000_rh95_sz50_sd2_g0.05_r50/` |
| 五种结果同图（原算法 + 四种模式），每个时次一张 | `output/fog_compare/<UTC>/modes_v1000_rh95_sz50_sd2_g0.05_r50.png` |
| 三个时次汇总 | `output/fog_compare/modes_v1000_rh95_sz50_sd2_g0.05_r50_202609202300-202609212300.png` |
| 局部细节（梅州平远一带：blend、blend_depth 与 DEM 并排） | `output/fog_compare/202609202300/blend_depth_detail_pingyuan.png` |

---

## 7. 个例对比工具用法

```bash
# 单个或多个时次（UTC），默认即 blend_depth
uv run python -m src.business.fog_compare --time 202609202300 202609212200 202609212300

# 指定其他模式，用于对照
uv run python -m src.business.fog_compare --time 202609202300 --fill-mode blend

# 各模式跑完后，不重新计算，把原算法和各模式画在同一张图上
uv run python -m src.business.fog_compare --time 202609202300 202609212200 202609212300 --plot-modes
```

- 可覆盖的参数：`--fill-mode`、`--vis-threshold`、`--rh-threshold`、`--precip-threshold`、`--precip-missing-as-wet`、`--sigma-z`、`--sigma-d`、`--g-cutoff`、`--infer-radius`、`--valley`、`--source`。
- 本地原始 CSV 用 `--national-csv` / `--regional-csv` 传入，此时只能给一个时次。
- 输出目录为 `output/fog_compare/<UTC>/<模式>_<参数>/`。

---

## 8. 已知局限与后续方向

1. **可疑雾站尚未处理**：
   - 龙川 59107：RH 69%，能见度 200 m；
   - 德庆 59269：09-22 07 时能见度 0 m。

   可考虑给实测雾站加一个湿度下限（如 RH ≥ 85%），或剔除能见度为 0 的记录。
2. **虚拟雾站会整谷填雾**：例如连山天鹅湖站，原算法估算约 25 km，订正后为 100–400 m。需对照卫星云图判断是否偏多；偏多时可提高 `rhThresholdPct` 或减小 `inferRadiusKm`。
3. **谷缘值等于 `visThresholdM`**：谷缘能见度直接取判别阈值。如果希望谷缘更浓或更淡，可以新增独立配置项（如 `edgeVisM`）。
4. **深度相关常数没有配置化**：谷底取 5% 分位数、指数上限 2 均写在代码中。如需调参，可以提到 `radiationFog` 配置中。
5. **全省图上看不出深度渐变**：0–1 km 色段在全省尺度上较难分辨，需查看放大图或局部细节图。
6. **验证样本有限**：目前只有 4 个个例（2025-02-28，以及 2026-09 的三个时次），没有与卫星雾产品做定量检验。

---

## 9. 本轮改动文件

| 文件 | 改动 |
|---|---|
| `src/business/radiation_fog.py` | 新增 `_seed_idw`、`_depth_profile` 和四个常数；`blend_fog_grid` 增加 `depth` 参数；`build_corrected_products` 支持 `blend_depth`，域内 g = 1 |
| `src/business/config.py` | `FILL_MODES` 新增 `blend_depth`，默认 `fill_mode` 改为 `blend_depth`；新增 `radiation_fog_nc_root` / `radiation_fog_img_root`（配置项为 `radiationFog.ncRoot` / `imgRoot`） |
| `src/business/plot.py` | `plot_visibility` 新增可选参数 `stations_path`，由 `_plot_fog_stations` 标记实测雾站和虚拟雾站 |
| `src/business/pipeline.py` | 原算法产品恢复为不含订正的原样输出；订正产品由 `_build_radiation_fog_outputs` 单独发布到独立目录，失败时互不影响；抽出 `_publish`、`_plot_all`、`radiation_fog_paths` |
| `src/business/fog_compare.py` | `--fill-mode` 支持 `blend_depth`；`--time` 支持多个时次；新增 `--plot-modes` 多模式同图（`plot_modes`、`find_mode_files`）；抽出通用子图函数 `_draw_panel` |
| `src/config/business.config.example.json`、`business.docker.config.example.json` | `fillMode` 改为 `blend_depth` |
| `tests/test_radiation_fog.py` | 新增 `test_blend_depth_lowest_at_valley_floor`；默认模式断言改为 `blend_depth`；新增三个流水线测试：原算法产品不变且订正产品单独输出、订正失败不影响原算法、关闭时不产生订正文件；新增输出目录配置解析的测试 |
| `handoff/radiation_fog_delivery.md` | 补充 `blend_depth` 说明、对比表和多模式作图用法 |

个人配置（`local.config.json`、`server.config.json`）未修改；按 AGENTS.md 要求，`output/` 下的 NetCDF/PNG 和站点数据不应提交。
