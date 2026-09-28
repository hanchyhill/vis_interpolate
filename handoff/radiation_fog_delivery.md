# 山谷辐射雾订正：交付说明

交付日期：2026-09-28
设计与决策记录：`handoff/radiation_fog_valley.md`（第 1 节山谷边界，第 4 节订正方案，第 5 节实现补充）
状态：已实现并通过测试，**尚未提交**；待用更多冬季个例做新旧对比和调参

---

## 1. 交付内容概览

原算法在冬季辐射雾时，低能见度会在两个环节向外扩散：一是区域站估算（`_estimate_one()`）从雾站继承低能见度，二是 IDW 插值把雾站的影响带到周围格点。本次交付的做法是：先判定哪些站属于山谷辐射雾（雾站），再把雾站的影响限制在其所在山谷的"影响域"内，两个环节同时生效。非雾站的处理与原算法完全相同。

| 类别 | 内容 |
|---|---|
| 静态产品 | 山谷产品 `data/assets/dem/valley_fusion_t20_near_m0.nc`：`valley_id` 网格，以及每个山谷的雾顶、谷底平均海拔、面积、中心点 |
| 业务流水线 | 启用后，每个时次同时输出原算法结果和订正结果 |
| 对比工具 | `src/business/fog_compare.py`，用于个例新旧对比和调参 |
| 测试 | 新增 16 个测试，全部测试共 50 个，均通过 |

原算法的保留方式：
- `radiationFog.enabled = false`，或者配置中没有 `radiationFog` 段时，行为与改动前完全一致；
- 启用后，没有雾站的时次结果与原算法逐位一致；
- 原逐行估算实现保留为 `_estimate_one_legacy()`，并有测试保证新旧估算实现结果一致；
- 订正步骤出错时，记录日志并发布原算法结果。

---

## 2. 算法流程（每个时次、每条参考路径）

1. **站点归属**：站点取最近格点的 `valley_id`；站点海拔高于所在山谷雾顶时视为非山谷站（山顶站）。
2. **雾站判别**：同时满足以下条件：
   - 能见度 < 500 m；
   - 位于山谷内，且海拔不高于雾顶；
   - 1 小时降水 `pre_1h` ≤ 0 mm。降水缺测时默认按无降水处理。
3. **虚拟雾站**：同时满足以下条件的湿度站，按雾站处理，能见度取 50 km 内实测雾站能见度的中位数：
   - 所在山谷内没有能见度观测；
   - 湿度 ≥ 95%，且无降水；
   - 50 km 内至少有 1 个实测雾站。
4. **影响域**：在雾站所在山谷内，以谷内各站为种子做最近站划分（Voronoi），离雾站比离非雾站更近的部分为影响域。
5. **权重修正系数 g**：

   ```
   g = exp(-Δz / σz) · exp(-d_out / σd)，g < 0.05 时取 0
   ```

   Δz 为目标点高出雾顶的米数，d_out 为目标点到影响域的距离（km）。影响域内、不高于雾顶处 g = 1。
6. **估算环节**：雾站作为参考站时，其权重乘以 g。g = 0 的雾站被跳过，由更远的参考站补上。
7. **补充限制**：谷内估算值 < 500 m 的区域站，同样只在影响域内起作用，避免在 IDW 中再次向外扩散。
8. **IDW 环节**：格点对雾站的权重乘以 g，其余站不变。

两条参考路径分别判别：`national` 路径只用国家站的能见度，`national_and_regional` 路径同时使用国家站和区域站。

---

## 3. 部署与启用

### 3.1 生成山谷产品（每台机器一次）

`data/` 在 `.gitignore` 中，山谷产品不会进入版本库，需要在每台机器上生成（约 15 秒）或直接拷贝：

```bash
uv run python -m src.valley_boundary --product
# 输出 data/assets/dem/valley_fusion_t20_near_m0.nc
```

### 3.2 在配置中启用

在 `src/config/local.config.json` 或 `server.config.json` 中加入以下配置段（模板见 `src/config/business.config.example.json`；Docker 部署参照 `business.docker.config.example.json`，其中 `valleyPath` 为 `/data/...`）：

```json
"radiationFog": {
  "enabled": true,
  "valleyPath": "data/assets/dem/valley_fusion_t20_near_m0.nc",
  "visThresholdM": 500,
  "rhThresholdPct": 95,
  "precipThresholdMm": 0,
  "precipMissingAsDry": true,
  "sigmaZM": 50,
  "sigmaDKm": 2,
  "gCutoff": 0.05,
  "inferRadiusKm": 50
}
```

启用后如果找不到山谷产品，日志中会出现警告，并自动退回原算法。

### 3.3 回滚

把 `enabled` 改为 `false` 或删除该配置段即可，无需回退代码。

### 3.4 资源开销

在 2025-02-28 个例上（DEM 网格 612×1020）：
- 订正步骤每时次约增加一次 IDW 的耗时，两条路径合计约 5–9 秒；
- 估算环节改为向量化后，从每条路径约 2.2 秒降到约 0.01 秒，部分抵消了新增耗时；
- 没有雾站的时次直接复用原算法格点，不增加耗时。

---

## 4. 输出变化（仅在启用时）

| 输出 | 变化 |
|---|---|
| NetCDF | `visibility` 为订正后结果；新增 `visibility_original`（原算法）和 `fog_influence`（格点受雾站影响的 g 最大值）；全局属性中记录雾站统计（`fog_observed`、`fog_inferred`、`fog_domains` 等） |
| 站点 CSV | 新增 `valley_id`、`is_radiation_fog`（0 = 否，1 = 实测雾站，2 = 虚拟雾站）、`fog_domain_id`、`pre_1h`、`vis_original` |
| 出图 | 业务图使用订正后的 `visibility`，读取方式不变 |
| 日志 | timings 中增加 `radiation_fog_seconds` 和 `radiation_fog` 统计 |
| 接口 | 读取可选字段 `V13019`（1 小时降水）；接口不返回该字段时按缺测处理 |

---

## 5. 个例对比与调参

```bash
# 使用本地原始 CSV（SurfAuto / SurfAwst，GBK 或 UTF-8，有无首行数量行均可）
uv run python -m src.business.fog_compare --time 202502280000 \
    --national-csv data/SurfAuto_20250228000000.csv --regional-csv data/SurfAwst_20250228000000.csv

# 从接口获取（首次获取后缓存到 output/fog_compare/<时次>/input_*.csv；加 --refresh 重新获取）
uv run python -m src.business.fog_compare --time 202501150000

# 调参：每组参数输出到单独的子目录，可并排比较
uv run python -m src.business.fog_compare --time 202501150000 --sigma-d 3 --tag sd3
```

`--time` 为 UTC 时次。可覆盖的参数：`--vis-threshold`、`--rh-threshold`、`--precip-threshold`、`--precip-missing-as-wet`、`--sigma-z`、`--sigma-d`、`--g-cutoff`、`--infer-radius`、`--valley`、`--source`。

输出目录为 `output/fog_compare/<时次>/<参数标签>/`，参数标签默认由参数拼成，例如 `v500_rh95_sz50_sd2_g0.05_r50`：

| 文件 | 内容 |
|---|---|
| `<路径>_compare.png` | 原算法、订正结果、g 场三联图，叠加山谷边界、实测雾站（红点）和虚拟雾站（紫色三角） |
| `<路径>.nc` | `visibility`、`visibility_original`、`fog_influence` |
| `<路径>_stations.csv` | 订正后站点表，含 `vis_original` |
| `stats.csv` | 雾站数、影响域数、广东境内能见度 < 500 m 和 < 1 km 的面积（订正前后） |
| `settings.txt` | 本次使用的完整参数 |

调参方向：

| 参数 | 调大的效果 |
|---|---|
| `sigmaDKm` | 雾区水平边缘更宽、过渡更缓 |
| `sigmaZM` | 雾区沿坡向上延伸得更高 |
| `gCutoff` | 远处更早完全规避 |
| `rhThresholdPct` / `inferRadiusKm` | 控制推断起雾（虚拟雾站）的多少 |

---

## 6. 已完成的验证

- **单元测试**：`uv run python -m unittest discover -s tests -p "test_*.py"`，共 50 个测试，全部通过。新增测试覆盖：
  - 山顶站归属、降水和降水缺测的判别；
  - g 在影响域内为 1、水平和垂直方向的衰减；
  - 谷内 Voronoi 划分、虚拟雾站及推断半径；
  - 估算环节规避与补位、IDW 远处零权重；
  - 无雾时结果逐位一致、新旧估算实现结果一致；
  - 降水字段解析、配置读取、流水线输出变量。
- **山谷产品**：与已审定的 `fusion_t20_near_m0` 结果逐像元一致（全域 114 个山谷，广东境内 85 个）。
- **真实个例 2025-02-28 00 UTC（北京时间 08 时）**：两条路径均判出 8 个实测雾站、11 个虚拟雾站，共 16 个影响域。

  | 路径 | 广东境内 < 1 km 面积（原算法 → 订正） |
  |---|---|
  | national | 15430 → 3016 km² |
  | national_and_regional | 10099 → 3907 km² |

  - 粤西雷州半岛的低能见度不在山谷内，未被改动；
  - 对比图位于 `output/fog_compare/202502280000/v500_rh95_sz50_sd2_g0.05_r50/`。

---

## 7. 尚未完成 / 已知局限

1. **个例验证不足**：只检验了 1 个个例，尚未与卫星可见光云图对照；需要补充更多冬季辐射雾个例，以及降水导致低能见度的个例（后者应基本不被改动）。
2. **降水缺测**：默认按无降水处理。区域站 `V13019` 的缺测率在该个例中约 0.3%，影响不大；如果其他时次缺测较多，可以用 `--precip-missing-as-wet` 对比两种处理的差别。
3. **平原雾不处理**：订正只针对山谷。平原或沿海的大雾（例如平流雾）仍按原算法处理，这是设计上的选择。
4. **间接变化**：g = 0 的区域也可能有变化，原因是部分区域站的估算值不再继承雾站的低能见度。该个例中这类变化以能见度升高为主，属于预期效果。
5. **山谷编号**：编号不连续，而且会随山谷参数变化，不要在其他地方硬编码。

---

## 8. 改动文件清单

| 文件 | 类型 |
|---|---|
| `src/business/radiation_fog.py` | 新增：判别、影响域、订正估算、订正产品 |
| `src/business/fog_compare.py` | 新增：个例对比工具 |
| `tests/test_radiation_fog.py` | 新增：13 个测试 |
| `src/business/algorithms.py` | 修改：估算向量化并支持权重修正；保留 `_estimate_one_legacy()`；新增 `reference_sets()` |
| `src/business/idw.py` | 修改：可选参数 `fog_influence` |
| `src/business/pipeline.py` | 修改：接入订正流程，出错时退回原算法 |
| `src/business/config.py` | 修改：`RadiationFogSettings` |
| `src/business/api.py` | 修改：可选字段 `V13019 → pre_1h` |
| `src/valley_boundary.py` | 修改：`valley_product()`、`export_valley_product()`、`--product` |
| `tests/test_valley_boundary.py` | 修改：新增 3 个测试 |
| `src/config/business.config.example.json`、`business.docker.config.example.json` | 修改：新增 `radiationFog` 配置段 |
| `handoff/radiation_fog_valley.md` | 修改：状态更新，新增第 5 节 |
