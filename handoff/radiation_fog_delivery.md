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
| 测试 | 新增 18 个测试，全部测试共 52 个，均通过 |

原算法的保留方式：
- `radiationFog.enabled = false`，或者配置中没有 `radiationFog` 段时，行为与改动前完全一致；
- 启用后，没有雾站的时次结果与原算法逐位一致；
- 原逐行估算实现保留为 `_estimate_one_legacy()`，并有测试保证新旧估算实现结果一致；
- 订正步骤出错时，记录日志并发布原算法结果。

---

## 2. 算法流程（每个时次、每条参考路径）

1. **站点归属**：站点取最近格点的 `valley_id`；站点海拔高于所在山谷雾顶时视为非山谷站（山顶站）。
2. **雾站判别**：同时满足以下条件：
   - 能见度 < 1000 m（`visThresholdM`，2026-09-28 由 500 m 放宽）；
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
7. **补充限制**：谷内估算值 < `visThresholdM` 的区域站，同样只在影响域内起作用，避免在 IDW 中再次向外扩散。
8. **格点合成**，由 `fillMode` 选择：
   - `blend`（默认，2026-09-28 新增）：
     - 基底场 V_base：去掉全部雾域站（实测雾站、虚拟雾站、第 7 步的域内成员）后，用原 IDW 插出；
     - 雾场 V_fog：只用本影响域内的实测雾站和虚拟雾站做各向异性 IDW；
     - 影响域内一律取 g = 1，不做垂直衰减。山谷掩膜来自平滑 DEM，原始 DEM 在谷内局部高出雾顶，原先有 6%–20% 的域内格点因此没有被填充，形成斑点；
     - 合成在对数空间进行：V = V_fog^g · V_base^(1 − g)，能见度下限取 10 m。雾 200 m、基底 5 km 时，1 km 等值线约落在 g ≈ 0.5 处，与 g 场图中看到的边缘一致；线性混合要 g > 0.83 才低于 1 km，雾区边缘会内缩；
     - 输出的 `fog_influence` 同样按域内 g = 1 给出，与实际填充一致。
   - `blend_depth`（2026-09-28 新增，候选）：在 `blend` 的基础上，让域内能见度随深度渐变，谷底最低，向谷缘升高：
     - 相对深度 r = (雾顶 − z) / (雾顶 − 谷底)，截断到 [0, 1]。z 为 DEM 海拔，谷底取域内 DEM 海拔的 5% 分位数；
     - 雾站深度 r_s 用站点所在格点的 DEM 海拔计算，不用站点海拔。两者可相差上百米（如德庆站海拔 129 m，所在格点 DEM 为 27 m），混用会使雾站显得偏浅、谷底被过度外推。r_s 插值到格点得到 r̄（下限 0.1）；
     - 对数空间沿深度线性变化：V = V_edge · (V_fog / V_edge)^(r / r̄)，其中 V_edge = max(`visThresholdM`, V_fog)。在雾站深度处等于雾站插值，在雾顶处（包括域内高出雾顶的格点）升到 1000 m，更深处继续降低；指数上限为 2，能见度下限为 10 m；
     - 域外的对数混合与 `blend` 相同。影响域边界处的值约为 `visThresholdM`，因此边缘依旧清晰。
   - `blend_linear`：与 `blend` 相同的基底场和雾场，但做线性混合 V = g·V_fog + (1 − g)·V_base，并且域内保留垂直衰减。雾区比 `blend` 略小，谷内有少量斑点。
   - `weight`（最初方案）：IDW 中格点对雾站的权重乘以 g，其余站不变。

   改为 `blend` 的原因：在 `weight` 模式下，只限制了雾站不向外扩散，没有阻止谷外的好站影响谷内格点。影响域内格点的最近 12 个候选站中大多是谷外的好站，雾站只能在自身附近形成很小的"靶心"。以 09-21 07 时为例，5 个影响域内被填成 < 1 km 的格点只占 1%–16%。

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
  "visThresholdM": 1000,
  "rhThresholdPct": 95,
  "precipThresholdMm": 0,
  "precipMissingAsDry": true,
  "sigmaZM": 50,
  "sigmaDKm": 2,
  "gCutoff": 0.05,
  "inferRadiusKm": 50,
  "fillMode": "blend"
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

`--time` 为 UTC 时次。可覆盖的参数：`--fill-mode`（`blend` / `blend_depth` / `blend_linear` / `weight`）、`--vis-threshold`、`--rh-threshold`、`--precip-threshold`、`--precip-missing-as-wet`、`--sigma-z`、`--sigma-d`、`--g-cutoff`、`--infer-radius`、`--valley`、`--source`。

输出目录为 `output/fog_compare/<时次>/<参数标签>/`，参数标签默认由填充模式和参数拼成，例如 `blend_v1000_rh95_sz50_sd2_g0.05_r50`。注意：只跑单条路径（`--source`）时，会覆盖同一目录下的 `stats.csv`。

| 文件 | 内容 |
|---|---|
| `<路径>_compare.png` | 原算法、订正结果、g 场三联图，叠加山谷边界、实测雾站（红点）和虚拟雾站（紫色三角） |
| `<路径>_compare_zoom.png` | 同上，自动放大到影响域范围（外扩 0.3°） |
| `<路径>.nc` | `visibility`、`visibility_original`、`fog_influence` |
| `<路径>_stations.csv` | 订正后站点表，含 `vis_original` |
| `stats.csv` | 雾站数、影响域数、广东境内能见度 < 500 m 和 < 1 km 的面积（订正前后） |
| `settings.txt` | 本次使用的完整参数 |

**多模式同图对比**：先按各 `--fill-mode` 分别跑完上面的对比，再加 `--plot-modes`。这一步不重新计算，只读取已有的 `<路径>.nc`，把原算法、`weight`、`blend_linear`、`blend`、`blend_depth` 画成一行五幅（自动放大到影响域，标题里标注广东境内 < 1 km 的面积）：

```bash
uv run python -m src.business.fog_compare --time 202609202300 202609212200 202609212300 --plot-modes
```

- 每个时次输出 `output/fog_compare/<时次>/modes_<参数>.png`，两行分别是 `national` 和 `national_and_regional` 路径。
- 给多个时次时，还会另外输出汇总图 `output/fog_compare/modes_<参数>_<首时次>-<末时次>.png`。
- `weight` 模式会兼容早期不带模式名的目录（`<时次>/v1000_.../`）。缺某个模式时，对应子图留空并给出提示。
- `--time` 可以一次给多个时次，普通对比也会逐个运行。

调参方向：

| 参数 | 调大的效果 |
|---|---|
| `sigmaDKm` | 雾区水平边缘更宽、过渡更缓 |
| `sigmaZM` | 雾区沿坡向上延伸得更高 |
| `gCutoff` | 远处更早完全规避 |
| `rhThresholdPct` / `inferRadiusKm` | 控制推断起雾（虚拟雾站）的多少 |

---

## 6. 已完成的验证

- **单元测试**：`uv run python -m unittest discover -s tests -p "test_*.py"`，共 52 个测试，全部通过。新增测试覆盖：
  - `blend` 模式的域内填充、域外保持基底场、边缘按 g 混合，以及 `fillMode` 切换；
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
  - 对比图位于 `output/fog_compare/202502280000/v500_rh95_sz50_sd2_g0.05_r50/`（该个例使用的是旧阈值 500 m）。

- **2026 年 9 月三个时次**（`visThresholdM = 1000`，其余参数为默认值；数据来自接口，已缓存）。下表为广东境内 < 1 km 的面积：

  | 北京时 | UTC 目录 | 实测雾站 | 虚拟雾站 | 影响域 | 路径 | 原算法 | weight | blend_linear | blend | blend_depth |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 09-21 07 时 | `202609202300` | 3 | 4 | 5 | national | 326 | 284 | 2397 | 3467 | 2847 km² |
  | | | | | | national_and_regional | 391 | 377 | 2489 | 3576 | 2938 km² |
  | 09-22 06 时 | `202609212200` | 2 | 5 | 6 | national | 82 | 198 | 1323 | 1979 | 1810 km² |
  | | | | | | national_and_regional | 160 | 291 | 1402 | 2054 | 1886 km² |
  | 09-22 07 时 | `202609212300` | 2 | 2 | 4 | national | 67 | 107 | 1917 | 3168 | 2402 km² |
  | | | | | | national_and_regional | 79 | 118 | 1945 | 3251 | 2402 km² |

  `blend_depth` 与 `blend` 相比，影响域内（national_and_regional）能见度的分位数（m）：

  | 北京时 | 模式 | 5% | 25% | 50% | 75% | 95% |
  |---|---|---|---|---|---|---|
  | 09-21 07 时 | blend | 100 | 100 | 200 | 200 | 200 |
  | | blend_depth | 40 | 96 | 265 | 691 | 1000 |
  | 09-22 06 时 | blend | 200 | 200 | 400 | 400 | 400 |
  | | blend_depth | 173 | 312 | 547 | 1000 | 1000 |
  | 09-22 07 时 | blend | 10 | 10 | 100 | 100 | 100 |
  | | blend_depth | 10 | 59 | 162 | 849 | 1000 |

  局部放大图见 `output/fog_compare/202609202300/blend_depth_detail_pingyuan.png`（梅州平远一带，blend、blend_depth 与 DEM 并排）。

  对比图目录：
  - `blend` 结果：`output/fog_compare/<UTC 目录>/blend_v1000_rh95_sz50_sd2_g0.05_r50/`；
  - `blend_linear` 结果：`output/fog_compare/<UTC 目录>/blend_linear_v1000_rh95_sz50_sd2_g0.05_r50/`；
  - `weight` 结果：`output/fog_compare/<UTC 目录>/v1000_rh95_sz50_sd2_g0.05_r50/`（该目录生成时标签中尚未包含模式名，且没有放大图）；
  - 09-21 07 时另有带放大图的 `weight_v1000_...`，可直接与 `blend_v1000_...` 对照。

  在 `blend` 模式下，雾区按山谷形状填满，边缘与山谷边界吻合，低能见度面积因此明显增大。这些山谷是否真的整谷起雾，需要对照卫星云图判断。如果判断偏大，可调整的方向有：减小 `sigmaDKm` 或 `sigmaZM`（收窄边缘）；提高 `rhThresholdPct`（减少虚拟雾站）。

  以下两点在 `weight` 模式下就已存在，`blend` 模式下更明显：**虚拟雾站会填充原本没有低能见度的山谷**。例如：
  - 连山县福堂镇天鹅湖站（湿度 96%，海拔 368 m），原算法估算约 25 km，订正后被判为虚拟雾站，取 400 m 或 100 m；
  - 郁南桂圩站、英德九龙站等，原算法估算 2.5–5.6 km，订正后为 0–400 m。

  这些是否真的起雾，需要对照卫星云图或当地实况来判断。如果判断偏多，可以提高 `rhThresholdPct`（如 98）或缩小 `inferRadiusKm`。

  另外，德庆国家基本气象站（59269）在 09-22 07 时的能见度为 0 m，可能是异常值，需要核实。它作为实测雾站，使同一时次的郁南桂圩虚拟雾站也被赋值为 0 m（50 km 内的雾站中位数）；在 `blend` 模式下，这两个山谷整谷被填为 0 m。

  龙川县气象局（59107）在 09-21 07 时的能见度为 200 m，但湿度只有 69%，不太像辐射雾，却被判为实测雾站，所在山谷整谷被填充。可以考虑给实测雾站也加上湿度下限（目前只有虚拟雾站有湿度条件）。

---

## 7. 尚未完成 / 已知局限

1. **个例验证不足**：只检验了 4 个时次，尚未与卫星可见光云图对照；需要补充更多冬季辐射雾个例，以及降水导致低能见度的个例（后者应基本不被改动）。
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
| `tests/test_radiation_fog.py` | 新增：15 个测试 |
| `src/business/algorithms.py` | 修改：估算向量化并支持权重修正；保留 `_estimate_one_legacy()`；新增 `reference_sets()` |
| `src/business/idw.py` | 修改：可选参数 `fog_influence` |
| `src/business/pipeline.py` | 修改：接入订正流程，出错时退回原算法 |
| `src/business/config.py` | 修改：`RadiationFogSettings` |
| `src/business/api.py` | 修改：可选字段 `V13019 → pre_1h` |
| `src/valley_boundary.py` | 修改：`valley_product()`、`export_valley_product()`、`--product` |
| `tests/test_valley_boundary.py` | 修改：新增 3 个测试 |
| `src/config/business.config.example.json`、`business.docker.config.example.json` | 修改：新增 `radiationFog` 配置段 |
| `handoff/radiation_fog_valley.md` | 修改：状态更新，新增第 5 节 |
