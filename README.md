# 能见度插值算法与 DEM 处理工具

本项目是一个高性能的地理空间数据插值与处理工具，专注于**气象能见度分析**与**数字高程模型（DEM）处理**两大方向：

- **能见度插值**：从稀疏的站点观测出发，结合地形（DEM）信息，通过湿度-距离双约束重构 + 各向异性三维 IDW 插值，生成高分辨率（0.01°×0.01°）能见度连续场。
- **DEM 处理**：将高分辨率 DEM 数据稀疏化到 0.01°×0.01° 标准网格，实现数据压缩和地形预处理。

## 项目背景与解决的问题

1. 全省能见度站点仅约 100 个，比其他要素站点（约 4000 个）小一个数量级，观测极其稀疏。
2. 能见度是非线性变化要素，与地形、下垫面特征高度相关，简单空间插值难以反映真实能见度变化。

## 项目亮点

1. **站点能见度智能重构**：利用有湿度观测的站点，采用湿度-距离双约束最近邻权重分解策略，通过反平方距离衰减函数对气象要素相似性与空间邻近性解耦建模，实现上千个缺测站点的能见度估算。
2. **地形感知的各向异性插值**：引入垂直维度增强因子（β=10）对地形效应进行非线性放大，将离散观测映射至高分辨率连续场，显著改善山区低能见度事件的插值精度。
3. **1,364 倍性能加速**：完全向量化 + 批处理 + 多进程并行，50,000 格点插值由 300 秒降至 0.22 秒。

## 完整数据流

下图展示从原始数据到最终可视化/评估的完整链路：

```mermaid
flowchart TD
    subgraph S1["① 地形数据（DEM）处理"]
        A1[ASTGTM2 高分辨率DEM GeoTIFF 瓦片<br/>h:/data/DEM/*.tif] --> A2[dem_interpolation.py<br/>单文件/批处理下采样至 0.01°]
        A2 --> A3[0.01° NetCDF 瓦片<br/>h:/data/DEM/netcdf_output/]
        A3 --> A4[create_unified_grid 边界优化合并<br/>填补间隙 + 高斯平滑]
        A4 --> A5[merged_dem_data.nc<br/>h:/data/DEM/merged_dem_data.nc]
    end

    subgraph S2["② 站点能见度估算"]
        B1[国家站观测 CSV<br/>内网共享 SurfAuto_广东_*.csv] --> B2[get_vis_estimated_by_rh.py<br/>基于RH相似性加权插值]
        B2 --> B3[station_vis_all_estimated_{time}.csv<br/>data/vis_estimated_base_nation_station/]
        B3 -.-> B4[并入区域站<br/>station_vis_all_estimated_{time}_national_and_regional.csv]
    end

    subgraph S3["③ 各向异性 IDW 空间插值"]
        C1[站点估算 CSV<br/>national / national_and_regional] --> C2[vis_dem_dis.py<br/>create_visibility_grid<br/>β=10, power=2, n_neighbors=6]
        A5 --> C2
        C2 --> C3[visibility_anisotropic_idw_{time}.nc<br/>data/idw_nc/...]
    end

    subgraph S4["④ 调试可视化"]
        D1[CLDAS 5km 能见度产品<br/>thredds 内网服务 VIS_*.NC] --> D2
        C3 -. 默认本地IDW路径 .-> D2[debug_visibility_visualization.py<br/>load_visibility_data 单位转换 m→km]
        D2 --> D3[apply_guangdong_mask<br/>广东省省界 shapefile 遮罩]
        D3 --> D4[create_comprehensive_visualization<br/>2×3 六子图对比]
        D4 --> D5[debug_visibility_comprehensive.png<br/>或 cldas_..._{time}.png]
    end

    subgraph S5["⑤ 模型评估"]
        C3 --> E1[evaluate_visibility_model.py<br/>与 CLDAS 5km 产品对比验证]
        D1 --> E1
        E1 --> E2[cldas_score / model_score 评分 CSV]
    end
```

**数据流各环节一览**：

| 环节 | 脚本 | 输入 → 输出 |
|---|---|---|
| ① DEM 处理 | `src/dem_interpolation.py` | 高分辨率 GeoTIFF 瓦片 → `merged_dem_data.nc`（0.01° 网格，含边界优化） |
| ② 能见度估算 | `src/get_vis_estimated_by_rh.py` | 国家站观测 CSV（网络共享）→ 各站估算能见度 CSV |
| ③ IDW 插值 | `src/vis_dem_dis.py` | 站点 CSV + DEM 海拔 → `visibility_anisotropic_idw_{time}.nc` |
| ④ 调试可视化 | `debug_visibility_visualization.py` | **CLDAS 在线数据 或 本地 IDW NC** + 广东边界 → 综合对比图 |
| ⑤ 模型评估 | `src/evaluate_visibility_model.py` | IDW 结果 vs CLDAS 产品 → 评分 CSV |

> **调试可视化数据来源说明**：`debug_visibility_comprehensive.png` 的直接数据来源取决于 `load_visibility_data()` 传入参数——当前 `main()` 传入的是**内网 CLDAS 5km 能见度产品 URL**（`http://10.148.8.71:7080/thredds/dodsC/cldas/{YYYYMMDD}/VIS_{YYYYMMDDHH}.NC`）；若使用默认参数，则来源为**本地各向异性 IDW 插值结果**（`visibility_anisotropic_idw.nc`）。图中的广东省边界遮罩来自省界 Shapefile。

## 安装依赖

```bash
pip install rioxarray xarray numpy matplotlib cartopy scipy
```

## 使用方法

### 方法1: 运行主程序

```bash
python main.py
```

然后选择相应的操作选项。

### 方法2: 直接运行插值脚本

```bash
python src/dem_interpolation.py
```

### 方法3: 使用Jupyter Notebook

打开 `notebook/dem_interpolation_demo.ipynb` 进行交互式操作。

## 输入文件

- **DEM文件**: 支持GeoTIFF格式的DEM数据
- **文件路径**: 默认路径为 `h:\data\DEM\ASTGTM2_N23E111_dem.tif`
- **坐标系统**: 支持地理坐标系统（经纬度）

## 输出文件

- **`dem_comparison.png`**: 原始数据和插值后数据的对比图
- **`interpolated_dem_0.01deg.nc`**: 插值后的DEM数据（NetCDF格式）

## 处理流程

1. **加载DEM数据**: 使用rioxarray读取GeoTIFF文件
2. **创建目标网格**: 生成0.01°×0.01°的规则网格
3. **执行插值**: 使用scipy.interpolate.griddata进行插值
4. **可视化对比**: 生成对比图显示处理效果
5. **保存结果**: 将插值后的数据保存为NetCDF格式

## 数据压缩效果

- **原始数据**: 通常为高分辨率（如3601×3601像素）
- **插值后数据**: 0.01°×0.01°网格，数据量显著减少
- **压缩比**: 通常可达到10-100倍的压缩比
- **精度保持**: 在保持地形特征的同时实现数据稀疏化

## 插值方法说明

- **linear**: 线性插值，计算速度快，适合大部分情况
- **nearest**: 最近邻插值，保持原始值，计算最快
- **cubic**: 三次样条插值，精度最高但计算量最大

## 示例输出

```
=== DEM数据稀疏化处理 ===
输入文件: h:\data\DEM\ASTGTM2_N23E111_dem.tif

DEM数据已成功加载:
数据形状: (1, 3601, 3601)
数据大小: 26.00 MB

目标网格信息:
经度范围: 111.0000° - 112.0000°
纬度范围: 23.0000° - 24.0000°
网格大小: 100 × 100
网格分辨率: 0.01°

开始插值...
原始数据点数: 12967201
目标网格点数: 10000
插值完成!
插值后数据大小: 0.08 MB
数据压缩比: 325.00

=== 处理完成 ===
原始数据大小: 26.00 MB
插值后数据大小: 0.08 MB
数据压缩比: 325.00
```

## 注意事项

1. **内存使用**: 处理大型DEM文件时可能需要较多内存
2. **文件路径**: 确保DEM文件路径正确且文件存在
3. **坐标系统**: 确保输入数据使用地理坐标系统
4. **插值方法**: 根据精度要求选择合适的插值方法

## 核心处理流程

### 能见度处理完整流程

```
第一步：能见度估算（get_vis_estimated_by_rh.py）
├── 输入：国家站 + 区域站 CSV 文件
├── 处理：基于相对湿度相似性的加权插值（湿度-距离双约束）
└── 输出：所有站点的能见度估算值 CSV

第二步：各向异性 IDW 插值（vis_dem_dis.py）
├── 输入：站点 CSV + DEM NetCDF
├── 处理：地形感知的各向异性插值（β=10）
└── 输出：能见度网格（NetCDF）

第三步：模型评估（evaluate_visibility_model.py）
├── 输入：插值结果 + 参考数据（CLDAS）
├── 处理：统计对比分析
└── 输出：评估指标报告
```

### DEM 处理流程

```
单文件模式（dem_interpolation.py）
├── 加载 GeoTIFF
├── 创建 0.01° 目标网格
├── 插值到目标网格
└── 保存为 NetCDF

批处理模式（dem_interpolation.py batch）
├── 扫描目录中的 ASTGTM2_NyyExxx_dem.tif 文件
├── 从文件名解析经纬度
├── 转换每个瓦片为 NetCDF
├── 分析边界间隙/重叠
├── 应用边界优化（填充 + 平滑）
└── 合并为统一 NetCDF
```

## 核心技术模块

### 1. 站点能见度估算（src/get_vis_estimated_by_rh.py）
为无能见度观测的区域站估算能见度：读取国家站（有能见度）与区域站（有相对湿度），使用 `BallTree` 进行 KNN 搜索，综合「湿度相似性权重」与「空间距离权重」双重加权：

$$vis = 0.5 \times vis_{rh} + 0.5 \times vis_{dis}$$

- 湿度权重：$w_{rh,i} = 1/(|rh - rh_i|)^2$
- 距离权重：$w_{dis,i} = 1/distance_i^2$
- 支持多进程并行处理时间序列

### 2. 格点化各向异性插值（src/vis_dem_dis.py）
将站点能见度插值到 0.01° 格点并融合地形，各向异性距离函数：

$$d_i = \sqrt{(x_i-x_0)^2 + (y_i-y_0)^2 + \beta^2 (z_i-z_0)^2}, \quad \beta=10$$

权重 $w_i = 1/d_i^p$（$p=2$），插值结果 $vis = \sum w_i \cdot vis_i / \sum w_i$。性能经三阶段优化达到 **1,364 倍加速**（50,000 点：300 秒 → 0.22 秒）。

### 3. 数字高程模型处理（src/dem_interpolation.py）
高分辨率 DEM 降采样至 0.01°×0.01° 标准网格，支持 `vectorized`（推荐）/`fast_nearest`/`block_average`/`linear`/`nearest`/`cubic` 多种方法；批处理自动扫描瓦片并采用**三阶段边界优化**（间隙检测 → 智能填充 → 高斯平滑）实现无缝合并。

### 4. 模型评估（src/evaluate_visibility_model.py）
与国家级 5km CLDAS 能见度产品及站点观测进行多源交叉验证，输出 MAE / RMSE / 相关系数等指标至 `data/model_score/`。

### 5. 地形位置指数分析（src/tpi_ridge_valley.py）
基于 TPI 的山脊/山谷自动识别，支持分位数 / Z-score / MAD 三种阈值方案。

## 数据源与输出路径

### 数据源

| 数据类型 | 路径 | 格式 |
|---|---|---|
| 国家站数据 | `\\10.148.44.81\surf\idea\getSurfAutoOrg4Prov` | CSV |
| 区域站数据 | `\\10.148.44.81\surf\idea\getSurfAwstOrg4Prov` | CSV |
| CLDAS 能见度产品 | `http://10.148.8.71:7080/thredds/dodsC/cldas/` | NetCDF |
| DEM 输入 | `h:\data\DEM\` | GeoTIFF |
| 广东省边界 | `D:\Document\气象台\GIS\...\广东省_省界.shp` | Shapefile |

- 国家站：`\\10.148.44.81\surf\idea\getSurfAutoOrg4Prov\{YYYY}\{MM}\SurfAuto_广东_{YYYYMMDDHHmm}00.csv`
- 区域站：`\\10.148.44.81\surf\idea\getSurfAwstOrg4Prov\{YYYY}\{MM}\SurfAwst_广东_{YYYYMMDDHHmm}00.csv`
- CLDAS 产品：`http://10.148.8.71:7080/thredds/dodsC/cldas/{YYYYMMDD}/VIS_{YYYYMMDDHH}.NC`

> **TODO：更换国家局 5 km 能见度产品数据源。** 当前产品的主要延迟来自上游数据源，现有绘图与调度逻辑无法从根本上消除该延迟；后续需要接入时效性更稳定的数据源，并同步调整产品地址、读取方式和出图验证流程。

### 输出路径

| 输出类型 | 路径 | 格式 |
|---|---|---|
| 估算站点能见度 | `data/vis_estimated_base_nation_station/` | CSV |
| IDW 插值结果 | `data/idw_nc/` | NetCDF |
| 合并 DEM | `h:\data\DEM\merged_dem_data.nc` | NetCDF |
| 模型评估结果 | `data/model_score/`、`data/cldas_score/` | CSV |
| 调试可视化 | `debug_visibility_comprehensive*.png` | PNG |

## 测试

```bash
uv run test_idw_optimization.py               # 性能测试
uv run test_dem_interpolation.py              # DEM 功能测试
uv run test_visibility_interpolation.py       # 插值正确性测试
uv run test_boundary_optimization.py          # 边界优化测试
uv run test_regional_evaluation.py            # 区域评估测试
```

### 业务化定时流程

业务化入口位于 `src/business/`，从接口获取广东、广西、湖南、江西、福建、海南六省的国家站和区域站数据，合并后同时生成两路能见度估算CSV与IDW NetCDF。账号密码从 `src/config/local.config.json` 读取，DEM路径可通过参数覆盖。

#### 运行前准备

请在项目根目录执行命令，并确认项目依赖、接口账号和DEM文件均已准备好。使用 `uv` 安装依赖：

```bash
uv sync
```

账号配置文件 `src/config/local.config.json` 格式如下：

```json
{
  "userId": "username",
  "pwd": "password"
}
```

服务器部署建议复制 [business.config.example.json](src/config/business.config.example.json) 为独立配置文件，并按服务器实际目录修改：

```json
{
  "userId": "username",
  "pwd": "password",
  "dataRoot": "D:/vis_interpolate/data",
  "demPath": "D:/vis_interpolate/data/DEM/merged_dem_data.nc",
  "csvNationalRoot": "vis_estimated_base_nation_station",
  "csvCombinedRoot": "vis_estimated_base_nation_and_regional_station",
  "ncNationalRoot": "idw_nc/national",
  "ncCombinedRoot": "idw_nc/national_and_regional",
  "visImgRoot": "vis_img",
  "guangdongBoundaryPath": "D:/vis_interpolate/data/boundary/广东省_省界.shp",
  "statePath": "business/pipeline_state.sqlite",
  "lockPath": "business/pipeline.lock",
  "logPath": "business/business.log",
  "timeoutSeconds": 15,
  "retries": 2,
  "requestConcurrency": 6,
  "sourceReadyDelayMinutes": 2,
  "latestFirst": true,
  "maxBackfillSlotsPerCycle": 1,
  "asyncPlots": true
}
```

`dataRoot` 是统一数据根目录；CSV、NetCDF、状态库、锁文件和日志的相对路径均相对于它。`demPath` 的相对路径相对于项目根目录，生产环境建议使用绝对路径。默认DEM路径为 `h:\\data\\DEM\\merged_dem_data.nc`，也可以通过 `--dem-path` 临时覆盖。

单轮执行也可以指定历史时间、配置文件、DEM路径和省份：

```bash
# 使用当前时间执行
uv run python -m src.business run-once

# 指定时间补跑
uv run python -m src.business run-once --now 2026-08-05T08:02:00+08:00

# 指定配置文件和DEM路径
uv run python -m src.business run-once \
  --config src/config/local.config.json \
  --dem-path h:\\data\\DEM\\merged_dem_data.nc

# 临时指定六省（支持英文逗号或中文逗号）
uv run python -m src.business run-once --provinces 广东,广西,湖南,江西,福建,海南

# 兼容单省份执行
uv run python -m src.business run-once --province 广西
```

常驻模式会持续运行，在每小时 `02、07、12、17、22、27、32、37、42、47、52、57` 分启动一轮处理。默认距离上游时次 2 分钟后开始处理，最新就绪时次优先，最多补偿 1 个旧时次；图片进入后台单线程队列生成，不阻塞 CSV 和 NetCDF 发布。部署到Windows服务或任务计划时，建议保持 `serve` 进程常驻；手动停止可使用 `Ctrl+C`。

#### 使用 PM2 持久化守护

项目根目录提供 [ecosystem.config.cjs](H:\github\python\vis_interpolate\ecosystem.config.cjs)，用于守护业务常驻进程。首次部署时先完成依赖安装，再启动PM2：

```bash
# 安装PM2（只需执行一次）
npm install --global pm2

# 安装Python依赖
uv sync

# 启动业务守护进程
pm2 start ecosystem.config.cjs

# 查看进程状态和实时日志
pm2 status
pm2 logs vis-interpolate-business
```

常用运维命令：

```bash
# 重启、停止、删除守护进程
pm2 restart vis-interpolate-business
pm2 stop vis-interpolate-business
pm2 delete vis-interpolate-business

# 保存当前进程清单，便于PM2重启后恢复
pm2 save
```

PM2标准输出日志和错误日志分别写入 `output/pm2-business-out.log`、`output/pm2-business-error.log`；业务处理日志写入配置文件中的 `logPath`（未配置时为 `data/business/business.log`）。PM2启动时工作目录为项目根目录，配置会自动选择Windows或Linux虚拟环境中的Python路径。

服务器使用PM2时，可通过 `VIS_BUSINESS_CONFIG` 选择服务器配置文件。PowerShell示例：

```powershell
$env:VIS_BUSINESS_CONFIG = 'D:\vis_interpolate\config\business.config.json'
pm2 start ecosystem.config.cjs
pm2 save
```

也可以在 `ecosystem.config.cjs` 的 `env.VIS_BUSINESS_CONFIG` 中直接填写固定配置路径。

运行结果和状态文件位置如下：

- 估算CSV：由 `csvNationalRoot`、`csvCombinedRoot` 配置，默认位于 `data/vis_estimated_base_nation_station/YYYY/MM/DD/`、`data/vis_estimated_base_nation_and_regional_station/YYYY/MM/DD/`。
- IDW NetCDF：由 `ncNationalRoot`、`ncCombinedRoot` 配置，默认位于 `data/idw_nc/national/YYYY/MM/DD/`、`data/idw_nc/national_and_regional/YYYY/MM/DD/`。
- 广东省遮罩图片：由 `visImgRoot` 配置，默认位于 `data/vis_img/YYYY/MM/DD/`，文件名为 `visibility_national_*.png` 和 `visibility_national_and_regional_*.png`。

如需只对已有NetCDF重新绘图，可直接运行绘图模块：

```bash
uv run python -m src.business.plot \
  --nc data/idw_nc/national/2026/08/05/visibility_anisotropic_idw_202608050800.nc \
  --boundary D:/vis_interpolate/data/boundary/广东省_省界.shp \
  --output data/vis_img/2026/08/05/visibility_national_202608050800.png
```

如果配置的 `guangdongBoundaryPath` 不存在，业务数据仍会正常输出，但该资料时次会在日志中提示未生成图片；服务器部署时应把该边界文件路径配置正确。
- 样本计数、锁文件和运行日志：分别由 `statePath`、`lockPath`、`logPath` 配置。

常驻调度在每小时 `02、07、12、17、22、27、32、37、42、47、52、57` 分启动；每轮扫描世界时前30分钟内、达到 `sourceReadyDelayMinutes` 的5分钟资料时次，先处理最新时次，再从 `observation_queue` 补偿旧时次。业务状态、补偿队列和阶段指标保存在 `data/business/pipeline_state.sqlite`，运行日志保存在 `data/business/business.log`。日志中的 `api_seconds`、`parse_seconds`、`estimate_seconds`、`idw_seconds`、`publish_seconds`、`plot_submit_seconds`、`source_update_time_utc` 和 `data_delay_seconds` 用于统计真实延迟。输出文件按 `YYYY/MM/DD` 分层保存到两个估算CSV目录和 `data/idw_nc/national`、`data/idw_nc/national_and_regional` 目录。

## 项目结构

```
vis_interpolate/
├── main.py                                  # CLI 交互式入口
├── src/                                     # 核心模块
│   ├── dem_interpolation.py                 # DEM 处理
│   ├── vis_dem_dis.py                       # 能见度插值
│   ├── get_vis_estimated_by_rh.py           # 能见度估算
│   ├── evaluate_visibility_model.py         # 模型评估
│   ├── tpi_ridge_valley.py                  # 地形分析
│   └── plot_result.py                       # 可视化工具
├── data/                                    # 数据目录
│   ├── vis_estimated_base_nation_station/   # 国家站估算结果
│   ├── vis_estimated_base_nation_and_regional_station/
│   ├── idw_nc/                              # IDW 插值输出
│   ├── model_score/                         # 模型评估结果
│   └── cldas_score/                         # CLDAS 对比评分
├── test_*.py                                # 测试套件
├── debug_visibility_visualization.py        # 调试可视化
├── notebook/                                # Jupyter 演示
├── pyproject.toml                           # 项目配置
├── README.md, overview.md                   # 项目概览
└── PERFORMANCE_OPTIMIZATION.md              # 性能优化报告
```

## 技术依赖

- **xarray / numpy / pandas**：多维数组与数据操作
- **scipy / scikit-learn**：插值算法与 KNN（BallTree）
- **rasterio / rioxarray**：栅格 I/O
- **cartopy / matplotlib**：地理可视化
- **geopandas / shapely**：矢量地理空间（广东省遮罩）
- **numba**：JIT 编译支持

## 性能与创新亮点

- **1,364 倍性能提升**：50,000 点插值从 300 秒降至 0.22 秒
- **百万级格点处理**：7 进程并行约 0.64 秒
- **各向异性 IDW**：地形感知插值，山区误差减少约 66%
- **双重加权估算**：湿度 + 距离协同，精度提升约 12%
- **智能边界处理**：DEM 瓦片 NaN 像素 100% 消除
- **三级评估体系**：站点-区域-格点交叉验证
