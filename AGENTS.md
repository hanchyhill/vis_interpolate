# Repository Guidelines

## 项目结构

- `src/`：DEM 下采样、能见度估算、各向异性 IDW、模型评估及地形分析脚本。
- `src/business/`：业务化采集与处理流水线，入口为 `python -m src.business`。
- `src/config/`：业务配置示例和本地/服务器配置；`tests/`：业务流水线测试。
- `notebook/`：交互式示例；`data/`、`output/`：输入缓存和生成结果，通常不应提交大文件。
- 根目录的 `main.py`、`debug_visibility_visualization.py` 和 `test_*.py` 保留用于旧流程、调试和性能验证。

## 安装、运行与测试

项目要求 Python `>=3.13`，推荐使用 uv：

```bash
uv sync
uv run main.py                         # 交互式 DEM 工具
uv run src/dem_interpolation.py batch  # DEM 批处理与合并
uv run python -m src.business           # 业务化能见度流水线
uv run python -m unittest discover -s tests -p "test_*.py"
```

修改插值或性能相关代码时，可额外运行对应根目录脚本，如 `uv run python test_idw_optimization.py`。涉及真实 DEM、站点或 CLDAS 数据的命令需要本地数据源和较多内存，不应在无数据环境中强行执行。

## 编码规范

遵循 PEP 8，使用 4 空格缩进；函数、变量和模块使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`。优先复用现有 NumPy/xarray 向量化实现，避免在大网格处理中新增逐点 Python 循环。公共函数应保持清晰的参数、返回值和单位约定，并在涉及经纬度/海拔时注明坐标系。

## 测试要求

测试文件命名为 `test_*.py`，测试方法命名为 `test_<行为>`。新增算法应覆盖正常输入、边界/缺测数据及单位转换；改动绘图或 NetCDF 输出时，检查文件是否生成、维度和关键属性是否正确。提交前运行完整 `unittest` 发现命令及受影响的专项脚本。

## 提交与 Pull Request

历史提交采用简短的 Conventional Commits 前缀（如 `feat:`、`更新`、`新增`），提交主题应具体说明行为变化。PR 应说明目的、影响模块、输入输出变化和验证命令；涉及图表或可视化时附示例截图。若修改配置或数据处理路径，请说明兼容性、资源需求及回滚方式。

## 配置与数据安全

不要提交真实站点数据、生成的 NetCDF/PNG 大文件、内网 URL 凭据或个人路径。使用 `src/config/business.config.example.json` 作为模板，个人配置保存在本地；提交前确认路径、网络共享和数据脱敏情况。
