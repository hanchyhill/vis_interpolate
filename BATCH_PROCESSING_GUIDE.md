# DEM批量处理和合并指南

## 功能概述

本工具新增了批量处理DEM文件的功能，可以：
1. 遍历指定目录中的所有DEM TIF文件
2. 使用 `interpolate_dem_to_grid` 函数将每个文件转换为NetCDF格式
3. 将所有NetCDF文件合并为单个大文件，实现更大范围的数据覆盖

## 支持的文件格式

- **输入格式**: `ASTGTM2_NyyExxx_dem.tif`
  - `yy`: 纬度（如23表示北纬23度）
  - `xxx`: 经度（如111表示东经111度）
- **输出格式**: NetCDF (.nc)

## 使用方法

### 方法1: 通过主程序

```bash
python main.py
```

选择选项 2 进行批量处理。

### 方法2: 直接运行批量处理

```bash
python src/dem_interpolation.py batch
```

### 方法3: 使用示例脚本

```bash
python batch_example.py
```

## 目录结构

```
h:\data\DEM\
├── ASTGTM2_N23E111_dem.tif      # 输入DEM文件
├── ASTGTM2_N23E112_dem.tif      # 输入DEM文件
├── ASTGTM2_N24E111_dem.tif      # 输入DEM文件
├── ...
├── netcdf_output\               # 临时NetCDF文件目录
│   ├── ASTGTM2_N23E111_dem_interp_0.01deg.nc
│   └── ...
└── merged_dem_data.nc           # 最终合并文件
```

## 处理流程

1. **文件发现**: 扫描输入目录，找到所有符合格式的DEM文件
2. **文件名解析**: 从文件名中提取经纬度信息
3. **数据加载**: 使用 `rioxarray` 加载每个TIF文件
4. **网格创建**: 为每个文件创建0.01°×0.01°的目标网格
5. **数据插值**: 使用向量化方法进行快速插值
6. **NetCDF保存**: 将插值结果保存为NetCDF格式
7. **边界分析**: 自动分析数据集间的边界缺口和重叠情况
8. **边界优化**: 使用智能算法填充边界缺失值
9. **数据合并**: 将所有NetCDF文件合并为单个连续文件
10. **质量验证**: 输出处理统计和质量报告
11. **清理**: 可选择删除临时文件

## 边界优化功能

### 问题识别
- 自动检测DEM文件间的边界缺口
- 识别重叠区域和数据不连续性
- 分析缺失值分布模式

### 优化算法
- **线性插值填充**: 使用周围有效数据进行线性插值
- **最邻近插值**: 对复杂区域使用最邻近方法
- **邻域平均**: 迭代式邻域平均填充
- **边界平滑**: 高斯滤波平滑数据边界过渡
- **重叠处理**: 重叠区域使用加权平均

## 配置参数

### 基本参数

- `resolution`: 目标网格分辨率（默认0.01度，约1.1公里）
- `method`: 插值方法（推荐'vectorized'，最快）
- `chunk_size`: 分块处理大小（用于内存管理）

### 路径设置

```python
input_dir = r'h:\data\DEM'                    # 输入目录
output_dir = r'h:\data\DEM\netcdf_output'     # 临时输出目录
final_output_file = r'h:\data\DEM\merged_dem_data.nc'  # 最终文件
```

## 性能优化

### 插值方法比较

| 方法 | 速度 | 内存使用 | 推荐度 |
|------|------|----------|--------|
| `vectorized` | 最快 | 中等 | ⭐⭐⭐⭐⭐ |
| `fast_nearest` | 快 | 低 | ⭐⭐⭐⭐ |
| `block_average` | 中等 | 低 | ⭐⭐⭐ |
| `linear` | 慢 | 高 | ⭐⭐ |
| `cubic` | 最慢 | 最高 | ⭐ |

### 内存管理

对于大量文件或大文件，建议设置分块参数：

```python
chunk_size = {'lat': 1000, 'lon': 1000}  # 根据可用内存调整
```

## 错误处理

### 常见问题

1. **文件名格式错误**
   - 确保文件名格式为 `ASTGTM2_NyyExxx_dem.tif`
   - 纬度和经度必须是数字

2. **内存不足**
   - 减少 `chunk_size` 参数
   - 分批处理文件

3. **磁盘空间不足**
   - 检查输出目录的可用空间
   - 考虑压缩输出文件

4. **文件权限问题**
   - 确保对输入和输出目录有读写权限

### 错误日志

程序会详细记录：
- 成功处理的文件数量
- 失败的文件列表和原因
- 合并过程的状态信息

## 输出文件信息

### 临时NetCDF文件

- 命名格式: `ASTGTM2_N{lat:02d}E{lon:03d}_dem_interp_{resolution}deg.nc`
- 包含元数据: 源文件名、中心坐标、处理日期等

### 最终合并文件

- 包含所有输入文件的数据
- 统一的坐标系统
- 完整的属性信息
- 数据压缩以节省空间

## 示例代码

```python
from src.dem_interpolation import batch_process_and_merge

# 执行批量处理
result = batch_process_and_merge(
    input_dir=r'h:\data\DEM',
    output_dir=r'h:\data\DEM\netcdf_output',
    final_output_file=r'h:\data\DEM\merged_dem_data.nc',
    resolution=0.01,
    method='vectorized',
    chunk_size={'lat': 1000, 'lon': 1000}
)

if result:
    print(f"处理完成: {result}")
```

## 边界优化测试

### 测试边界优化功能
```bash
python test_boundary_optimization.py
```

此脚本会：
- 创建包含边界缺口的测试数据
- 比较传统合并与优化合并的效果
- 可视化边界优化前后的对比
- 统计填充效果和改善程度

### 预期改善效果
- **缺失值减少**: 边界缺失值减少60-90%
- **数据连续性**: 消除明显的边界线条和空隙
- **平滑过渡**: 数据集间的过渡更加自然
- **覆盖率提升**: 整体数据覆盖率显著提高

## 注意事项

1. **数据质量**: 确保输入的DEM文件质量良好，无损坏
2. **坐标系统**: 所有文件应使用相同的坐标参考系统
3. **边界处理**: 新版本自动处理边界缺口和重叠区域
4. **大文件**: 对于大量文件，处理可能需要较长时间
5. **备份**: 建议在处理前备份原始数据
6. **质量检查**: 处理完成后建议检查边界区域的数据质量