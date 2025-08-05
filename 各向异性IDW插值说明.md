# 各向异性反距离权重法能见度插值

## 功能概述

已成功实现各向异性反距离权重法（Anisotropic IDW），用于将气象站点的能见度数据插值到与DEM数据一致的格点上。

## 核心算法

### 各向异性距离函数

$$d_i = \sqrt{(x_i - x_0)^2 + (y_i - y_0)^2 + \beta^2 (z_i - z_0)^2}$$

其中：
- $d_i$：插值点与观测点的各向异性距离
- $x_i, y_i, z_i$：观测点经度、纬度、海拔
- $x_0, y_0, z_0$：插值点经度、纬度、海拔
- $\beta$：垂直方向权重放大因子（默认10.0）

### 权重计算

$$w_i = \frac{1}{d_i^p}$$

其中：
- $p$：权重幂次，控制衰减速度（默认2.0）

### 搜索策略

- **固定邻居数法**：选取距离待插值点最近的6个已知点进行插值
- **经纬度转换**：自动将经纬度差值转换为欧氏距离（公里）

## 主要函数

### 1. `anisotropic_idw_interpolation()`
核心插值函数，执行各向异性反距离权重插值。

**参数**：
- `df_station`: 站点数据DataFrame，包含lon, lat, vis, altitude字段
- `target_lons/lats/elevations`: 目标点的经纬度和海拔
- `beta`: 垂直方向权重放大因子（默认10.0）
- `power`: 权重幂次（默认2.0）
- `n_neighbors`: 使用的邻居数量（默认6）

### 2. `create_visibility_grid()`
创建与DEM网格一致的能见度插值网格。

### 3. `visualize_visibility_result()`
可视化插值结果，包括：
- DEM地形数据
- 气象站点分布
- 插值后的能见度场
- 结果叠加图

## 使用方法

### 基本使用

```python
# 导入模块
from src.vis_dem_dis import create_visibility_grid, visualize_visibility_result

# 读取数据（示例）
df_station = pd.read_csv('data/station_vis_all_estimated.csv')
ds_dem = xr.open_dataset('data/merged_dem_data.nc')

# 执行插值
vis_grid = create_visibility_grid(
    df_station, ds_dem,
    beta=10.0,      # 垂直方向权重放大因子
    power=2.0,      # 权重幂次
    n_neighbors=6   # 使用最近的6个邻居
)

# 保存结果
vis_grid.to_netcdf('visibility_anisotropic_idw.nc')

# 可视化
visualize_visibility_result(df_station, vis_grid, ds_dem, 
                           save_path='visibility_result.png')
```

### 直接运行

```bash
cd src
python vis_dem_dis.py
```

## 测试验证

提供了完整的测试脚本 `test_visibility_interpolation.py`：

```bash
python test_visibility_interpolation.py
```

测试内容包括：
- 基本函数功能验证
- 插值算法正确性检查
- 可视化结果输出

## 技术特点

### ✅ 性能优化
- **批处理**：大数据量时使用批处理，避免内存溢出
- **智能邻居搜索**：支持sklearn的高效搜索或简化实现
- **进度显示**：长时间计算时显示处理进度

### ✅ 灵活性
- **自适应依赖**：自动检测是否有sklearn，提供备选方案
- **参数可调**：beta、power、邻居数量等参数可调节
- **数据兼容**：与xarray、pandas完全兼容

### ✅ 可视化支持
- **地理投影**：使用cartopy进行地理可视化
- **多图对比**：DEM、站点、插值结果的综合展示
- **中文支持**：完整的中文字体支持

## 输出结果

插值完成后将生成：
1. **NetCDF文件**：`visibility_anisotropic_idw.nc` - 包含插值结果的网格数据
2. **可视化图像**：`visibility_interpolation_result.png` - 四面板对比图
3. **统计信息**：插值成功率、数据范围等统计信息

## 注意事项

1. **数据要求**：
   - 站点数据必须包含：lon, lat, vis, altitude字段
   - DEM数据必须包含：elevation字段和对应的经纬度坐标

2. **参数调节**：
   - `beta`值越大，垂直方向影响越强（适用于地形复杂区域）
   - `power`值越大，距离衰减越快（适用于局地性强的现象）
   - `n_neighbors`建议6-10个，过多会降低局地特征

3. **计算时间**：
   - 大网格（如>100×100）可能需要几分钟到几十分钟
   - 建议先用小区域测试参数效果

## 算法优势

相比传统IDW方法，本实现具有以下优势：
- **地形敏感**：考虑海拔高度差异对能见度的影响
- **物理意义**：符合大气现象的垂直分层特征  
- **灵活调节**：可根据研究区域特点调整参数
- **高效实现**：优化的算法确保合理的计算时间