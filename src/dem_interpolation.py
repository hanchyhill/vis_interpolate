import rioxarray
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

def load_dem_data(file_path):
    """
    加载DEM数据
    
    Parameters:
    file_path (str): DEM文件路径
    
    Returns:
    xarray.DataArray: 加载的DEM数据
    """
    try:
        dem_data = rioxarray.open_rasterio(file_path)
        print(f"DEM数据已成功加载:")
        print(f"数据形状: {dem_data.shape}")
        print(f"数据大小: {dem_data.nbytes / 1024 / 1024:.2f} MB")
        return dem_data
    except Exception as e:
        print(f"加载DEM数据时发生错误: {e}")
        return None

def create_target_grid(dem_data, resolution=0.01):
    """
    创建目标网格 (0.01°×0.01°)
    
    Parameters:
    dem_data (xarray.DataArray): 原始DEM数据
    resolution (float): 网格分辨率，默认0.01度
    
    Returns:
    tuple: (lon_grid, lat_grid) 目标网格的经纬度坐标
    """
    # 获取原始数据的边界
    left, bottom, right, top = dem_data.rio.bounds()
    
    # 创建目标网格
    lon_target = np.arange(left, right + resolution, resolution)
    lat_target = np.arange(bottom, top + resolution, resolution)
    
    # 创建网格
    lon_grid, lat_grid = np.meshgrid(lon_target, lat_target)
    
    print(f"目标网格信息:")
    print(f"经度范围: {lon_target.min():.4f}° - {lon_target.max():.4f}°")
    print(f"纬度范围: {lat_target.min():.4f}° - {lat_target.max():.4f}°")
    print(f"网格大小: {lat_grid.shape[0]} × {lon_grid.shape[1]}")
    print(f"网格分辨率: {resolution}°")
    
    return lon_grid, lat_grid

def fast_nearest_interpolation(dem_data, lon_grid, lat_grid):
    """
    快速最邻近插值方法
    直接通过索引查找最近的原始数据点，比scipy的griddata快很多
    
    Parameters:
    dem_data (xarray.DataArray): 原始DEM数据
    lon_grid (numpy.ndarray): 目标经度网格
    lat_grid (numpy.ndarray): 目标纬度网格
    
    Returns:
    numpy.ndarray: 插值后的网格数据
    """
    # 获取原始数据的坐标
    x_coords = dem_data.x.values
    y_coords = dem_data.y.values
    dem_values = dem_data.values
    
    # 创建结果数组
    result = np.full(lat_grid.shape, np.nan)
    
    # 计算原始数据的分辨率
    x_res = x_coords[1] - x_coords[0]
    y_res = y_coords[1] - y_coords[0]
    
    # 获取原始数据的边界
    x_min, x_max = x_coords[0], x_coords[-1]
    y_min, y_max = y_coords[0], y_coords[-1]
    
    # 对每个目标点找最近的原始数据点
    for i in range(lat_grid.shape[0]):
        for j in range(lat_grid.shape[1]):
            target_lon = lon_grid[i, j]
            target_lat = lat_grid[i, j]
            
            # 检查是否在数据范围内
            if x_min <= target_lon <= x_max and y_min <= target_lat <= y_max:
                # 找到最近的索引
                x_idx = int(round((target_lon - x_min) / x_res))
                y_idx = int(round((target_lat - y_min) / y_res))
                
                # 确保索引在有效范围内
                x_idx = max(0, min(x_idx, len(x_coords) - 1))
                y_idx = max(0, min(y_idx, len(y_coords) - 1))
                
                result[i, j] = dem_values[y_idx, x_idx]
    
    return result

def block_average_interpolation(dem_data, lon_grid, lat_grid):
    """
    块平均插值方法 - 最快的方法
    将原始数据分块，每个块计算平均值作为目标网格点的值
    
    Parameters:
    dem_data (xarray.DataArray): 原始DEM数据
    lon_grid (numpy.ndarray): 目标经度网格
    lat_grid (numpy.ndarray): 目标纬度网格
    
    Returns:
    numpy.ndarray: 插值后的网格数据
    """
    # 获取原始数据的坐标和值
    x_coords = dem_data.x.values
    y_coords = dem_data.y.values
    dem_values = dem_data.values
    
    # 计算目标网格的分辨率
    target_x_res = lon_grid[0, 1] - lon_grid[0, 0]
    target_y_res = lat_grid[1, 0] - lat_grid[0, 0]
    
    # 计算原始数据的分辨率
    orig_x_res = x_coords[1] - x_coords[0]
    orig_y_res = y_coords[1] - y_coords[0]
    
    # 计算降采样的步长
    x_step = max(1, int(abs(target_x_res / orig_x_res)))
    y_step = max(1, int(abs(target_y_res / orig_y_res)))
    
    print(f"块大小: {x_step} × {y_step} 像素")
    
    # 创建结果数组
    result = np.full(lat_grid.shape, np.nan)
    
    # 获取原始数据边界
    x_min, x_max = x_coords[0], x_coords[-1]
    y_min, y_max = y_coords[0], y_coords[-1]
    
    # 对每个目标网格点进行块平均
    for i in range(lat_grid.shape[0]):
        for j in range(lat_grid.shape[1]):
            target_lon = lon_grid[i, j]
            target_lat = lat_grid[i, j]
            
            # 检查是否在数据范围内
            if x_min <= target_lon <= x_max and y_min <= target_lat <= y_max:
                # 计算中心索引
                center_x_idx = int((target_lon - x_min) / orig_x_res)
                center_y_idx = int((target_lat - y_min) / orig_y_res)
                
                # 计算块的边界
                x_start = max(0, center_x_idx - x_step // 2)
                x_end = min(len(x_coords), center_x_idx + x_step // 2 + 1)
                y_start = max(0, center_y_idx - y_step // 2)
                y_end = min(len(y_coords), center_y_idx + y_step // 2 + 1)
                
                # 提取块数据并计算平均值
                block = dem_values[y_start:y_end, x_start:x_end]
                if block.size > 0:
                    # 忽略nan值计算平均
                    result[i, j] = np.nanmean(block)
    
    return result

def vectorized_block_average(dem_data, lon_grid, lat_grid):
    """
    向量化的块平均插值方法 - 最高效的方法
    使用xarray的内置重采样功能，速度最快
    
    Parameters:
    dem_data (xarray.DataArray): 原始DEM数据
    lon_grid (numpy.ndarray): 目标经度网格
    lat_grid (numpy.ndarray): 目标纬度网格
    
    Returns:
    numpy.ndarray: 插值后的网格数据
    """
    # 创建目标坐标
    target_lons = lon_grid[0, :]
    target_lats = lat_grid[:, 0]
    
    # 使用xarray的内置插值功能进行最邻近插值
    interpolated = dem_data.interp(
        x=target_lons,
        y=target_lats,
        method='nearest'
    )
    
    return interpolated.values

def interpolate_dem_to_grid(dem_data, lon_grid, lat_grid, method='fast_nearest'):
    """
    将DEM数据插值到目标网格
    
    Parameters:
    dem_data (xarray.DataArray): 原始DEM数据
    lon_grid (numpy.ndarray): 目标经度网格
    lat_grid (numpy.ndarray): 目标纬度网格
    method (str): 插值方法 ('linear', 'nearest', 'cubic', 'fast_nearest', 'block_average', 'vectorized')
    
    Returns:
    xarray.DataArray: 插值后的DEM数据
    """
    import time
    start_time = time.time()
    
    dem_squeezed = dem_data.squeeze()
    
    if method == 'fast_nearest':
        # 使用快速最邻近插值方法
        print(f"使用快速最邻近插值方法...")
        interpolated_grid = fast_nearest_interpolation(dem_squeezed, lon_grid, lat_grid)
        
    elif method == 'block_average':
        # 使用块平均方法
        print(f"使用块平均插值方法...")
        interpolated_grid = block_average_interpolation(dem_squeezed, lon_grid, lat_grid)
        
    elif method == 'vectorized':
        # 使用向量化方法（最快）
        print(f"使用向量化插值方法（推荐）...")
        interpolated_grid = vectorized_block_average(dem_squeezed, lon_grid, lat_grid)
        
    else:
        # 原始的scipy插值方法
        print(f"使用传统插值方法: {method}...")
        
        # 获取原始数据的坐标
        x_coords = dem_squeezed.x.values
        y_coords = dem_squeezed.y.values
        
        # 创建原始数据的网格
        x_orig, y_orig = np.meshgrid(x_coords, y_coords)
        
        # 准备插值数据
        points = np.column_stack([x_orig.flatten(), y_orig.flatten()])
        values = dem_squeezed.values.flatten()
        
        # 准备目标点
        target_points = np.column_stack([lon_grid.flatten(), lat_grid.flatten()])
        
        print(f"原始数据点数: {len(points)}")
        print(f"目标网格点数: {len(target_points)}")
        
        # 执行插值
        interpolated_values = griddata(
            points, 
            values, 
            target_points, 
            method=method,
            fill_value=np.nan
        )
        
        # 重塑为网格形状
        interpolated_grid = interpolated_values.reshape(lat_grid.shape)
    
    # 创建xarray DataArray
    interpolated_da = xr.DataArray(
        interpolated_grid,
        coords={
            'lat': ('lat', lat_grid[:, 0]),
            'lon': ('lon', lon_grid[0, :])
        },
        dims=['lat', 'lon'],
        name='elevation',
        attrs={
            'units': 'meters',
            'long_name': 'Elevation',
            'interpolation_method': method,
            'original_resolution': f"{dem_data.x.values[1] - dem_data.x.values[0]:.6f}°",
            'target_resolution': '0.01°'
        }
    )
    
    elapsed_time = time.time() - start_time
    print(f"插值完成! 耗时: {elapsed_time:.2f}秒")
    print(f"插值后数据大小: {interpolated_da.nbytes / 1024 / 1024:.2f} MB")
    print(f"数据压缩比: {dem_data.nbytes / interpolated_da.nbytes:.2f}")
    
    return interpolated_da

def visualize_comparison(original_dem, interpolated_dem, save_path=None):
    """
    可视化原始数据和插值后的数据对比
    使用cartopy统一绘制地理信息，确保坐标轴一致
    
    Parameters:
    original_dem (xarray.DataArray): 原始DEM数据
    interpolated_dem (xarray.DataArray): 插值后的DEM数据
    save_path (str, optional): 保存图片的路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # 设置中文显示
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取统一的数据范围（用于colorbar）
    original_squeezed = original_dem.squeeze()
    vmin = min(original_squeezed.min().values, interpolated_dem.min().values)
    vmax = max(original_squeezed.max().values, interpolated_dem.max().values)
    
    # 绘制原始数据
    ax1 = axes[0]
    
    # 获取原始数据的坐标
    orig_lons = original_squeezed.x.values
    orig_lats = original_squeezed.y.values
    orig_lon_grid, orig_lat_grid = np.meshgrid(orig_lons, orig_lats)
    
    # 使用pcolormesh绘制原始数据（正确的地理投影方式）
    img1 = ax1.pcolormesh(
        orig_lon_grid, orig_lat_grid, 
        original_squeezed.values,
        cmap='terrain',
        vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(),
        shading='nearest'
    )
    
    # 设置地图属性
    ax1.set_extent([orig_lons.min(), orig_lons.max(), orig_lats.min(), orig_lats.max()], 
                   crs=ccrs.PlateCarree())
    ax1.coastlines(resolution='10m', alpha=0.8)
    ax1.gridlines(draw_labels=True, alpha=0.5)
    ax1.set_title(f"原始DEM数据\n{original_squeezed.shape[1]}×{original_squeezed.shape[0]} 像素\n分辨率: ~{abs(orig_lons[1]-orig_lons[0])*111000:.0f}m")
    
    # 添加colorbar
    cbar1 = plt.colorbar(img1, ax=ax1, label='高程 (m)', shrink=0.8)
    
    # 绘制插值后的数据
    ax2 = axes[1]
    
    # 获取插值后数据的坐标
    interp_lons = interpolated_dem.lon.values
    interp_lats = interpolated_dem.lat.values
    interp_lon_grid, interp_lat_grid = np.meshgrid(interp_lons, interp_lats)
    
    # 使用pcolormesh绘制插值数据
    img2 = ax2.pcolormesh(
        interp_lon_grid, interp_lat_grid,
        interpolated_dem.values,
        cmap='terrain',
        vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(),
        shading='nearest'
    )
    
    # 设置地图属性
    ax2.set_extent([interp_lons.min(), interp_lons.max(), interp_lats.min(), interp_lats.max()], 
                   crs=ccrs.PlateCarree())
    ax2.coastlines(resolution='10m', alpha=0.8)
    ax2.gridlines(draw_labels=True, alpha=0.5)
    ax2.set_title(f"插值后DEM数据\n{interpolated_dem.shape[1]}×{interpolated_dem.shape[0]} 网格\n分辨率: ~{abs(interp_lons[1]-interp_lons[0])*111000:.0f}m")
    
    # 添加colorbar
    cbar2 = plt.colorbar(img2, ax=ax2, label='高程 (m)', shrink=0.8)
    
    plt.tight_layout()
    
    # 显示数据统计信息
    print(f"\n=== 数据对比统计 ===")
    print(f"原始数据: 最小值={original_squeezed.min().values:.1f}m, 最大值={original_squeezed.max().values:.1f}m, 平均值={original_squeezed.mean().values:.1f}m")
    print(f"插值数据: 最小值={interpolated_dem.min().values:.1f}m, 最大值={interpolated_dem.max().values:.1f}m, 平均值={interpolated_dem.mean().values:.1f}m")
    print(f"高程范围: {vmin:.1f}m - {vmax:.1f}m")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
    
    plt.show()

def visualize_multiple_methods(original_dem, interpolated_results, save_path=None):
    """
    可视化多种插值方法的对比结果
    
    Parameters:
    original_dem (xarray.DataArray): 原始DEM数据
    interpolated_results (dict): 插值结果字典 {method_name: result_data}
    save_path (str, optional): 保存图片的路径
    """
    n_methods = len(interpolated_results)
    n_cols = min(3, n_methods + 1)  # 最多3列，包括原始数据
    n_rows = (n_methods + 1 + n_cols - 1) // n_cols  # 向上取整
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), 
                           subplot_kw={'projection': ccrs.PlateCarree()})
    
    # 如果只有一行，确保axes是数组
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else np.array([[axes]])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 设置中文显示
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取统一的数据范围
    original_squeezed = original_dem.squeeze()
    all_values = [original_squeezed.values.flatten()]
    for result in interpolated_results.values():
        all_values.append(result.values.flatten())
    
    all_data = np.concatenate(all_values)
    vmin, vmax = np.nanpercentile(all_data, [1, 99])  # 使用1%和99%分位数避免极值影响
    
    plot_idx = 0
    
    # 绘制原始数据
    row, col = plot_idx // n_cols, plot_idx % n_cols
    ax = axes[row, col]
    
    orig_lons = original_squeezed.x.values
    orig_lats = original_squeezed.y.values
    orig_lon_grid, orig_lat_grid = np.meshgrid(orig_lons, orig_lats)
    
    img = ax.pcolormesh(
        orig_lon_grid, orig_lat_grid, 
        original_squeezed.values,
        cmap='terrain',
        vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(),
        shading='nearest'
    )
    
    ax.set_extent([orig_lons.min(), orig_lons.max(), orig_lats.min(), orig_lats.max()], 
                  crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', alpha=0.8)
    ax.gridlines(alpha=0.3)
    ax.set_title(f"原始DEM数据\n~{abs(orig_lons[1]-orig_lons[0])*111000:.0f}m分辨率")
    
    plot_idx += 1
    
    # 绘制各种插值方法的结果
    for method_name, result_data in interpolated_results.items():
        if plot_idx >= n_rows * n_cols:
            break
            
        row, col = plot_idx // n_cols, plot_idx % n_cols
        ax = axes[row, col]
        
        interp_lons = result_data.lon.values
        interp_lats = result_data.lat.values
        interp_lon_grid, interp_lat_grid = np.meshgrid(interp_lons, interp_lats)
        
        img = ax.pcolormesh(
            interp_lon_grid, interp_lat_grid,
            result_data.values,
            cmap='terrain',
            vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(),
            shading='nearest'
        )
        
        ax.set_extent([interp_lons.min(), interp_lons.max(), interp_lats.min(), interp_lats.max()], 
                      crs=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', alpha=0.8)
        ax.gridlines(alpha=0.3)
        ax.set_title(f"{method_name}\n~{abs(interp_lons[1]-interp_lons[0])*111000:.0f}m分辨率")
        
        plot_idx += 1
    
    # 隐藏多余的子图
    for idx in range(plot_idx, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    # 添加一个公共的colorbar
    plt.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    fig.colorbar(img, cax=cbar_ax, label='高程 (m)')
    
    plt.suptitle('DEM插值方法对比', fontsize=16, y=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"多方法对比图已保存到: {save_path}")
    
    plt.show()

def save_interpolated_data(interpolated_dem, output_path):
    """
    保存插值后的数据
    
    Parameters:
    interpolated_dem (xarray.DataArray): 插值后的DEM数据
    output_path (str): 输出文件路径
    """
    try:
        # 保存为NetCDF格式
        interpolated_dem.to_netcdf(output_path)
        print(f"插值后的数据已保存到: {output_path}")
    except Exception as e:
        print(f"保存数据时发生错误: {e}")

def compare_interpolation_methods(dem_data, lon_grid, lat_grid):
    """
    比较不同插值方法的性能
    
    Parameters:
    dem_data (xarray.DataArray): 原始DEM数据
    lon_grid (numpy.ndarray): 目标经度网格
    lat_grid (numpy.ndarray): 目标纬度网格
    
    Returns:
    dict: 各方法的性能统计
    """
    methods = ['vectorized', 'fast_nearest', 'nearest', 'linear']
    results = {}
    
    print("=== 插值方法性能比较 ===")
    print(f"原始数据大小: {dem_data.shape}")
    print(f"目标网格大小: {lon_grid.shape}")
    print("-" * 50)
    
    for method in methods:
        try:
            print(f"\n测试方法: {method}")
            import time
            start_time = time.time()
            
            interpolated = interpolate_dem_to_grid(dem_data, lon_grid, lat_grid, method=method)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            results[method] = {
                'time': elapsed,
                'shape': interpolated.shape,
                'memory_mb': interpolated.nbytes / 1024 / 1024
            }
            
            print(f"✓ 完成! 耗时: {elapsed:.2f}秒")
            
        except Exception as e:
            print(f"✗ 失败: {str(e)}")
            results[method] = {'time': float('inf'), 'error': str(e)}
    
    # 显示比较结果
    print("\n" + "=" * 50)
    print("性能比较结果:")
    print("-" * 50)
    
    # 按速度排序
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    sorted_methods = sorted(valid_results.items(), key=lambda x: x[1]['time'])
    
    fastest_time = sorted_methods[0][1]['time'] if sorted_methods else 0
    
    for method, stats in sorted_methods:
        speedup = fastest_time / stats['time'] if stats['time'] > 0 else 1
        print(f"{method:12} | {stats['time']:8.2f}秒 | {speedup:5.1f}x | {stats['memory_mb']:6.1f}MB")
    
    print("-" * 50)
    print(f"推荐使用: {sorted_methods[0][0]} (最快)")
    
    return results

def main():
    """
    主函数：执行DEM数据稀疏化
    """
    # DEM文件路径
    dem_file_path = r'h:\data\DEM\ASTGTM2_N23E111_dem.tif'
    
    print("=== DEM数据稀疏化处理 ===")
    print(f"输入文件: {dem_file_path}")
    
    # 1. 加载DEM数据
    dem_data = load_dem_data(dem_file_path)
    if dem_data is None:
        return
    
    # 2. 创建目标网格 (0.01°×0.01°)
    lon_grid, lat_grid = create_target_grid(dem_data, resolution=0.01)
    
    # 3. 执行插值（使用快速向量化方法）
    interpolated_dem = interpolate_dem_to_grid(dem_data, lon_grid, lat_grid, method='vectorized')
    
    # 4. 可视化对比
    visualize_comparison(dem_data, interpolated_dem, save_path='dem_comparison.png')
    
    # 5. 保存结果
    output_path = 'interpolated_dem_0.01deg.nc'
    save_interpolated_data(interpolated_dem, output_path)
    
    print("\n=== 处理完成 ===")
    print(f"原始数据大小: {dem_data.nbytes / 1024 / 1024:.2f} MB")
    print(f"插值后数据大小: {interpolated_dem.nbytes / 1024 / 1024:.2f} MB")
    print(f"数据压缩比: {dem_data.nbytes / interpolated_dem.nbytes:.2f}")

if __name__ == "__main__":
    main() 