import rioxarray
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from skimage.morphology import binary_closing
import warnings
import os
import glob
import re
from pathlib import Path
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

def parse_dem_filename(filename):
    """
    解析DEM文件名，提取纬度和经度信息
    
    Parameters:
    filename (str): DEM文件名，格式如 ASTGTM2_NyyExxx_dem.tif
    
    Returns:
    tuple: (lat, lon) 纬度和经度，如果解析失败返回 (None, None)
    """
    pattern = r'ASTGTM2_N(\d{2})E(\d{3})_dem\.tif'
    match = re.match(pattern, filename)
    
    if match:
        lat = int(match.group(1))
        lon = int(match.group(2))
        return lat, lon
    else:
        print(f"警告: 无法解析文件名 {filename}")
        return None, None

def process_dem_directory(input_dir, output_dir, resolution=0.01, method='vectorized'):
    """
    批量处理DEM目录中的所有TIF文件，转换为NetCDF格式
    
    Parameters:
    input_dir (str): 输入目录路径
    output_dir (str): 输出目录路径
    resolution (float): 目标网格分辨率，默认0.01度
    method (str): 插值方法，默认'vectorized'
    
    Returns:
    list: 成功处理的文件列表
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有DEM TIF文件
    tif_files = list(input_path.glob('ASTGTM2_N*E*_dem.tif'))
    
    print(f"=== 批量处理DEM文件 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(tif_files)} 个DEM文件")
    print(f"目标分辨率: {resolution}°")
    print(f"插值方法: {method}")
    print("-" * 50)
    
    processed_files = []
    failed_files = []
    
    for i, tif_file in enumerate(tif_files, 1):
        print(f"\n处理文件 {i}/{len(tif_files)}: {tif_file.name}")
        
        try:
            # 解析文件名获取经纬度信息
            lat, lon = parse_dem_filename(tif_file.name)
            if lat is None or lon is None:
                failed_files.append(str(tif_file))
                continue
            
            # 加载DEM数据
            dem_data = load_dem_data(str(tif_file))
            if dem_data is None:
                failed_files.append(str(tif_file))
                continue
            
            # 创建目标网格
            lon_grid, lat_grid = create_target_grid(dem_data, resolution=resolution)
            
            # 执行插值
            interpolated_dem = interpolate_dem_to_grid(dem_data, lon_grid, lat_grid, method=method)
            
            # 添加额外的属性信息
            interpolated_dem.attrs.update({
                'source_file': tif_file.name,
                'center_lat': lat,
                'center_lon': lon,
                'processing_date': str(np.datetime64('now')),
                'grid_resolution_degrees': resolution
            })
            
            # 生成输出文件名
            output_filename = f"ASTGTM2_N{lat:02d}E{lon:03d}_dem_interp_{resolution}deg.nc"
            output_file_path = output_path / output_filename
            
            # 保存为NetCDF
            interpolated_dem.to_netcdf(output_file_path)
            
            processed_files.append(str(output_file_path))
            print(f"✓ 成功处理并保存: {output_filename}")
            
        except Exception as e:
            print(f"✗ 处理失败: {e}")
            failed_files.append(str(tif_file))
    
    print(f"\n=== 批量处理完成 ===")
    print(f"成功处理: {len(processed_files)} 个文件")
    print(f"处理失败: {len(failed_files)} 个文件")
    
    if failed_files:
        print("失败的文件:")
        for file in failed_files:
            print(f"  - {file}")
    
    return processed_files

def merge_netcdf_files(netcdf_files, output_file, chunk_size=None):
    """
    合并多个NetCDF文件为单个文件
    
    Parameters:
    netcdf_files (list): NetCDF文件路径列表
    output_file (str): 输出合并文件路径
    chunk_size (dict): xarray的chunk参数，用于处理大文件
    
    Returns:
    xarray.Dataset: 合并后的数据集
    """
    print(f"\n=== 合并NetCDF文件 ===")
    print(f"要合并的文件数量: {len(netcdf_files)}")
    print(f"输出文件: {output_file}")
    
    if not netcdf_files:
        print("没有找到要合并的文件")
        return None
    
    try:
        # 读取所有NetCDF文件
        datasets = []
        coords_info = []
        
        for i, file_path in enumerate(netcdf_files):
            print(f"读取文件 {i+1}/{len(netcdf_files)}: {Path(file_path).name}")
            
            if chunk_size:
                ds = xr.open_dataset(file_path, chunks=chunk_size)
            else:
                ds = xr.open_dataset(file_path)
            
            datasets.append(ds)
            
            # 记录坐标信息用于后续分析
            coords_info.append({
                'file': Path(file_path).name,
                'lat_range': (ds.lat.min().values, ds.lat.max().values),
                'lon_range': (ds.lon.min().values, ds.lon.max().values)
            })
        
        print("正在合并数据集...")
        
        # 合并数据集
        # 使用concat沿着空间维度合并
        combined_ds = xr.concat(datasets, dim='tile')
        
        # 如果数据在空间上是连续的，可以尝试使用merge
        try:
            # 先尝试merge（适用于空间上不重叠的数据）
            print("尝试使用merge方法合并...")
            merged_ds = xr.merge(datasets)
            
            # 检查合并结果的维度
            if 'elevation' in merged_ds:
                elevation_data = merged_ds['elevation']
                print(f"合并后的数据维度: {elevation_data.dims}")
                print(f"合并后的形状: {elevation_data.shape}")
                
                # 创建一个新的合并数据集，重新组织坐标
                merged_ds = create_unified_grid(datasets)
                
        except Exception as e:
            print(f"merge方法失败，使用concat方法: {e}")
            merged_ds = combined_ds
        
        # 添加全局属性
        merged_ds.attrs.update({
            'title': 'Merged DEM data',
            'description': f'Combined from {len(netcdf_files)} individual DEM tiles',
            'creation_date': str(np.datetime64('now')),
            'source_files_count': len(netcdf_files),
            'processing_info': 'Merged using xarray'
        })
        
        # 保存合并后的文件
        print("保存合并后的文件...")
        if chunk_size:
            # 对大文件使用分块保存
            merged_ds.to_netcdf(output_file, encoding={'elevation': {'zlib': True, 'complevel': 5}})
        else:
            merged_ds.to_netcdf(output_file)
        
        print(f"✓ 成功合并并保存到: {output_file}")
        
        # 显示合并后的数据信息
        print("\n=== 合并结果信息 ===")
        if 'elevation' in merged_ds:
            elevation = merged_ds['elevation']
            print(f"数据变量: elevation")
            print(f"数据类型: {elevation.dtype}")
            print(f"数据形状: {elevation.shape}")
            print(f"文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
            
            # 显示空间范围
            if 'lat' in merged_ds.coords and 'lon' in merged_ds.coords:
                lat_range = (merged_ds.lat.min().values, merged_ds.lat.max().values)
                lon_range = (merged_ds.lon.min().values, merged_ds.lon.max().values)
                print(f"纬度范围: {lat_range[0]:.4f}° - {lat_range[1]:.4f}°")
                print(f"经度范围: {lon_range[0]:.4f}° - {lon_range[1]:.4f}°")
        
        # 关闭数据集
        for ds in datasets:
            ds.close()
        
        return merged_ds
        
    except Exception as e:
        print(f"合并文件时发生错误: {e}")
        return None

def fill_boundary_gaps(data, method='linear', max_distance=5):
    """
    填充边界处的缺失值
    
    Parameters:
    data (numpy.ndarray): 2D数组，包含缺失值(NaN)
    method (str): 填充方法 ('linear', 'nearest', 'cubic')
    max_distance (int): 最大填充距离（像素）
    
    Returns:
    numpy.ndarray: 填充后的数据
    """
    print(f"使用{method}方法填充边界缺失值...")
    
    # 创建数据副本
    filled_data = data.copy()
    
    # 识别有效数据和缺失数据的掩码
    valid_mask = ~np.isnan(data)
    missing_mask = np.isnan(data)
    
    if not np.any(missing_mask):
        print("没有发现缺失值")
        return filled_data
    
    # 计算距离变换，找到每个缺失点到最近有效点的距离
    distance_map = distance_transform_edt(missing_mask)
    
    # 只填充距离小于max_distance的缺失值
    fill_mask = missing_mask & (distance_map <= max_distance)
    
    if not np.any(fill_mask):
        print("没有需要填充的边界缺失值")
        return filled_data
    
    print(f"找到 {np.sum(fill_mask)} 个需要填充的边界缺失值")
    
    # 获取有效数据点的坐标和值
    valid_coords = np.column_stack(np.where(valid_mask))
    valid_values = data[valid_mask]
    
    # 获取需要填充的点的坐标
    fill_coords = np.column_stack(np.where(fill_mask))
    
    try:
        if method == 'nearest':
            # 最邻近插值
            interpolator = NearestNDInterpolator(valid_coords, valid_values)
            filled_values = interpolator(fill_coords)
        elif method == 'linear':
            # 线性插值
            interpolator = LinearNDInterpolator(valid_coords, valid_values, fill_value=np.nan)
            filled_values = interpolator(fill_coords)
            # 如果线性插值失败，使用最邻近插值作为备选
            nan_mask = np.isnan(filled_values)
            if np.any(nan_mask):
                backup_interpolator = NearestNDInterpolator(valid_coords, valid_values)
                filled_values[nan_mask] = backup_interpolator(fill_coords[nan_mask])
        else:
            # 使用scipy的griddata进行插值
            filled_values = griddata(
                valid_coords, valid_values, fill_coords, 
                method=method, fill_value=np.nan
            )
            # 如果插值失败，使用最邻近插值
            nan_mask = np.isnan(filled_values)
            if np.any(nan_mask):
                filled_values[nan_mask] = griddata(
                    valid_coords, valid_values, fill_coords[nan_mask], 
                    method='nearest'
                )
        
        # 将填充的值放回数组中
        filled_data[fill_coords[:, 0], fill_coords[:, 1]] = filled_values
        
        filled_count = np.sum(~np.isnan(filled_values))
        print(f"成功填充 {filled_count} 个缺失值")
        
    except Exception as e:
        print(f"插值填充失败，使用邻域平均: {e}")
        # 备选方案：使用邻域平均
        filled_data = fill_with_neighborhood_average(data, max_distance)
    
    return filled_data

def fill_with_neighborhood_average(data, max_distance=3):
    """
    使用邻域平均方法填充缺失值
    
    Parameters:
    data (numpy.ndarray): 2D数组
    max_distance (int): 邻域半径
    
    Returns:
    numpy.ndarray: 填充后的数据
    """
    filled_data = data.copy()
    missing_mask = np.isnan(data)
    
    # 迭代填充，每次只填充有邻近有效值的缺失点
    for iteration in range(max_distance):
        new_filled = filled_data.copy()
        
        # 使用3x3卷积核计算邻域平均
        from scipy.ndimage import uniform_filter
        
        # 创建权重数组（有效值为1，缺失值为0）
        weights = ~np.isnan(filled_data)
        
        # 计算加权平均
        sum_values = uniform_filter(np.nan_to_num(filled_data), size=3, mode='constant')
        sum_weights = uniform_filter(weights.astype(float), size=3, mode='constant')
        
        # 避免除零
        valid_avg_mask = sum_weights > 0
        avg_values = np.full_like(sum_values, np.nan)
        avg_values[valid_avg_mask] = sum_values[valid_avg_mask] / sum_weights[valid_avg_mask]
        
        # 只填充原本缺失且有有效邻域的点
        fill_this_round = missing_mask & ~np.isnan(avg_values) & (sum_weights > 0.1)
        new_filled[fill_this_round] = avg_values[fill_this_round]
        
        if not np.any(fill_this_round):
            break
            
        filled_data = new_filled
        missing_mask = np.isnan(filled_data)
        
        print(f"第{iteration+1}轮填充了 {np.sum(fill_this_round)} 个缺失值")
    
    return filled_data

def smooth_boundaries(data, datasets_info, smooth_width=3):
    """
    平滑数据集边界处的过渡
    
    Parameters:
    data (numpy.ndarray): 合并后的数据
    datasets_info (list): 数据集信息列表
    smooth_width (int): 平滑宽度
    
    Returns:
    numpy.ndarray: 平滑后的数据
    """
    print("平滑数据集边界...")
    
    smoothed_data = data.copy()
    
    # 使用高斯滤波进行边界平滑
    from scipy.ndimage import gaussian_filter
    
    # 识别边界区域（有数据变化的区域）
    valid_mask = ~np.isnan(data)
    
    # 计算梯度来识别边界
    from scipy.ndimage import sobel
    gradient_x = np.abs(sobel(data, axis=0))
    gradient_y = np.abs(sobel(data, axis=1))
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # 识别高梯度区域作为边界
    threshold = np.nanpercentile(gradient_magnitude, 95)
    boundary_mask = gradient_magnitude > threshold
    
    # 扩展边界区域
    boundary_mask = binary_dilation(boundary_mask, iterations=smooth_width)
    
    # 只在边界区域应用平滑
    if np.any(boundary_mask & valid_mask):
        # 对整个数据应用轻微的高斯平滑
        temp_data = np.nan_to_num(data, nan=0)
        smoothed_temp = gaussian_filter(temp_data, sigma=0.8)
        
        # 只在边界区域应用平滑结果
        boundary_and_valid = boundary_mask & valid_mask
        smoothed_data[boundary_and_valid] = smoothed_temp[boundary_and_valid]
        
        print(f"平滑了 {np.sum(boundary_and_valid)} 个边界点")
    
    return smoothed_data

def create_unified_grid(datasets):
    """
    创建统一的网格，将多个数据集合并到同一个坐标系统中
    优化版本：包含边界处理和缺失值填充
    
    Parameters:
    datasets (list): xarray数据集列表
    
    Returns:
    xarray.Dataset: 统一网格的数据集
    """
    print("创建优化的统一网格...")
    
    # 收集所有的坐标信息
    datasets_info = []
    all_lats = []
    all_lons = []
    
    for i, ds in enumerate(datasets):
        if 'lat' in ds.coords and 'lon' in ds.coords:
            ds_lats = ds.lat.values
            ds_lons = ds.lon.values
            all_lats.extend(ds_lats)
            all_lons.extend(ds_lons)
            
            # 记录每个数据集的详细信息
            datasets_info.append({
                'index': i,
                'lat_range': (ds_lats.min(), ds_lats.max()),
                'lon_range': (ds_lons.min(), ds_lons.max()),
                'lat_coords': ds_lats,
                'lon_coords': ds_lons,
                'shape': ds['elevation'].shape if 'elevation' in ds else None
            })
    
    # 创建统一的坐标网格
    unique_lats = np.unique(all_lats)
    unique_lons = np.unique(all_lons)
    
    # 排序坐标
    unique_lats = np.sort(unique_lats)
    unique_lons = np.sort(unique_lons)
    
    print(f"统一网格大小: {len(unique_lats)} × {len(unique_lons)}")
    print(f"纬度范围: {unique_lats.min():.4f}° - {unique_lats.max():.4f}°")
    print(f"经度范围: {unique_lons.min():.4f}° - {unique_lons.max():.4f}°")
    
    # 创建结果数组和权重数组
    result_data = np.full((len(unique_lats), len(unique_lons)), np.nan)
    weight_data = np.zeros((len(unique_lats), len(unique_lons)))
    
    # 将每个数据集的数据填入统一网格
    for i, ds in enumerate(datasets):
        if 'elevation' in ds:
            print(f"处理数据集 {i+1}/{len(datasets)}")
            
            elevation = ds['elevation']
            ds_lats = ds.lat.values
            ds_lons = ds.lon.values
            
            # 更精确的坐标匹配
            lat_tolerance = np.diff(unique_lats).min() / 2 if len(unique_lats) > 1 else 1e-6
            lon_tolerance = np.diff(unique_lons).min() / 2 if len(unique_lons) > 1 else 1e-6
            
            for j, lat in enumerate(ds_lats):
                for k, lon in enumerate(ds_lons):
                    # 找到最接近的网格点
                    lat_idx = np.argmin(np.abs(unique_lats - lat))
                    lon_idx = np.argmin(np.abs(unique_lons - lon))
                    
                    # 检查是否在容差范围内
                    if (abs(unique_lats[lat_idx] - lat) <= lat_tolerance and 
                        abs(unique_lons[lon_idx] - lon) <= lon_tolerance):
                        
                        elev_value = elevation.values[j, k]
                        if not np.isnan(elev_value):
                            # 如果该位置已有数据，使用加权平均
                            if weight_data[lat_idx, lon_idx] > 0:
                                current_value = result_data[lat_idx, lon_idx]
                                current_weight = weight_data[lat_idx, lon_idx]
                                new_weight = 1.0
                                
                                # 加权平均
                                total_weight = current_weight + new_weight
                                result_data[lat_idx, lon_idx] = (
                                    current_value * current_weight + elev_value * new_weight
                                ) / total_weight
                                weight_data[lat_idx, lon_idx] = total_weight
                            else:
                                result_data[lat_idx, lon_idx] = elev_value
                                weight_data[lat_idx, lon_idx] = 1.0
    
    print("应用边界优化处理...")
    
    # 1. 填充边界缺失值
    result_data = fill_boundary_gaps(result_data, method='linear', max_distance=5)
    
    # 2. 平滑边界过渡
    result_data = smooth_boundaries(result_data, datasets_info, smooth_width=2)
    
    # 3. 最后一次边界填充（处理可能仍存在的小缺口）
    result_data = fill_boundary_gaps(result_data, method='nearest', max_distance=3)
    
    # 创建新的数据集
    unified_ds = xr.Dataset({
        'elevation': (('lat', 'lon'), result_data)
    }, coords={
        'lat': unique_lats,
        'lon': unique_lons
    })
    
    # 添加属性
    if datasets and 'elevation' in datasets[0]:
        unified_ds['elevation'].attrs = datasets[0]['elevation'].attrs.copy()
    
    # 添加处理信息
    unified_ds['elevation'].attrs.update({
        'boundary_processing': 'Optimized with gap filling and smoothing',
        'gap_filling_method': 'linear + nearest',
        'boundary_smoothing': 'applied'
    })
    
    # 显示处理结果统计
    nan_count = np.sum(np.isnan(result_data))
    total_count = result_data.size
    valid_percent = (total_count - nan_count) / total_count * 100
    
    print(f"边界优化完成:")
    print(f"  有效数据点: {total_count - nan_count:,} ({valid_percent:.2f}%)")
    print(f"  剩余缺失值: {nan_count:,} ({100-valid_percent:.2f}%)")
    
    return unified_ds

def visualize_boundary_optimization(original_data, optimized_data, save_path=None):
    """
    可视化边界优化前后的对比
    
    Parameters:
    original_data (xarray.Dataset): 原始合并数据
    optimized_data (xarray.Dataset): 优化后的数据
    save_path (str, optional): 保存图片的路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取数据
    orig_elev = original_data['elevation'] if 'elevation' in original_data else original_data
    opt_elev = optimized_data['elevation'] if 'elevation' in optimized_data else optimized_data
    
    # 统一颜色范围
    vmin = min(np.nanmin(orig_elev.values), np.nanmin(opt_elev.values))
    vmax = max(np.nanmax(orig_elev.values), np.nanmax(opt_elev.values))
    
    # 原始数据
    ax1 = axes[0, 0]
    im1 = ax1.imshow(orig_elev.values, cmap='terrain', vmin=vmin, vmax=vmax, aspect='auto')
    ax1.set_title('原始合并数据\n(含边界缺失值)')
    ax1.set_xlabel('经度方向')
    ax1.set_ylabel('纬度方向')
    plt.colorbar(im1, ax=ax1, label='高程 (m)')
    
    # 优化后数据
    ax2 = axes[0, 1]
    im2 = ax2.imshow(opt_elev.values, cmap='terrain', vmin=vmin, vmax=vmax, aspect='auto')
    ax2.set_title('边界优化后数据\n(缺失值已填充)')
    ax2.set_xlabel('经度方向')
    ax2.set_ylabel('纬度方向')
    plt.colorbar(im2, ax=ax2, label='高程 (m)')
    
    # 缺失值分布对比
    ax3 = axes[1, 0]
    orig_mask = np.isnan(orig_elev.values)
    ax3.imshow(orig_mask, cmap='RdYlBu_r', aspect='auto')
    ax3.set_title(f'原始缺失值分布\n缺失点数: {np.sum(orig_mask):,}')
    ax3.set_xlabel('经度方向')
    ax3.set_ylabel('纬度方向')
    
    ax4 = axes[1, 1]
    opt_mask = np.isnan(opt_elev.values)
    ax4.imshow(opt_mask, cmap='RdYlBu_r', aspect='auto')
    ax4.set_title(f'优化后缺失值分布\n缺失点数: {np.sum(opt_mask):,}')
    ax4.set_xlabel('经度方向')
    ax4.set_ylabel('纬度方向')
    
    plt.tight_layout()
    
    # 显示统计信息
    orig_valid = np.sum(~orig_mask)
    opt_valid = np.sum(~opt_mask)
    total_points = orig_elev.size
    
    print(f"\n=== 边界优化效果统计 ===")
    print(f"总数据点数: {total_points:,}")
    print(f"原始有效点: {orig_valid:,} ({orig_valid/total_points*100:.2f}%)")
    print(f"优化后有效点: {opt_valid:,} ({opt_valid/total_points*100:.2f}%)")
    print(f"新填充点数: {opt_valid - orig_valid:,}")
    print(f"改善程度: {(opt_valid - orig_valid)/orig_valid*100:.2f}%")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"边界优化对比图已保存到: {save_path}")
    
    plt.show()

def analyze_boundary_gaps(datasets, resolution=0.01):
    """
    分析数据集边界间的缺口情况
    
    Parameters:
    datasets (list): xarray数据集列表
    resolution (float): 网格分辨率
    
    Returns:
    dict: 缺口分析结果
    """
    print("=== 分析边界缺口情况 ===")
    
    gaps_info = {
        'total_datasets': len(datasets),
        'overlaps': [],
        'gaps': [],
        'boundary_stats': {}
    }
    
    # 分析每对相邻数据集的边界情况
    for i, ds1 in enumerate(datasets):
        for j, ds2 in enumerate(datasets[i+1:], i+1):
            if 'elevation' in ds1 and 'elevation' in ds2:
                # 获取边界信息
                lat1_range = (ds1.lat.min().values, ds1.lat.max().values)
                lon1_range = (ds1.lon.min().values, ds1.lon.max().values)
                lat2_range = (ds2.lat.min().values, ds2.lat.max().values)
                lon2_range = (ds2.lon.min().values, ds2.lon.max().values)
                
                # 检查是否相邻
                lat_adjacent = (abs(lat1_range[1] - lat2_range[0]) <= resolution * 1.5 or 
                               abs(lat2_range[1] - lat1_range[0]) <= resolution * 1.5)
                lon_adjacent = (abs(lon1_range[1] - lon2_range[0]) <= resolution * 1.5 or 
                               abs(lon2_range[1] - lon1_range[0]) <= resolution * 1.5)
                
                # 检查重叠
                lat_overlap = not (lat1_range[1] < lat2_range[0] or lat2_range[1] < lat1_range[0])
                lon_overlap = not (lon1_range[1] < lon2_range[0] or lon2_range[1] < lon1_range[0])
                
                if lat_adjacent and lon_adjacent:
                    if lat_overlap and lon_overlap:
                        gaps_info['overlaps'].append({
                            'datasets': (i, j),
                            'lat_overlap': (max(lat1_range[0], lat2_range[0]), 
                                          min(lat1_range[1], lat2_range[1])),
                            'lon_overlap': (max(lon1_range[0], lon2_range[0]), 
                                          min(lon1_range[1], lon2_range[1]))
                        })
                    else:
                        # 计算缺口大小
                        lat_gap = 0
                        lon_gap = 0
                        
                        if lat1_range[1] < lat2_range[0]:
                            lat_gap = lat2_range[0] - lat1_range[1]
                        elif lat2_range[1] < lat1_range[0]:
                            lat_gap = lat1_range[0] - lat2_range[1]
                        
                        if lon1_range[1] < lon2_range[0]:
                            lon_gap = lon2_range[0] - lon1_range[1]
                        elif lon2_range[1] < lon1_range[0]:
                            lon_gap = lon1_range[0] - lon2_range[1]
                        
                        if lat_gap > resolution * 0.5 or lon_gap > resolution * 0.5:
                            gaps_info['gaps'].append({
                                'datasets': (i, j),
                                'lat_gap': lat_gap,
                                'lon_gap': lon_gap,
                                'gap_size_pixels': (lat_gap/resolution, lon_gap/resolution)
                            })
    
    # 输出分析结果
    print(f"数据集总数: {gaps_info['total_datasets']}")
    print(f"发现重叠区域: {len(gaps_info['overlaps'])} 个")
    print(f"发现边界缺口: {len(gaps_info['gaps'])} 个")
    
    if gaps_info['gaps']:
        print("\n边界缺口详情:")
        for gap in gaps_info['gaps']:
            i, j = gap['datasets']
            print(f"  数据集 {i} - {j}: 纬向缺口 {gap['lat_gap']:.4f}°, "
                  f"经向缺口 {gap['lon_gap']:.4f}°")
    
    return gaps_info

def merge_netcdf_files_optimized(netcdf_files, output_file, chunk_size=None, 
                                enable_boundary_optimization=True):
    """
    优化版的NetCDF文件合并函数，包含边界处理
    
    Parameters:
    netcdf_files (list): NetCDF文件路径列表
    output_file (str): 输出合并文件路径
    chunk_size (dict): xarray的chunk参数，用于处理大文件
    enable_boundary_optimization (bool): 是否启用边界优化
    
    Returns:
    xarray.Dataset: 合并后的数据集
    """
    print(f"\n=== 优化版NetCDF文件合并 ===")
    print(f"要合并的文件数量: {len(netcdf_files)}")
    print(f"输出文件: {output_file}")
    print(f"边界优化: {'启用' if enable_boundary_optimization else '禁用'}")
    
    if not netcdf_files:
        print("没有找到要合并的文件")
        return None
    
    try:
        # 读取所有NetCDF文件
        datasets = []
        
        for i, file_path in enumerate(netcdf_files):
            print(f"读取文件 {i+1}/{len(netcdf_files)}: {Path(file_path).name}")
            
            if chunk_size:
                ds = xr.open_dataset(file_path, chunks=chunk_size)
            else:
                ds = xr.open_dataset(file_path)
            
            datasets.append(ds)
        
        # 分析边界缺口（如果启用边界优化）
        if enable_boundary_optimization:
            gaps_analysis = analyze_boundary_gaps(datasets)
        
        print("正在合并数据集...")
        
        # 使用优化的合并方法
        if enable_boundary_optimization:
            merged_ds = create_unified_grid(datasets)
        else:
            # 传统合并方法
            try:
                merged_ds = xr.merge(datasets)
            except Exception as e:
                print(f"传统merge失败: {e}")
                merged_ds = create_unified_grid(datasets)
        
        # 添加全局属性
        merged_ds.attrs.update({
            'title': 'Optimized Merged DEM data',
            'description': f'Combined from {len(netcdf_files)} individual DEM tiles with boundary optimization',
            'creation_date': str(np.datetime64('now')),
            'source_files_count': len(netcdf_files),
            'processing_info': 'Merged using optimized algorithm with boundary gap filling',
            'boundary_optimization_enabled': 'true' if enable_boundary_optimization else 'false'
        })
        
        # 保存合并后的文件
        print("保存合并后的文件...")
        if chunk_size:
            merged_ds.to_netcdf(output_file, encoding={'elevation': {'zlib': True, 'complevel': 5}})
        else:
            merged_ds.to_netcdf(output_file)
        
        print(f"✓ 成功合并并保存到: {output_file}")
        
        # 显示合并后的数据信息
        print("\n=== 优化合并结果信息 ===")
        if 'elevation' in merged_ds:
            elevation = merged_ds['elevation']
            print(f"数据变量: elevation")
            print(f"数据类型: {elevation.dtype}")
            print(f"数据形状: {elevation.shape}")
            print(f"文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
            
            # 显示空间范围
            if 'lat' in merged_ds.coords and 'lon' in merged_ds.coords:
                lat_range = (merged_ds.lat.min().values, merged_ds.lat.max().values)
                lon_range = (merged_ds.lon.min().values, merged_ds.lon.max().values)
                print(f"纬度范围: {lat_range[0]:.4f}° - {lat_range[1]:.4f}°")
                print(f"经度范围: {lon_range[0]:.4f}° - {lon_range[1]:.4f}°")
        
        # 关闭数据集
        for ds in datasets:
            ds.close()
        
        return merged_ds
        
    except Exception as e:
        print(f"优化合并文件时发生错误: {e}")
        return None

def batch_process_and_merge(input_dir, output_dir, final_output_file, 
                          resolution=0.01, method='vectorized', chunk_size=None):
    """
    批量处理DEM文件并合并的完整流程
    
    Parameters:
    input_dir (str): 输入DEM文件目录
    output_dir (str): 临时NetCDF文件输出目录
    final_output_file (str): 最终合并文件路径
    resolution (float): 目标网格分辨率
    method (str): 插值方法
    chunk_size (dict): 处理大文件时的分块大小
    
    Returns:
    str: 最终输出文件路径
    """
    print("=== 开始批量处理和合并DEM数据 ===")
    
    # 步骤1: 批量处理TIF文件转换为NetCDF
    processed_files = process_dem_directory(input_dir, output_dir, resolution, method)
    
    if not processed_files:
        print("没有成功处理的文件，无法进行合并")
        return None
    
    # 步骤2: 使用优化版合并所有NetCDF文件
    merged_dataset = merge_netcdf_files_optimized(
        processed_files, final_output_file, chunk_size, 
        enable_boundary_optimization=True
    )
    
    if merged_dataset is not None:
        print(f"\n=== 全部处理完成 ===")
        print(f"最终输出文件: {final_output_file}")
        return final_output_file
    else:
        print("合并过程失败")
        return None

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

def main_batch():
    """
    批量处理主函数
    """
    # 设置路径
    input_dir = r'h:\data\DEM'
    output_dir = r'h:\data\DEM\netcdf_output'
    final_output_file = r'h:\data\DEM\merged_dem_data.nc'
    
    # 处理参数
    resolution = 0.01  # 目标分辨率 0.01度
    method = 'vectorized'  # 使用最快的插值方法
    
    # 对于大文件，可以设置分块处理
    chunk_size = {'lat': 1000, 'lon': 1000}  # 根据内存情况调整
    
    # 执行批量处理和合并
    result = batch_process_and_merge(
        input_dir=input_dir,
        output_dir=output_dir,
        final_output_file=final_output_file,
        resolution=resolution,
        method=method,
        chunk_size=chunk_size
    )
    
    if result:
        print(f"\n🎉 所有处理完成！最终文件保存在: {result}")
    else:
        print("\n❌ 处理过程中出现错误")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'batch':
        main_batch()
    else:
        main() 