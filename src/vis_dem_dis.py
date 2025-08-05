
from os import path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import pandas as pd
import time
from sklearn.neighbors import NearestNeighbors


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

path_csv = path.join(path.dirname(__file__), '../data/station_vis_all_estimated.csv')
# 读取站点数据
df_station = pd.read_csv(path_csv)
print("=== 站点数据信息 ===")
print(df_station.info())
print(df_station.head())

# 读取DEM数据
ds_dem = xr.open_dataset(r'h:\data\DEM\merged_dem_data.nc')
print("\n=== DEM数据信息 ===")
print(ds_dem)

def deg2km(lat1, lon1, lat2, lon2):
    """
    将经纬度差值转换为近似的欧氏距离（公里）
    使用简化的球面距离计算
    
    Parameters:
    lat1, lon1, lat2, lon2: float, 两点的经纬度
    
    Returns:
    float: 距离（公里）
    """
    # 平均纬度（用于经度距离计算）
    avg_lat = np.radians((lat1 + lat2) / 2)
    
    # 纬度和经度的差值（转为弧度）
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    # 地球半径（公里）
    R = 6371.0
    
    # 计算欧氏距离的近似值
    dx = R * dlon * np.cos(avg_lat)  # 经向距离
    dy = R * dlat                   # 纬向距离
    
    return np.sqrt(dx**2 + dy**2)



def anisotropic_idw_interpolation(df_station, target_lons, target_lats, target_elevations, 
                                 beta=10.0, power=2.0, n_neighbors=6):
    """
    各向异性反距离权重插值
    
    Parameters:
    df_station: DataFrame, 站点数据，包含 lon, lat, vis, altitude 字段
    target_lons: array, 目标点经度
    target_lats: array, 目标点纬度  
    target_elevations: array, 目标点海拔
    beta: float, 垂直方向权重放大因子，默认10.0
    power: float, 权重幂次，默认2.0
    n_neighbors: int, 使用的邻居数量，默认6
    
    Returns:
    array: 插值后的能见度值
    """
    print(f"开始各向异性IDW插值，使用{n_neighbors}个最近邻居，β={beta}, p={power}")
    
    # 提取站点数据
    station_lons = df_station['lon'].values
    station_lats = df_station['lat'].values  
    station_vis = df_station['vis'].values
    station_alts = df_station['altitude'].values
    
    # 检查数据有效性
    valid_mask = (~np.isnan(station_vis)) & (~np.isnan(station_alts))
    station_lons = station_lons[valid_mask]
    station_lats = station_lats[valid_mask]
    station_vis = station_vis[valid_mask]
    station_alts = station_alts[valid_mask]
    
    print(f"有效站点数量: {len(station_lons)}")
    
    # 将站点经纬度转换为平面坐标（用于快速邻居搜索）
    station_coords_2d = np.column_stack([station_lons, station_lats])
    
    # 建立最近邻搜索器（仅用于快速筛选候选点）
    nbrs = NearestNeighbors(n_neighbors=min(n_neighbors*2, len(station_lons)), 
                           algorithm='ball_tree').fit(station_coords_2d)
    
    # 目标点数量
    n_targets = len(target_lons)
    interpolated_vis = np.full(n_targets, np.nan)
    
    print(f"开始插值 {n_targets} 个目标点...")
    
    # 批处理以提高效率
    batch_size = 1000
    for batch_start in range(0, n_targets, batch_size):
        batch_end = min(batch_start + batch_size, n_targets)
        batch_size_actual = batch_end - batch_start
        
        if batch_start % 5000 == 0:
            print(f"处理进度: {batch_start}/{n_targets} ({batch_start/n_targets*100:.1f}%)")
        
        # 批量目标点坐标
        batch_target_coords = np.column_stack([
            target_lons[batch_start:batch_end],
            target_lats[batch_start:batch_end]
        ])
        
        # 对每个批次中的目标点，找到候选邻居
        distances_2d, indices = nbrs.kneighbors(batch_target_coords)
        
        # 对批次中的每个目标点进行插值
        for i in range(batch_size_actual):
            target_idx = batch_start + i
            target_lon = target_lons[target_idx]
            target_lat = target_lats[target_idx]
            target_elev = target_elevations[target_idx]
            
            # 跳过目标点海拔为NaN的情况
            if np.isnan(target_elev):
                continue
            
            # 获取候选邻居的索引
            candidate_indices = indices[i]
            
            # 计算各向异性距离
            aniso_distances = []
            valid_candidates = []
            
            for idx in candidate_indices:
                station_lon = station_lons[idx]
                station_lat = station_lats[idx]
                station_alt = station_alts[idx]
                
                # 计算水平距离（转换为公里）
                horizontal_dist = deg2km(target_lat, target_lon, station_lat, station_lon)
                
                # 计算垂直距离（转换为公里，1000m = 1km）
                vertical_dist = abs(target_elev - station_alt) / 1000.0
                
                # 各向异性距离
                aniso_dist = np.sqrt(horizontal_dist**2 + (beta * vertical_dist)**2)
                
                if aniso_dist > 0:  # 避免除零
                    aniso_distances.append(aniso_dist)
                    valid_candidates.append(idx)
            
            # 选择最近的n_neighbors个点
            if len(aniso_distances) >= n_neighbors:
                # 排序并选择最近的n_neighbors个
                sorted_indices = np.argsort(aniso_distances)[:n_neighbors]
                selected_distances = np.array(aniso_distances)[sorted_indices]
                selected_candidates = np.array(valid_candidates)[sorted_indices]
                
                # 计算权重
                weights = 1.0 / (selected_distances ** power)
                
                # 加权平均
                weighted_vis = np.sum(weights * station_vis[selected_candidates])
                total_weight = np.sum(weights)
                
                interpolated_vis[target_idx] = weighted_vis / total_weight
            elif len(aniso_distances) > 0:
                # 如果候选点不足n_neighbors个，使用所有可用的点
                weights = 1.0 / (np.array(aniso_distances) ** power)
                weighted_vis = np.sum(weights * station_vis[valid_candidates])
                total_weight = np.sum(weights)
                interpolated_vis[target_idx] = weighted_vis / total_weight
    
    print("插值完成!")
    valid_count = np.sum(~np.isnan(interpolated_vis))
    print(f"成功插值点数: {valid_count}/{n_targets} ({valid_count/n_targets*100:.1f}%)")
    
    return interpolated_vis

def create_visibility_grid(df_station, ds_dem, beta=10.0, power=2.0, n_neighbors=6):
    """
    创建与DEM网格一致的能见度插值网格
    
    Parameters:
    df_station: DataFrame, 站点数据
    ds_dem: xarray.Dataset, DEM数据
    beta: float, 垂直方向权重放大因子
    power: float, 权重幂次
    n_neighbors: int, 使用的邻居数量
    
    Returns:
    xarray.DataArray: 插值后的能见度网格
    """
    print("=== 创建能见度插值网格 ===")
    
    # 获取DEM的坐标信息
    dem_lons = ds_dem.lon.values
    dem_lats = ds_dem.lat.values
    dem_elevation = ds_dem['elevation'].values
    
    print(f"目标网格大小: {len(dem_lats)} × {len(dem_lons)}")
    print(f"经度范围: {dem_lons.min():.4f}° - {dem_lons.max():.4f}°")
    print(f"纬度范围: {dem_lats.min():.4f}° - {dem_lats.max():.4f}°")
    
    # 创建目标点坐标网格
    lon_grid, lat_grid = np.meshgrid(dem_lons, dem_lats)
    
    # 将网格展平为一维数组
    target_lons = lon_grid.flatten()
    target_lats = lat_grid.flatten()
    target_elevations = dem_elevation.flatten()
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行各向异性IDW插值
    interpolated_vis = anisotropic_idw_interpolation(
        df_station, target_lons, target_lats, target_elevations,
        beta=beta, power=power, n_neighbors=n_neighbors
    )
    
    # 将结果重塑为网格形状
    vis_grid = interpolated_vis.reshape(lon_grid.shape)
    
    # 创建xarray DataArray
    vis_da = xr.DataArray(
        vis_grid,
        coords={
            'lat': ('lat', dem_lats),
            'lon': ('lon', dem_lons)
        },
        dims=['lat', 'lon'],
        name='visibility',
        attrs={
            'units': 'km',
            'long_name': 'Visibility',
            'interpolation_method': 'Anisotropic IDW',
            'beta': beta,
            'power': power,
            'n_neighbors': n_neighbors,
            'description': '各向异性反距离权重插值得到的能见度'
        }
    )
    
    elapsed_time = time.time() - start_time
    print(f"插值完成! 耗时: {elapsed_time/60:.2f}分钟")
    
    return vis_da

def visualize_visibility_result(df_station, vis_da, ds_dem, save_path=None):
    """
    可视化能见度插值结果
    
    Parameters:
    df_station: DataFrame, 站点数据
    vis_da: xarray.DataArray, 插值后的能见度网格
    ds_dem: xarray.Dataset, DEM数据
    save_path: str, 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    # 设置中文显示
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取地理范围
    lon_min, lon_max = vis_da.lon.min().values, vis_da.lon.max().values
    lat_min, lat_max = vis_da.lat.min().values, vis_da.lat.max().values
    extent = [lon_min, lon_max, lat_min, lat_max]
    
    # 1. 绘制DEM数据
    ax1 = axes[0, 0]
    dem_plot = ds_dem['elevation'].plot(ax=ax1, cmap='terrain', 
                                       transform=ccrs.PlateCarree(),
                                       add_colorbar=False)
    ax1.coastlines(resolution='50m', alpha=0.8)
    ax1.gridlines(draw_labels=True, alpha=0.5)
    ax1.set_title('DEM地形数据', fontsize=14)
    ax1.set_extent(extent, crs=ccrs.PlateCarree())
    plt.colorbar(dem_plot, ax=ax1, label='海拔 (m)', shrink=0.8)
    
    # 2. 绘制站点分布
    ax2 = axes[0, 1]
    # 添加地形作为背景
    ds_dem['elevation'].plot(ax=ax2, cmap='terrain', alpha=0.6,
                            transform=ccrs.PlateCarree(), add_colorbar=False)
    # 绘制站点
    scatter = ax2.scatter(df_station['lon'], df_station['lat'], 
                         c=df_station['vis'], s=30, cmap='viridis',
                         transform=ccrs.PlateCarree(), edgecolors='white', linewidth=0.5)
    ax2.coastlines(resolution='50m', alpha=0.8)
    ax2.gridlines(draw_labels=True, alpha=0.5)
    ax2.set_title(f'气象站点分布 (共{len(df_station)}个站点)', fontsize=14)
    ax2.set_extent(extent, crs=ccrs.PlateCarree())
    plt.colorbar(scatter, ax=ax2, label='能见度 (km)', shrink=0.8)
    
    # 3. 绘制插值后的能见度场
    ax3 = axes[1, 0]
    vis_plot = vis_da.plot(ax=ax3, cmap='viridis', 
                          transform=ccrs.PlateCarree(),
                          add_colorbar=False)
    ax3.coastlines(resolution='50m', alpha=0.8)
    ax3.gridlines(draw_labels=True, alpha=0.5)
    ax3.set_title('各向异性IDW插值结果', fontsize=14)
    ax3.set_extent(extent, crs=ccrs.PlateCarree())
    plt.colorbar(vis_plot, ax=ax3, label='能见度 (km)', shrink=0.8)
    
    # 4. 绘制插值结果叠加站点
    ax4 = axes[1, 1]
    vis_da.plot(ax=ax4, cmap='viridis', alpha=0.8,
               transform=ccrs.PlateCarree(), add_colorbar=False)
    # 叠加站点
    ax4.scatter(df_station['lon'], df_station['lat'], 
               c='white', s=20, transform=ccrs.PlateCarree(), 
               edgecolors='black', linewidth=0.5)
    ax4.coastlines(resolution='50m', alpha=0.8)
    ax4.gridlines(draw_labels=True, alpha=0.5)
    ax4.set_title('插值结果 + 站点位置', fontsize=14)
    ax4.set_extent(extent, crs=ccrs.PlateCarree())
    
    plt.tight_layout()
    
    # 显示统计信息
    print("\n=== 插值结果统计 ===")
    print(f"插值网格大小: {vis_da.shape}")
    print(f"能见度范围: {vis_da.min().values:.2f} - {vis_da.max().values:.2f} km")
    print(f"平均能见度: {vis_da.mean().values:.2f} km")
    print(f"有效数据比例: {(~np.isnan(vis_da.values)).sum() / vis_da.size * 100:.1f}%")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()

# 主执行代码
if __name__ == "__main__":
    # 创建能见度插值网格
    vis_grid = create_visibility_grid(
        df_station, ds_dem, 
        beta=10.0,      # 垂直方向权重放大因子
        power=2.0,      # 权重幂次
        n_neighbors=6   # 使用最近的6个邻居
    )
    
    # 保存插值结果
    output_file = 'visibility_anisotropic_idw.nc'
    vis_grid.to_netcdf(output_file)
    print(f"插值结果已保存到: {output_file}")
    
    # 可视化结果
    visualize_visibility_result(df_station, vis_grid, ds_dem, 
                               save_path='visibility_interpolation_result.png')




