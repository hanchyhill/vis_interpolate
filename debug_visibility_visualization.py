#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门用于调试绘制 visibility_anisotropic_idw.nc 的可视化工具

功能特点：
1. 非线性色彩映射，突出低能见度事件
2. 分层显示：雾(0-1000)、轻雾(1000-10000)、无雾(>10000)
3. 多细分渐变，强调雾区域
4. 多种可视化模式对比
"""

from os import path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

# 广东省shapefile路径
GUANGDONG_SHP_PATH = r"D:\Document\气象台\GIS\ChinaAdminDivisonSHP-master\中国省市县和乡镇行政区划\广东省\广东省_省界.shp"


def load_guangdong_boundary():
    """
    加载广东省边界shapefile
    """
    try:
        if not path.exists(GUANGDONG_SHP_PATH):
            print(f"警告：找不到广东省shapefile文件: {GUANGDONG_SHP_PATH}")
            return None
        
        gdf = gpd.read_file(GUANGDONG_SHP_PATH)
        print(f"成功加载广东省边界，共{len(gdf)}个多边形")
        return gdf
    except Exception as e:
        print(f"加载广东省边界时出错: {e}")
        return None


def apply_guangdong_mask(vis_data, gdf_guangdong):
    """
    应用广东省遮罩，只保留广东省范围内的数据
    
    Parameters:
    vis_data: xarray.DataArray, 能见度数据
    gdf_guangdong: GeoDataFrame, 广东省边界
    
    Returns:
    xarray.DataArray, 遮罩后的能见度数据
    """
    if gdf_guangdong is None:
        print("广东省边界数据为空，返回原始数据")
        return vis_data
    
    try:
        # 创建经纬度网格
        lon_grid, lat_grid = np.meshgrid(vis_data.lon.values, vis_data.lat.values)
        
        # 创建点集合
        points = [Point(lon, lat) for lon, lat in zip(lon_grid.flatten(), lat_grid.flatten())]
        
        # 创建遮罩
        mask = np.zeros(len(points), dtype=bool)
        
        # 检查每个点是否在广东省边界内
        for i, point in enumerate(points):
            mask[i] = gdf_guangdong.geometry.contains(point).any()
        
        # 重塑遮罩为原始网格形状
        mask_grid = mask.reshape(lon_grid.shape)
        
        # 应用遮罩
        masked_data = vis_data.where(mask_grid)
        
        print(f"遮罩应用完成，保留了 {mask.sum()} / {len(mask)} 个网格点")
        
        return masked_data
        
    except Exception as e:
        print(f"应用广东省遮罩时出错: {e}")
        return vis_data


def create_visibility_colormap():
    """
    创建专门的能见度色彩映射
    分为三个区间：雾(0-1000)、轻雾(1000-10000)、无雾(>10000)
    """
    # 定义颜色区间和对应的颜色
    # 雾区域(0-1000)：深红到橙色系，突出显示
    fog_colors = [
        '#8B0000',  # 深红色
        '#B22222',  # 火砖色
        '#DC143C',  # 深红色
        '#FF0000',  # 红色
        '#FF4500',  # 橙红色
        '#FF6347',  # 番茄色
        '#FF7F50',  # 珊瑚色
        '#FFA500',  # 橙色
    ]
    
    # 轻雾区域(1000-10000)：黄色到绿色系
    light_fog_colors = [
        '#FFD700',  # 金色
        '#FFFF00',  # 黄色
        '#ADFF2F',  # 绿黄色
        '#9ACD32',  # 黄绿色
        '#7CFC00',  # 草绿色
        '#32CD32',  # 石灰绿
        '#00FF00',  # 绿色
        '#00CED1',  # 深青色
    ]
    
    # 无雾区域(>10000)：蓝色系
    no_fog_colors = [
        '#00BFFF',  # 深天蓝
        '#87CEEB',  # 天蓝色
        '#87CEFA',  # 淡天蓝
        '#ADD8E6',  # 淡蓝色
        '#E0F6FF',  # 爱丽丝蓝
        '#F0F8FF',  # 爱丽丝蓝
    ]
    
    # 合并所有颜色
    all_colors = fog_colors + light_fog_colors + no_fog_colors
    
    # 创建颜色映射
    n_colors = len(all_colors)
    cmap = ListedColormap(all_colors, name='visibility_custom')
    
    return cmap, fog_colors, light_fog_colors, no_fog_colors


def create_nonlinear_norm(vmin=0, vmax=20):
    """
    创建非线性的色彩标准化，突出低能见度区域
    """
    # 定义分段边界，给雾区域分配更多的色彩级别（单位：千米）
    boundaries = (
        # 雾区域(0-1km)：细分为8个等级
        list(np.linspace(0, 1, 9)) +
        # 轻雾区域(1-10km)：细分为8个等级
        list(np.linspace(1, 10, 9)[1:]) +
        # 无雾区域(10-20km)：分为6个等级
        list(np.linspace(10, vmax, 7)[1:])
    )
    
    # 移除重复的边界值
    boundaries = sorted(list(set(boundaries)))
    
    # 创建边界归一化
    norm = BoundaryNorm(boundaries, ncolors=len(boundaries)-1)
    
    return norm, boundaries


def load_visibility_data(data_path='src/visibility_anisotropic_idw.nc'):
    """
    加载能见度数据
    """
    if not path.exists(data_path):
        print(f"错误：找不到文件 {data_path}")
        print("请确保已运行插值程序生成该文件")
        return None
    
    try:
        # 加载数据
        ds = xr.open_dataset(data_path)
        print(f"成功加载数据: {data_path}")
        print(f"数据维度: {ds.dims}")
        print(f"数据变量: {list(ds.data_vars)}")
        
        # 获取能见度数据
        if 'visibility' in ds:
            vis_data = ds['visibility']
        elif 'vis000' in ds:
            vis_data = ds['vis000'][0,0,:,:]
        else:
            # 尝试获取第一个数据变量
            var_name = list(ds.data_vars)[0]
            vis_data = ds[var_name]
            print(f"使用数据变量: {var_name}")
        
        # 转换单位如果需要（从m转换为km）
        if vis_data.max() > 100:  # 如果最大值大于100，可能是米单位
            vis_data = vis_data / 1000
            print("已将单位从米转换为千米")
        
        print(f"能见度数据范围: {vis_data.min().values:.2f} - {vis_data.max().values:.2f} km")
        
        return vis_data
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None


def create_comprehensive_visualization(vis_data, save_path='debug_visibility_comprehensive.png'):
    """
    创建综合的能见度可视化图，包含多种显示模式
    """
    # 加载广东省边界
    gdf_guangdong = load_guangdong_boundary()
    
    # 应用广东省遮罩
    vis_data_masked = apply_guangdong_mask(vis_data, gdf_guangdong)
    
    # 创建自定义色彩映射和归一化
    cmap, fog_colors, light_fog_colors, no_fog_colors = create_visibility_colormap()
    norm, boundaries = create_nonlinear_norm(vmin=0, vmax=20)
    
    # 创建子图
    fig = plt.figure(figsize=(20, 16))
    
    # 获取地理范围（基于原始数据，但会根据广东省边界调整）
    if gdf_guangdong is not None:
        # 使用广东省边界确定显示范围
        bounds = gdf_guangdong.total_bounds  # [minx, miny, maxx, maxy]
        extent = [bounds[0]-0.5, bounds[2]+0.5, bounds[1]-0.5, bounds[3]+0.5]
    else:
        # 使用原始数据范围
        lon_min, lon_max = vis_data.lon.min().values, vis_data.lon.max().values
        lat_min, lat_max = vis_data.lat.min().values, vis_data.lat.max().values
        extent = [lon_min, lon_max, lat_min, lat_max]
    
    # 1. 主要的非线性色彩映射图
    ax1 = plt.subplot(2, 3, 1, projection=ccrs.PlateCarree())
    im1 = vis_data_masked.plot(ax=ax1, cmap=cmap, norm=norm, 
                       transform=ccrs.PlateCarree(), add_colorbar=False)
    ax1.coastlines(resolution='50m', alpha=0.8)
    ax1.gridlines(draw_labels=True, alpha=0.5, linewidth=0.5)
    # 添加广东省边界
    if gdf_guangdong is not None:
        gdf_guangdong.plot(ax=ax1, facecolor='none', edgecolor='black', 
                          linewidth=2, transform=ccrs.PlateCarree())
    ax1.set_title('非线性色彩映射\n（突出雾区域）', fontsize=12, pad=10)
    ax1.set_extent(extent, crs=ccrs.PlateCarree())
    
    # 添加自定义色标
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, pad=0.05)
    cbar1.set_label('能见度 (km)', fontsize=10)
    
    # 2. 传统线性映射对比
    ax2 = plt.subplot(2, 3, 2, projection=ccrs.PlateCarree())
    im2 = vis_data_masked.plot(ax=ax2, cmap='viridis', vmin=0, vmax=20,
                       transform=ccrs.PlateCarree(), add_colorbar=False)
    ax2.coastlines(resolution='50m', alpha=0.8)
    ax2.gridlines(draw_labels=True, alpha=0.5, linewidth=0.5)
    # 添加广东省边界
    if gdf_guangdong is not None:
        gdf_guangdong.plot(ax=ax2, facecolor='none', edgecolor='black', 
                          linewidth=2, transform=ccrs.PlateCarree())
    ax2.set_title('传统线性映射\n（对比参考）', fontsize=12, pad=10)
    ax2.set_extent(extent, crs=ccrs.PlateCarree())
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, pad=0.05)
    cbar2.set_label('能见度 (km)', fontsize=10)
    
    # 3. 雾区域突出显示（0-1km）
    ax3 = plt.subplot(2, 3, 3, projection=ccrs.PlateCarree())
    # 创建掩膜，只显示雾区域
    fog_mask = vis_data_masked.where(vis_data_masked <= 1.0)  # 只显示雾区域
    fog_cmap = LinearSegmentedColormap.from_list('fog', fog_colors, N=256)
    im3 = fog_mask.plot(ax=ax3, cmap=fog_cmap, vmin=0, vmax=1.0,
                       transform=ccrs.PlateCarree(), add_colorbar=False)
    ax3.coastlines(resolution='50m', alpha=0.8)
    ax3.gridlines(draw_labels=True, alpha=0.5, linewidth=0.5)
    # 添加广东省边界
    if gdf_guangdong is not None:
        gdf_guangdong.plot(ax=ax3, facecolor='none', edgecolor='black', 
                          linewidth=2, transform=ccrs.PlateCarree())
    ax3.set_title('雾区域详细显示\n（0-1km）', fontsize=12, pad=10)
    ax3.set_extent(extent, crs=ccrs.PlateCarree())
    
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8, pad=0.05)
    cbar3.set_label('能见度 (km)', fontsize=10)
    
    # 4. 轻雾区域显示（1-10km）
    ax4 = plt.subplot(2, 3, 4, projection=ccrs.PlateCarree())
    light_fog_mask = vis_data_masked.where((vis_data_masked > 1.0) & (vis_data_masked <= 10.0))
    light_fog_cmap = LinearSegmentedColormap.from_list('light_fog', light_fog_colors, N=256)
    im4 = light_fog_mask.plot(ax=ax4, cmap=light_fog_cmap, vmin=1.0, vmax=10.0,
                             transform=ccrs.PlateCarree(), add_colorbar=False)
    ax4.coastlines(resolution='50m', alpha=0.8)
    ax4.gridlines(draw_labels=True, alpha=0.5, linewidth=0.5)
    # 添加广东省边界
    if gdf_guangdong is not None:
        gdf_guangdong.plot(ax=ax4, facecolor='none', edgecolor='black', 
                          linewidth=2, transform=ccrs.PlateCarree())
    ax4.set_title('轻雾区域显示\n（1-10km）', fontsize=12, pad=10)
    ax4.set_extent(extent, crs=ccrs.PlateCarree())
    
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8, pad=0.05)
    cbar4.set_label('能见度 (km)', fontsize=10)
    
    # 5. 无雾区域显示（>10km）
    ax5 = plt.subplot(2, 3, 5, projection=ccrs.PlateCarree())
    no_fog_mask = vis_data_masked.where(vis_data_masked > 10.0)
    no_fog_cmap = LinearSegmentedColormap.from_list('no_fog', no_fog_colors, N=256)
    im5 = no_fog_mask.plot(ax=ax5, cmap=no_fog_cmap, vmin=10.0, vmax=20.0,
                          transform=ccrs.PlateCarree(), add_colorbar=False)
    ax5.coastlines(resolution='50m', alpha=0.8)
    ax5.gridlines(draw_labels=True, alpha=0.5, linewidth=0.5)
    # 添加广东省边界
    if gdf_guangdong is not None:
        gdf_guangdong.plot(ax=ax5, facecolor='none', edgecolor='black', 
                          linewidth=2, transform=ccrs.PlateCarree())
    ax5.set_title('无雾区域显示\n（>10km）', fontsize=12, pad=10)
    ax5.set_extent(extent, crs=ccrs.PlateCarree())
    
    cbar5 = plt.colorbar(im5, ax=ax5, shrink=0.8, pad=0.05)
    cbar5.set_label('能见度 (km)', fontsize=10)
    
    # 6. 分类显示（离散化）
    ax6 = plt.subplot(2, 3, 6, projection=ccrs.PlateCarree())
    
    # 创建分类数据
    vis_classified = xr.where(vis_data_masked <= 1.0, 0,  # 雾
                             xr.where(vis_data_masked <= 10.0, 1,  # 轻雾
                                     2))  # 无雾
    
    # 分类色彩映射
    class_colors = ['#DC143C', '#FFD700', '#87CEEB']  # 红、黄、蓝
    class_cmap = ListedColormap(class_colors)
    class_norm = BoundaryNorm([0, 1, 2, 3], ncolors=3)
    
    im6 = vis_classified.plot(ax=ax6, cmap=class_cmap, norm=class_norm,
                             transform=ccrs.PlateCarree(), add_colorbar=False)
    ax6.coastlines(resolution='50m', alpha=0.8)
    ax6.gridlines(draw_labels=True, alpha=0.5, linewidth=0.5)
    # 添加广东省边界
    if gdf_guangdong is not None:
        gdf_guangdong.plot(ax=ax6, facecolor='none', edgecolor='black', 
                          linewidth=2, transform=ccrs.PlateCarree())
    ax6.set_title('分类显示\n（雾/轻雾/无雾）', fontsize=12, pad=10)
    ax6.set_extent(extent, crs=ccrs.PlateCarree())
    
    # 自定义色标
    cbar6 = plt.colorbar(im6, ax=ax6, shrink=0.8, pad=0.05, ticks=[0.5, 1.5, 2.5])
    cbar6.set_ticklabels(['雾\n(≤1km)', '轻雾\n(1-10km)', '无雾\n(>10km)'])
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"综合可视化图已保存至: {save_path}")
    
    return fig


def create_detailed_fog_visualization(vis_data, save_path='debug_visibility_fog_detail.png'):
    """
    创建专门针对雾区域的详细可视化
    """
    # 加载广东省边界
    gdf_guangdong = load_guangdong_boundary()
    
    # 应用广东省遮罩
    vis_data_masked = apply_guangdong_mask(vis_data, gdf_guangdong)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    # 获取地理范围（基于广东省边界）
    if gdf_guangdong is not None:
        bounds = gdf_guangdong.total_bounds
        extent = [bounds[0]-0.5, bounds[2]+0.5, bounds[1]-0.5, bounds[3]+0.5]
    else:
        lon_min, lon_max = vis_data.lon.min().values, vis_data.lon.max().values
        lat_min, lat_max = vis_data.lat.min().values, vis_data.lat.max().values
        extent = [lon_min, lon_max, lat_min, lat_max]
    
    # 1. 0-0.2km（极低能见度）
    ax1 = axes[0, 0]
    very_low_vis = vis_data_masked.where(vis_data_masked <= 0.2)
    im1 = very_low_vis.plot(ax=ax1, cmap='Reds', vmin=0, vmax=0.2,
                           transform=ccrs.PlateCarree(), add_colorbar=False)
    ax1.coastlines(resolution='50m', alpha=0.8)
    ax1.gridlines(draw_labels=True, alpha=0.5)
    # 添加广东省边界  
    if gdf_guangdong is not None:
        gdf_guangdong.plot(ax=ax1, facecolor='none', edgecolor='black', 
                          linewidth=2, transform=ccrs.PlateCarree())
    ax1.set_title('极浓雾 (0-200m)', fontsize=12)
    ax1.set_extent(extent, crs=ccrs.PlateCarree())
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='能见度 (km)')
    
    # 2. 0.2-0.5km（浓雾）
    ax2 = axes[0, 1]
    dense_fog = vis_data_masked.where((vis_data_masked > 0.2) & (vis_data_masked <= 0.5))
    im2 = dense_fog.plot(ax=ax2, cmap='OrRd', vmin=0.2, vmax=0.5,
                        transform=ccrs.PlateCarree(), add_colorbar=False)
    ax2.coastlines(resolution='50m', alpha=0.8)
    ax2.gridlines(draw_labels=True, alpha=0.5)
    # 添加广东省边界
    if gdf_guangdong is not None:
        gdf_guangdong.plot(ax=ax2, facecolor='none', edgecolor='black', 
                          linewidth=2, transform=ccrs.PlateCarree())
    ax2.set_title('浓雾 (200-500m)', fontsize=12)
    ax2.set_extent(extent, crs=ccrs.PlateCarree())
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='能见度 (km)')
    
    # 3. 0.5-1.0km（雾）
    ax3 = axes[1, 0]
    fog = vis_data_masked.where((vis_data_masked > 0.5) & (vis_data_masked <= 1.0))
    im3 = fog.plot(ax=ax3, cmap='YlOrRd', vmin=0.5, vmax=1.0,
                  transform=ccrs.PlateCarree(), add_colorbar=False)
    ax3.coastlines(resolution='50m', alpha=0.8)
    ax3.gridlines(draw_labels=True, alpha=0.5)
    # 添加广东省边界
    if gdf_guangdong is not None:
        gdf_guangdong.plot(ax=ax3, facecolor='none', edgecolor='black', 
                          linewidth=2, transform=ccrs.PlateCarree())
    ax3.set_title('雾 (500m-1km)', fontsize=12)
    ax3.set_extent(extent, crs=ccrs.PlateCarree())
    plt.colorbar(im3, ax=ax3, shrink=0.8, label='能见度 (km)')
    
    # 4. 所有雾区域综合
    ax4 = axes[1, 1]
    all_fog = vis_data_masked.where(vis_data_masked <= 1.0)
    # 使用自定义色彩映射突出不同雾的强度
    fog_colors_grad = ['#8B0000', '#DC143C', '#FF4500', '#FFA500']
    fog_cmap = LinearSegmentedColormap.from_list('detailed_fog', fog_colors_grad, N=256)
    im4 = all_fog.plot(ax=ax4, cmap=fog_cmap, vmin=0, vmax=1.0,
                      transform=ccrs.PlateCarree(), add_colorbar=False)
    ax4.coastlines(resolution='50m', alpha=0.8)
    ax4.gridlines(draw_labels=True, alpha=0.5)
    # 添加广东省边界
    if gdf_guangdong is not None:
        gdf_guangdong.plot(ax=ax4, facecolor='none', edgecolor='black', 
                          linewidth=2, transform=ccrs.PlateCarree())
    ax4.set_title('雾区域综合 (0-1km)', fontsize=12)
    ax4.set_extent(extent, crs=ccrs.PlateCarree())
    plt.colorbar(im4, ax=ax4, shrink=0.8, label='能见度 (km)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"雾区域详细可视化已保存至: {save_path}")
    
    return fig


def print_visibility_statistics(vis_data, title="能见度数据统计分析"):
    """
    打印能见度数据的详细统计信息
    """
    print("=" * 50)
    print(title)
    print("=" * 50)
    
    # 计算有效数据（非NaN值）
    valid_data = vis_data.where(~np.isnan(vis_data))
    total_points = vis_data.size
    valid_points = (~np.isnan(vis_data.values)).sum()
    
    # 基本统计
    print(f"数据维度: {vis_data.shape}")
    print(f"总网格点数: {total_points}")
    print(f"有效数据点数: {valid_points} ({valid_points/total_points*100:.1f}%)")
    
    if valid_points > 0:
        print(f"最小能见度: {vis_data.min().values:.3f} km")
        print(f"最大能见度: {vis_data.max().values:.3f} km")
        print(f"平均能见度: {vis_data.mean().values:.3f} km")
        print(f"能见度中位数: {vis_data.median().values:.3f} km")
        print(f"标准差: {vis_data.std().values:.3f} km")
        
        # 各区间统计（基于有效数据）
        print("\n各能见度区间统计:")
        
        # 雾区域 (0-1km)
        fog_mask = vis_data <= 1.0
        fog_count = fog_mask.sum().values
        fog_percent = (fog_count / valid_points) * 100
        print(f"雾区域 (≤1km): {fog_count} 个网格点 ({fog_percent:.1f}%)")
        
        # 轻雾区域 (1-10km)
        light_fog_mask = (vis_data > 1.0) & (vis_data <= 10.0)
        light_fog_count = light_fog_mask.sum().values
        light_fog_percent = (light_fog_count / valid_points) * 100
        print(f"轻雾区域 (1-10km): {light_fog_count} 个网格点 ({light_fog_percent:.1f}%)")
        
        # 无雾区域 (>10km)
        no_fog_mask = vis_data > 10.0
        no_fog_count = no_fog_mask.sum().values
        no_fog_percent = (no_fog_count / valid_points) * 100
        print(f"无雾区域 (>10km): {no_fog_count} 个网格点 ({no_fog_percent:.1f}%)")
        
        # 详细雾分类
        print("\n详细雾分类:")
        extremely_low = (vis_data <= 0.05).sum().values  # 50m以下
        very_low = ((vis_data > 0.05) & (vis_data <= 0.2)).sum().values  # 50-200m
        low = ((vis_data > 0.2) & (vis_data <= 0.5)).sum().values  # 200-500m
        moderate_fog = ((vis_data > 0.5) & (vis_data <= 1.0)).sum().values  # 500-1000m
        
        print(f"极浓雾 (≤50m): {extremely_low} 个网格点")
        print(f"浓雾 (50-200m): {very_low} 个网格点")
        print(f"中雾 (200-500m): {low} 个网格点")
        print(f"轻雾 (500-1000m): {moderate_fog} 个网格点")
    else:
        print("警告：没有有效数据！")
    
    print("=" * 50)


def main():
    """
    主函数 - 执行完整的能见度数据调试可视化
    """
    print("开始能见度数据调试可视化...")
    
    # 加载数据
    vis_data = load_visibility_data('data/VIS_2025022800.NC')
    if vis_data is None:
        return
    
    # 加载广东省边界（用于统计分析）
    gdf_guangdong = load_guangdong_boundary()
    
    # 应用广东省遮罩
    vis_data_masked = apply_guangdong_mask(vis_data, gdf_guangdong)
    
    # 打印统计信息
    print_visibility_statistics(vis_data, "原始能见度数据统计分析")
    print_visibility_statistics(vis_data_masked, "广东省范围能见度数据统计分析")
    
    # 创建综合可视化
    print("\n创建综合可视化图...")
    create_comprehensive_visualization(vis_data)
    
    # 创建雾区域详细可视化
    print("创建雾区域详细可视化...")
    create_detailed_fog_visualization(vis_data)
    
    print("\n可视化完成！生成的文件：")
    print("1. debug_visibility_comprehensive.png - 综合对比图（含广东省遮罩）")
    print("2. debug_visibility_fog_detail.png - 雾区域详细图（含广东省遮罩）")
    print("\n所有图像已应用广东省边界遮罩，只显示广东省范围内的数据")
    
    # 显示图片
    plt.show()


if __name__ == "__main__":
    main()