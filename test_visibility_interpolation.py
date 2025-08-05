#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
能见度各向异性IDW插值测试脚本
测试主要功能是否正常工作
"""

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_test_data():
    """创建测试数据"""
    print("创建测试数据...")
    
    # 创建简单的测试站点数据
    np.random.seed(42)
    n_stations = 20
    
    # 在一个小区域内创建站点
    lon_range = (113.0, 114.0)
    lat_range = (23.0, 24.0)
    
    test_stations = pd.DataFrame({
        'lon': np.random.uniform(lon_range[0], lon_range[1], n_stations),
        'lat': np.random.uniform(lat_range[0], lat_range[1], n_stations),
        'vis': np.random.uniform(5.0, 25.0, n_stations),  # 能见度 5-25km
        'altitude': np.random.uniform(50.0, 500.0, n_stations)  # 海拔 50-500m
    })
    
    # 创建简单的测试DEM数据
    lons = np.linspace(lon_range[0], lon_range[1], 50)
    lats = np.linspace(lat_range[0], lat_range[1], 50)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # 创建简单的地形（中心高，四周低）
    center_lon, center_lat = (lon_range[0] + lon_range[1])/2, (lat_range[0] + lat_range[1])/2
    elevation = 100 + 200 * np.exp(-((lon_grid - center_lon)**2 + (lat_grid - center_lat)**2) * 100)
    
    test_dem = xr.Dataset({
        'elevation': (('lat', 'lon'), elevation)
    }, coords={
        'lat': lats,
        'lon': lons
    })
    
    print(f"测试站点数量: {len(test_stations)}")
    print(f"测试DEM网格大小: {test_dem.elevation.shape}")
    
    return test_stations, test_dem

def test_basic_functions():
    """测试基本功能"""
    print("\n=== 测试基本功能 ===")
    
    # 导入我们的模块
    import sys
    sys.path.append('src')
    
    try:
        from vis_dem_dis import deg2km, anisotropic_idw_interpolation
        print("✓ 成功导入插值函数")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    # 测试距离计算函数
    try:
        dist = deg2km(23.0, 113.0, 23.1, 113.1)
        print(f"✓ 距离计算函数正常，0.1度距离约为 {dist:.2f} km")
    except Exception as e:
        print(f"✗ 距离计算函数错误: {e}")
        return False
    
    return True

def test_interpolation():
    """测试插值功能"""
    print("\n=== 测试插值功能 ===")
    
    # 创建测试数据
    df_test, ds_test = create_test_data()
    
    # 导入插值函数
    import sys
    sys.path.append('src')
    from vis_dem_dis import create_visibility_grid, visualize_visibility_result
    
    try:
        # 执行插值（使用较小的参数以加快测试速度）
        print("执行各向异性IDW插值...")
        vis_result = create_visibility_grid(
            df_test, ds_test,
            beta=5.0,       # 较小的beta值
            power=2.0,      # 标准power值
            n_neighbors=6   # 6个邻居
        )
        
        print("✓ 插值执行成功")
        print(f"插值结果形状: {vis_result.shape}")
        print(f"插值结果范围: {vis_result.min().values:.2f} - {vis_result.max().values:.2f} km")
        
        # 简单可视化（不包含地图投影以避免复杂依赖）
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始站点数据
        scatter = axes[0].scatter(df_test['lon'], df_test['lat'], 
                                 c=df_test['vis'], s=50, cmap='viridis')
        axes[0].set_title('测试站点分布')
        axes[0].set_xlabel('经度')
        axes[0].set_ylabel('纬度')
        plt.colorbar(scatter, ax=axes[0], label='能见度 (km)')
        
        # DEM数据
        dem_plot = axes[1].imshow(ds_test.elevation.values, cmap='terrain', 
                                 extent=[ds_test.lon.min(), ds_test.lon.max(),
                                        ds_test.lat.min(), ds_test.lat.max()],
                                 aspect='auto', origin='lower')
        axes[1].set_title('测试DEM数据')
        axes[1].set_xlabel('经度')
        axes[1].set_ylabel('纬度')
        plt.colorbar(dem_plot, ax=axes[1], label='海拔 (m)')
        
        # 插值结果
        vis_plot = axes[2].imshow(vis_result.values, cmap='viridis',
                                 extent=[vis_result.lon.min(), vis_result.lon.max(),
                                        vis_result.lat.min(), vis_result.lat.max()],
                                 aspect='auto', origin='lower')
        # 叠加站点位置
        axes[2].scatter(df_test['lon'], df_test['lat'], 
                       c='white', s=20, edgecolors='black', linewidth=0.5)
        axes[2].set_title('插值结果')
        axes[2].set_xlabel('经度')
        axes[2].set_ylabel('纬度')
        plt.colorbar(vis_plot, ax=axes[2], label='能见度 (km)')
        
        plt.tight_layout()
        plt.savefig('test_interpolation_result.png', dpi=150, bbox_inches='tight')
        print("✓ 测试结果图已保存为 test_interpolation_result.png")
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"✗ 插值测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=== 能见度各向异性IDW插值功能测试 ===")
    
    # 测试基本功能
    if not test_basic_functions():
        print("基本功能测试失败，终止测试")
        return
    
    # 测试插值功能
    if test_interpolation():
        print("\n🎉 所有测试通过！各向异性IDW插值功能正常工作")
    else:
        print("\n❌ 插值功能测试失败")

if __name__ == "__main__":
    main()