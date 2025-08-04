#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEM数据稀疏化测试脚本
"""

import sys
import os
import numpy as np
import xarray as xr

def create_test_dem_data():
    """
    创建测试用的DEM数据
    """
    print("创建测试DEM数据...")
    
    # 创建测试网格 (1°×1°范围，0.001°分辨率)
    lon = np.arange(111.0, 112.0, 0.001)
    lat = np.arange(23.0, 24.0, 0.001)
    
    # 创建网格
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # 创建模拟的高程数据 (简单的正弦波模式)
    elevation = 100 + 50 * np.sin(lon_grid * 10) + 30 * np.cos(lat_grid * 8)
    
    # 添加一些随机噪声
    np.random.seed(42)
    elevation += np.random.normal(0, 5, elevation.shape)
    
    # 创建xarray DataArray
    dem_data = xr.DataArray(
        elevation[np.newaxis, :, :],  # 添加band维度
        coords={
            'band': [1],
            'x': ('x', lon),
            'y': ('y', lat)
        },
        dims=['band', 'y', 'x'],
        name='elevation',
        attrs={
            'units': 'meters',
            'long_name': 'Elevation',
            'description': 'Test DEM data'
        }
    )
    
    print(f"测试DEM数据创建完成:")
    print(f"数据形状: {dem_data.shape}")
    print(f"数据大小: {dem_data.nbytes / 1024 / 1024:.2f} MB")
    
    return dem_data

def test_interpolation():
    """
    测试插值功能
    """
    print("\n=== 开始测试DEM插值功能 ===")
    
    # 创建测试数据
    test_dem = create_test_dem_data()
    
    # 导入插值模块
    try:
        from src.dem_interpolation import create_target_grid, interpolate_dem_to_grid
    except ImportError as e:
        print(f"导入模块失败: {e}")
        return False
    
    # 测试创建目标网格
    print("\n1. 测试创建目标网格...")
    try:
        lon_grid, lat_grid = create_target_grid(test_dem, resolution=0.01)
        print("✓ 目标网格创建成功")
    except Exception as e:
        print(f"✗ 目标网格创建失败: {e}")
        return False
    
    # 测试插值
    print("\n2. 测试插值功能...")
    try:
        interpolated_dem = interpolate_dem_to_grid(test_dem, lon_grid, lat_grid, method='linear')
        print("✓ 插值功能测试成功")
    except Exception as e:
        print(f"✗ 插值功能测试失败: {e}")
        return False
    
    # 验证结果
    print("\n3. 验证插值结果...")
    print(f"原始数据形状: {test_dem.shape}")
    print(f"插值后数据形状: {interpolated_dem.shape}")
    print(f"原始数据大小: {test_dem.nbytes / 1024 / 1024:.2f} MB")
    print(f"插值后数据大小: {interpolated_dem.nbytes / 1024 / 1024:.2f} MB")
    print(f"数据压缩比: {test_dem.nbytes / interpolated_dem.nbytes:.2f}")
    
    # 检查数据范围
    print(f"\n数据统计:")
    print(f"原始数据 - 最小值: {test_dem.min().values:.2f} m, 最大值: {test_dem.max().values:.2f} m")
    print(f"插值后数据 - 最小值: {interpolated_dem.min().values:.2f} m, 最大值: {interpolated_dem.max().values:.2f} m")
    
    # 检查是否有NaN值
    nan_count = np.isnan(interpolated_dem.values).sum()
    print(f"插值后数据中的NaN值数量: {nan_count}")
    
    if nan_count == 0:
        print("✓ 插值结果验证通过")
        return True
    else:
        print("✗ 插值结果包含NaN值")
        return False

def test_different_methods():
    """
    测试不同的插值方法
    """
    print("\n=== 测试不同插值方法 ===")
    
    # 创建测试数据
    test_dem = create_test_dem_data()
    
    try:
        from src.dem_interpolation import create_target_grid, interpolate_dem_to_grid
    except ImportError as e:
        print(f"导入模块失败: {e}")
        return
    
    # 创建目标网格
    lon_grid, lat_grid = create_target_grid(test_dem, resolution=0.01)
    
    # 测试不同方法
    methods = ['linear', 'nearest', 'cubic']
    results = {}
    
    for method in methods:
        print(f"\n测试 {method} 插值方法...")
        try:
            result = interpolate_dem_to_grid(test_dem, lon_grid, lat_grid, method=method)
            results[method] = result
            print(f"✓ {method} 插值成功")
        except Exception as e:
            print(f"✗ {method} 插值失败: {e}")
    
    # 比较结果
    if len(results) > 1:
        print("\n插值方法对比:")
        for method, result in results.items():
            print(f"{method}:")
            print(f"  数据大小: {result.nbytes / 1024 / 1024:.2f} MB")
            print(f"  最小值: {result.min().values:.2f} m")
            print(f"  最大值: {result.max().values:.2f} m")
            print(f"  平均值: {result.mean().values:.2f} m")

def main():
    """
    主测试函数
    """
    print("DEM数据稀疏化功能测试")
    print("=" * 50)
    
    # 测试基本插值功能
    success = test_interpolation()
    
    if success:
        print("\n✓ 基本功能测试通过")
        
        # 测试不同插值方法
        test_different_methods()
        
        print("\n=== 所有测试完成 ===")
        print("DEM数据稀疏化功能正常工作！")
    else:
        print("\n✗ 基本功能测试失败")
        print("请检查代码和依赖包")

if __name__ == "__main__":
    main() 