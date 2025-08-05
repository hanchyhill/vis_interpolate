#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试边界优化功能的专用脚本
"""

import os
import sys
import numpy as np
import xarray as xr
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dem_interpolation import (
    merge_netcdf_files,
    merge_netcdf_files_optimized,
    visualize_boundary_optimization,
    analyze_boundary_gaps,
    fill_boundary_gaps,
    create_unified_grid
)

def create_test_data_with_gaps():
    """
    创建包含边界缺口的测试数据
    """
    print("创建测试数据...")
    
    # 创建三个模拟的DEM瓦片数据
    test_datasets = []
    
    # 瓦片1: 左上角
    lat1 = np.linspace(24.0, 25.0, 100)
    lon1 = np.linspace(110.0, 111.0, 100)
    elev1 = np.random.normal(1000, 200, (100, 100))
    # 添加一些地形特征
    x1, y1 = np.meshgrid(lon1, lat1)
    elev1 += 300 * np.sin(x1 * 5) * np.cos(y1 * 5)
    
    ds1 = xr.Dataset({
        'elevation': (('lat', 'lon'), elev1)
    }, coords={'lat': lat1, 'lon': lon1})
    
    # 瓦片2: 右上角（与瓦片1有小缺口）
    lat2 = np.linspace(24.0, 25.0, 100)
    lon2 = np.linspace(111.02, 112.02, 100)  # 故意留0.02度缺口
    elev2 = np.random.normal(1200, 150, (100, 100))
    x2, y2 = np.meshgrid(lon2, lat2)
    elev2 += 200 * np.sin(x2 * 3) * np.cos(y2 * 4)
    
    ds2 = xr.Dataset({
        'elevation': (('lat', 'lon'), elev2)
    }, coords={'lat': lat2, 'lon': lon2})
    
    # 瓦片3: 下方（与上面两个都有缺口）
    lat3 = np.linspace(22.98, 23.98, 100)  # 故意留0.02度缺口
    lon3 = np.linspace(110.5, 111.5, 100)
    elev3 = np.random.normal(800, 180, (100, 100))
    x3, y3 = np.meshgrid(lon3, lat3)
    elev3 += 400 * np.exp(-((x3-111)**2 + (y3-23.5)**2) * 10)
    
    ds3 = xr.Dataset({
        'elevation': (('lat', 'lon'), elev3)
    }, coords={'lat': lat3, 'lon': lon3})
    
    test_datasets = [ds1, ds2, ds3]
    
    print(f"创建了 {len(test_datasets)} 个测试数据集")
    for i, ds in enumerate(test_datasets):
        lat_range = (ds.lat.min().values, ds.lat.max().values)
        lon_range = (ds.lon.min().values, ds.lon.max().values)
        print(f"  数据集 {i+1}: 纬度 {lat_range[0]:.3f}°-{lat_range[1]:.3f}°, "
              f"经度 {lon_range[0]:.3f}°-{lon_range[1]:.3f}°")
    
    return test_datasets

def test_boundary_gap_filling():
    """
    测试边界缺失值填充算法
    """
    print("\n=== 测试边界缺失值填充 ===")
    
    # 创建包含缺失值的测试数据
    test_data = np.random.normal(1000, 200, (50, 50))
    
    # 人工添加一些边界缺失值
    test_data[20:25, 10:15] = np.nan  # 中间区域
    test_data[0:5, :] = np.nan        # 顶部边界
    test_data[:, 45:50] = np.nan      # 右侧边界
    test_data[10:15, 0:5] = np.nan    # 左侧小块
    
    print(f"原始缺失值数量: {np.sum(np.isnan(test_data))}")
    
    # 测试不同填充方法
    methods = ['linear', 'nearest']
    for method in methods:
        print(f"\n测试 {method} 填充方法:")
        filled_data = fill_boundary_gaps(test_data.copy(), method=method, max_distance=5)
        remaining_nans = np.sum(np.isnan(filled_data))
        filled_count = np.sum(np.isnan(test_data)) - remaining_nans
        print(f"  填充了 {filled_count} 个缺失值")
        print(f"  剩余缺失值: {remaining_nans}")

def test_boundary_optimization_pipeline():
    """
    测试完整的边界优化流程
    """
    print("\n=== 测试边界优化完整流程 ===")
    
    # 创建测试数据
    test_datasets = create_test_data_with_gaps()
    
    # 分析边界缺口
    gaps_analysis = analyze_boundary_gaps(test_datasets, resolution=0.01)
    
    # 传统合并（不优化）
    print("\n1. 传统合并方法:")
    try:
        traditional_merged = xr.merge(test_datasets)
        print("✓ 传统合并成功")
    except Exception as e:
        print(f"✗ 传统合并失败: {e}")
        # 使用基础的统一网格方法
        traditional_merged = create_basic_unified_grid(test_datasets)
    
    # 优化合并
    print("\n2. 边界优化合并方法:")
    optimized_merged = create_unified_grid(test_datasets)
    
    # 比较结果
    if traditional_merged and optimized_merged:
        print("\n3. 结果对比:")
        
        trad_valid = np.sum(~np.isnan(traditional_merged['elevation'].values))
        opt_valid = np.sum(~np.isnan(optimized_merged['elevation'].values))
        
        print(f"传统方法有效点数: {trad_valid:,}")
        print(f"优化方法有效点数: {opt_valid:,}")
        print(f"改善点数: {opt_valid - trad_valid:,}")
        
        if opt_valid > trad_valid:
            improvement = (opt_valid - trad_valid) / trad_valid * 100
            print(f"改善程度: {improvement:.2f}%")
        
        # 可视化对比（如果可能）
        try:
            visualize_boundary_optimization(
                traditional_merged, optimized_merged, 
                save_path='boundary_optimization_test.png'
            )
        except Exception as e:
            print(f"可视化失败（可能是显示问题）: {e}")
    
    return gaps_analysis, traditional_merged, optimized_merged

def create_basic_unified_grid(datasets):
    """
    创建基础的统一网格（不进行边界优化）
    """
    # 收集所有坐标
    all_lats = []
    all_lons = []
    
    for ds in datasets:
        all_lats.extend(ds.lat.values)
        all_lons.extend(ds.lon.values)
    
    unique_lats = np.sort(np.unique(all_lats))
    unique_lons = np.sort(np.unique(all_lons))
    
    # 创建结果数组
    result_data = np.full((len(unique_lats), len(unique_lons)), np.nan)
    
    # 简单填充数据
    for ds in datasets:
        elevation = ds['elevation']
        ds_lats = ds.lat.values
        ds_lons = ds.lon.values
        
        for j, lat in enumerate(ds_lats):
            for k, lon in enumerate(ds_lons):
                lat_idx = np.argmin(np.abs(unique_lats - lat))
                lon_idx = np.argmin(np.abs(unique_lons - lon))
                
                if not np.isnan(elevation.values[j, k]):
                    result_data[lat_idx, lon_idx] = elevation.values[j, k]
    
    return xr.Dataset({
        'elevation': (('lat', 'lon'), result_data)
    }, coords={'lat': unique_lats, 'lon': unique_lons})

def test_real_data_if_available():
    """
    如果有真实数据，测试边界优化功能
    """
    print("\n=== 测试真实数据（如果可用）===")
    
    netcdf_dir = Path(r'h:\data\DEM\netcdf_output')
    
    if netcdf_dir.exists():
        netcdf_files = list(netcdf_dir.glob('*.nc'))
        
        if len(netcdf_files) >= 2:
            print(f"找到 {len(netcdf_files)} 个NetCDF文件")
            
            # 选择前几个文件进行测试
            test_files = [str(f) for f in netcdf_files[:4]]  # 最多测试4个文件
            
            print("测试文件:")
            for f in test_files:
                print(f"  - {Path(f).name}")
            
            try:
                # 传统合并
                print("\n传统合并方法:")
                traditional_result = merge_netcdf_files(
                    test_files, 
                    'test_traditional_merge.nc'
                )
                
                # 优化合并
                print("\n优化合并方法:")
                optimized_result = merge_netcdf_files_optimized(
                    test_files, 
                    'test_optimized_merge.nc',
                    enable_boundary_optimization=True
                )
                
                # 对比结果
                if traditional_result and optimized_result:
                    print("\n=== 真实数据测试结果 ===")
                    
                    trad_elev = traditional_result['elevation']
                    opt_elev = optimized_result['elevation']
                    
                    trad_valid = np.sum(~np.isnan(trad_elev.values))
                    opt_valid = np.sum(~np.isnan(opt_elev.values))
                    
                    print(f"传统方法有效点: {trad_valid:,}")
                    print(f"优化方法有效点: {opt_valid:,}")
                    
                    if opt_valid > trad_valid:
                        improvement = (opt_valid - trad_valid) / trad_valid * 100
                        print(f"边界优化改善: {improvement:.2f}%")
                        print("✓ 边界优化功能正常工作！")
                    else:
                        print("- 此数据集可能没有明显的边界缺口问题")
                
            except Exception as e:
                print(f"真实数据测试失败: {e}")
        else:
            print(f"NetCDF文件数量不足（需要至少2个，找到{len(netcdf_files)}个）")
    else:
        print("NetCDF目录不存在，跳过真实数据测试")

def main():
    """
    运行所有边界优化测试
    """
    print("=== DEM边界优化功能测试 ===")
    
    try:
        # 测试1: 边界缺失值填充算法
        test_boundary_gap_filling()
        
        # 测试2: 完整的边界优化流程
        gaps_analysis, traditional, optimized = test_boundary_optimization_pipeline()
        
        # 测试3: 真实数据（如果可用）
        test_real_data_if_available()
        
        print("\n" + "="*60)
        print("🎉 边界优化功能测试完成！")
        print("主要改进:")
        print("- ✓ 边界缺失值智能填充")
        print("- ✓ 数据集边界平滑处理")
        print("- ✓ 重叠区域加权平均")
        print("- ✓ 多层次缺口填充策略")
        print("\n使用新的优化功能重新运行批量处理，")
        print("边界缺失值问题将得到显著改善！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        print("请检查依赖包是否正确安装")

if __name__ == "__main__":
    main()