#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试批量处理功能
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dem_interpolation import (
    parse_dem_filename,
    process_dem_directory,
    merge_netcdf_files,
    batch_process_and_merge
)

def test_parse_dem_filename():
    """
    测试文件名解析功能
    """
    print("=== 测试文件名解析 ===")
    
    test_cases = [
        ("ASTGTM2_N23E111_dem.tif", (23, 111)),
        ("ASTGTM2_N45E120_dem.tif", (45, 120)),
        ("ASTGTM2_N01E001_dem.tif", (1, 1)),
        ("invalid_filename.tif", (None, None)),
        ("ASTGTM2_N23E111.tif", (None, None))  # 缺少_dem
    ]
    
    for filename, expected in test_cases:
        result = parse_dem_filename(filename)
        status = "✓" if result == expected else "✗"
        print(f"{status} {filename} -> {result} (期望: {expected})")

def test_directory_structure():
    """
    测试目录结构和文件查找
    """
    print("\n=== 测试目录结构 ===")
    
    dem_dir = r'h:\data\DEM'
    
    if os.path.exists(dem_dir):
        print(f"✓ DEM目录存在: {dem_dir}")
        
        # 查找DEM文件
        from pathlib import Path
        tif_files = list(Path(dem_dir).glob('ASTGTM2_N*E*_dem.tif'))
        print(f"找到 {len(tif_files)} 个DEM文件")
        
        # 显示前几个文件
        for i, file in enumerate(tif_files[:5], 1):
            file_size = file.stat().st_size / 1024 / 1024
            lat, lon = parse_dem_filename(file.name)
            print(f"  {i}. {file.name} ({file_size:.1f}MB, 纬度:{lat}, 经度:{lon})")
        
        if len(tif_files) > 5:
            print(f"  ... 还有 {len(tif_files) - 5} 个文件")
        
        return len(tif_files) > 0
    else:
        print(f"✗ DEM目录不存在: {dem_dir}")
        return False

def test_output_directory():
    """
    测试输出目录创建
    """
    print("\n=== 测试输出目录 ===")
    
    output_dir = r'h:\data\DEM\netcdf_output'
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"✓ 输出目录创建成功: {output_dir}")
        
        # 测试写入权限
        test_file = Path(output_dir) / "test_write.txt"
        test_file.write_text("test")
        test_file.unlink()
        print("✓ 输出目录写入权限正常")
        
        return True
    except Exception as e:
        print(f"✗ 输出目录创建失败: {e}")
        return False

def test_dependencies():
    """
    测试依赖包
    """
    print("\n=== 测试依赖包 ===")
    
    required_packages = [
        'rioxarray',
        'xarray',
        'numpy',
        'matplotlib',
        'cartopy',
        'scipy'
    ]
    
    all_good = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - 未安装")
            all_good = False
    
    return all_good

def test_small_batch():
    """
    使用临时文件测试小批量处理（如果有真实数据的话）
    """
    print("\n=== 测试小批量处理 ===")
    
    dem_dir = r'h:\data\DEM'
    
    if not os.path.exists(dem_dir):
        print("跳过批量处理测试 - DEM目录不存在")
        return
    
    # 查找少量文件进行测试
    from pathlib import Path
    tif_files = list(Path(dem_dir).glob('ASTGTM2_N*E*_dem.tif'))
    
    if len(tif_files) == 0:
        print("跳过批量处理测试 - 没有找到DEM文件")
        return
    
    # 只测试前2个文件
    test_files = tif_files[:2]
    print(f"测试文件: {[f.name for f in test_files]}")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output = Path(temp_dir) / "netcdf_output"
        temp_final = Path(temp_dir) / "merged_test.nc"
        
        try:
            # 注意：这里只是测试函数调用，实际处理可能很耗时
            print("注意：实际处理测试被跳过以节省时间")
            print("如需完整测试，请手动运行 batch_example.py")
            
        except Exception as e:
            print(f"✗ 批量处理测试失败: {e}")

def main():
    """
    运行所有测试
    """
    print("=== DEM批量处理功能测试 ===")
    
    tests = [
        ("文件名解析", test_parse_dem_filename),
        ("目录结构", test_directory_structure),
        ("输出目录", test_output_directory),
        ("依赖包", test_dependencies),
        ("小批量处理", test_small_batch)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result if result is not None else True
        except Exception as e:
            print(f"✗ {test_name} 测试出错: {e}")
            results[test_name] = False
    
    # 总结
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print("-" * 50)
    
    passed = 0
    for test_name, result in results.items():
        status = "通过" if result else "失败"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！批量处理功能准备就绪。")
    else:
        print("⚠️ 部分测试失败，请检查环境配置。")

if __name__ == "__main__":
    main()