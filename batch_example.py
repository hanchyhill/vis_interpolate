#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理DEM文件示例脚本
"""

from src.dem_interpolation import batch_process_and_merge
import os

def main():
    """
    批量处理DEM文件的示例
    """
    print("=== DEM批量处理示例 ===")
    
    # 设置路径
    input_dir = r'h:\data\DEM'
    output_dir = r'h:\data\DEM\netcdf_output'
    final_output_file = r'h:\data\DEM\merged_dem_data.nc'
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"警告: 输入目录不存在 {input_dir}")
        print("请确保DEM文件存放在正确的目录中")
        return
    
    # 处理参数设置
    resolution = 0.01  # 目标分辨率 0.01度 (约1.1公里)
    method = 'vectorized'  # 使用最快的插值方法
    
    # 分块设置（用于处理大文件，避免内存不足）
    chunk_size = {'lat': 1000, 'lon': 1000}
    
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"最终文件: {final_output_file}")
    print(f"目标分辨率: {resolution}度")
    print(f"插值方法: {method}")
    print("-" * 50)
    
    # 执行批量处理
    try:
        result = batch_process_and_merge(
            input_dir=input_dir,
            output_dir=output_dir,
            final_output_file=final_output_file,
            resolution=resolution,
            method=method,
            chunk_size=chunk_size
        )
        
        if result:
            print(f"\n🎉 批量处理成功完成！")
            print(f"最终合并文件: {result}")
            
            # 显示文件信息
            if os.path.exists(result):
                file_size = os.path.getsize(result) / 1024 / 1024
                print(f"最终文件大小: {file_size:.2f} MB")
        else:
            print("\n❌ 批量处理失败")
            
    except Exception as e:
        print(f"\n❌ 处理过程中发生错误: {e}")
        print("请检查:")
        print("1. 输入目录是否包含正确格式的DEM文件")
        print("2. 文件名格式是否为 ASTGTM2_NyyExxx_dem.tif")
        print("3. 是否有足够的磁盘空间")
        print("4. 是否有文件写入权限")

def check_dem_files(directory):
    """
    检查DEM文件情况
    """
    import glob
    
    print(f"\n=== 检查DEM文件 ===")
    print(f"目录: {directory}")
    
    if not os.path.exists(directory):
        print("目录不存在")
        return
    
    # 查找DEM文件
    pattern = os.path.join(directory, "ASTGTM2_N*E*_dem.tif")
    dem_files = glob.glob(pattern)
    
    print(f"找到 {len(dem_files)} 个DEM文件:")
    
    for i, file in enumerate(dem_files[:10], 1):  # 只显示前10个
        filename = os.path.basename(file)
        file_size = os.path.getsize(file) / 1024 / 1024
        print(f"  {i}. {filename} ({file_size:.2f} MB)")
    
    if len(dem_files) > 10:
        print(f"  ... 还有 {len(dem_files) - 10} 个文件")

if __name__ == "__main__":
    # 可以先检查文件情况
    check_dem_files(r'h:\data\DEM')
    
    # 询问是否继续处理
    response = input("\n是否开始批量处理? (y/n): ").strip().lower()
    
    if response == 'y' or response == 'yes':
        main()
    else:
        print("已取消处理")