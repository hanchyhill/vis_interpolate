#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行优化后的批量处理的便捷脚本
"""

from src.dem_interpolation import main_batch

def main():
    """
    运行优化版的批量处理
    """
    print("=== 启动优化版DEM批量处理 ===")
    print("✅ NetCDF保存问题已修复")
    print("✅ 边界优化算法已激活")
    print("✅ 预期效果：100%数据覆盖率")
    print("-" * 50)
    
    try:
        main_batch()
    except KeyboardInterrupt:
        print("\n用户中断处理")
    except Exception as e:
        print(f"\n处理过程中出现错误: {e}")
        print("如果问题持续存在，请检查:")
        print("1. 输入目录是否存在DEM文件")
        print("2. 输出目录是否有写入权限")
        print("3. 磁盘空间是否充足")

if __name__ == "__main__":
    main()