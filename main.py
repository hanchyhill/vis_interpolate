import sys
import os

def main():
    """
    主程序入口
    """
    print("=== DEM数据稀疏化处理工具 ===")
    print("1. 运行DEM插值脚本")
    print("2. 查看帮助信息")
    
    choice = input("请选择操作 (1/2): ").strip()
    
    if choice == "1":
        # 运行DEM插值脚本
        try:
            from src.dem_interpolation import main as dem_main
            dem_main()
        except ImportError as e:
            print(f"导入模块失败: {e}")
            print("请确保已安装所需的依赖包")
        except Exception as e:
            print(f"运行过程中发生错误: {e}")
    
    elif choice == "2":
        print_help()
    
    else:
        print("无效选择，请重新运行程序")

def print_help():
    """
    打印帮助信息
    """
    print("\n=== 帮助信息 ===")
    print("本工具用于将高分辨率的DEM数据插值到0.01°×0.01°的网格中，实现数据稀疏化。")
    print("\n功能特点:")
    print("- 支持多种插值方法 (linear, nearest, cubic)")
    print("- 自动创建目标网格")
    print("- 可视化对比原始数据和插值后的数据")
    print("- 保存结果为NetCDF格式")
    print("- 显示数据压缩比和统计信息")
    print("\n使用方法:")
    print("1. 确保DEM文件路径正确")
    print("2. 运行主程序或直接运行 src/dem_interpolation.py")
    print("3. 查看生成的对比图和结果文件")
    print("\n输出文件:")
    print("- dem_comparison.png: 原始数据和插值后数据的对比图")
    print("- interpolated_dem_0.01deg.nc: 插值后的DEM数据")

if __name__ == "__main__":
    main()
