import sys
import os

def main():
    """
    主程序入口
    """
    print("=== DEM数据稀疏化处理工具 ===")
    print("1. 运行单个DEM插值脚本")
    print("2. 批量处理DEM文件并合并")
    print("3. 查看帮助信息")
    
    choice = input("请选择操作 (1/2/3): ").strip()
    
    if choice == "1":
        # 运行单个DEM插值脚本
        try:
            from src.dem_interpolation import main as dem_main
            dem_main()
        except ImportError as e:
            print(f"导入模块失败: {e}")
            print("请确保已安装所需的依赖包")
        except Exception as e:
            print(f"运行过程中发生错误: {e}")
    
    elif choice == "2":
        # 批量处理DEM文件
        try:
            from src.dem_interpolation import main_batch
            main_batch()
        except ImportError as e:
            print(f"导入模块失败: {e}")
            print("请确保已安装所需的依赖包")
        except Exception as e:
            print(f"运行过程中发生错误: {e}")
    
    elif choice == "3":
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
    print("- 支持多种插值方法 (linear, nearest, cubic, fast_nearest, vectorized)")
    print("- 自动创建目标网格")
    print("- 单个文件处理和批量文件处理")
    print("- 可视化对比原始数据和插值后的数据")
    print("- NetCDF文件合并功能")
    print("- 保存结果为NetCDF格式")
    print("- 显示数据压缩比和统计信息")
    print("\n处理模式:")
    print("1. 单个文件模式: 处理单个DEM TIF文件")
    print("2. 批量处理模式: 遍历目录中所有DEM文件，转换并合并为单个NetCDF文件")
    print("\n批量处理功能:")
    print("- 自动识别ASTGTM2_NyyExxx_dem.tif格式的文件")
    print("- 解析文件名中的经纬度信息")
    print("- 将所有TIF文件转换为NetCDF格式")
    print("- 合并所有NetCDF文件为单个大文件")
    print("- 支持大文件分块处理，避免内存溢出")
    print("\n文件路径设置:")
    print("- 输入目录: h:\\data\\DEM\\")
    print("- 临时输出目录: h:\\data\\DEM\\netcdf_output\\")
    print("- 最终合并文件: h:\\data\\DEM\\merged_dem_data.nc")
    print("\n使用方法:")
    print("1. 确保DEM文件放在指定目录中")
    print("2. 选择对应的处理模式")
    print("3. 查看生成的结果文件")
    print("\n输出文件:")
    print("单个文件模式:")
    print("  - dem_comparison.png: 原始数据和插值后数据的对比图")
    print("  - interpolated_dem_0.01deg.nc: 插值后的DEM数据")
    print("批量处理模式:")
    print("  - netcdf_output/: 临时NetCDF文件目录")
    print("  - merged_dem_data.nc: 最终合并的大文件")

if __name__ == "__main__":
    main()
