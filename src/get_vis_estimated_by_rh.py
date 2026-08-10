"""
能见度估算脚本（基于相对湿度和距离的加权插值）
从指定日期范围内循环处理国家站和区域站数据，估算区域站能见度
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta
from pathlib import Path, PureWindowsPath
from os import path
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')


def build_filenames_and_urls(ts: datetime, province: str = "广东"):
    """
    根据指定时间 ts（datetime），返回国家站和区域站的文件路径

    参数
    ----
    ts : datetime
        目标时间
    province : str
        省份名称，默认 "广东"

    返回
    ----
    dict，结构如下：
    {
        "national": {"filename": str, "fullpath": str},
        "regional": {"filename": str, "fullpath": str},
    }
    """
    YYYY = ts.strftime("%Y")
    MM   = ts.strftime("%m")
    DD   = ts.strftime("%d")
    HH   = ts.strftime("%H")
    mm   = ts.strftime("%M")

    # 国家站
    national_filename = f"SurfAuto_{province}_{YYYY}{MM}{DD}{HH}{mm}00.csv"
    national_fullpath = str(
        PureWindowsPath(r"\\10.148.44.81\surf\idea\getSurfAutoOrg4Prov") / YYYY / MM / national_filename
    )

    # 区域站
    regional_filename = f"SurfAwst_{province}_{YYYY}{MM}{DD}{HH}{mm}00.csv"
    regional_fullpath = str(
        PureWindowsPath(r"\\10.148.44.81\surf\idea\getSurfAwstOrg4Prov") / YYYY / MM / regional_filename
    )

    return {
        "national":  {"filename": national_filename, "fullpath": national_fullpath},
        "regional":  {"filename": regional_filename, "fullpath": regional_fullpath},
    }


# 定义字段
fields = [
    'V01301',    # 站号
    'VF01015_CN',# 站点名称
    'V_CITY',    # 所属地市
    'V_COUNTY',  # 所属县
    'V06001',    # 经度
    'V05001',    # 纬度
    'V07001',    # 海拔
    'V20001',    # 能见度
    'V13003'     # 相对湿度
]

fields_2 = [
    'V01301',    # 站号
    'VF01015_CN',# 站点名称
    'V_CITY',    # 所属地市
    'V_COUNTY',  # 所属县
    'V06001',    # 经度
    'V05001',    # 纬度
    'V07001',    # 海拔
    'V13003',     # 相对湿度
    'V20001_701_01', # 能见度
]

# 字段映射
field_map = {
    'V01301': 'code',
    'VF01015_CN': 'name',
    'V_CITY': 'city',
    'V_COUNTY': 'county',
    'V06001': 'lon',
    'V05001': 'lat',
    'V07001': 'altitude',
    'V20001': 'vis',
    'V13003': 'rh',
    'V20001_701_01': 'vis',
}


def find_nearest_stations(target_coords, reference_coords, k=4):
    """
    找到最近的k个站点

    Parameters:
    target_coords: 目标站点坐标 (lat, lon)
    reference_coords: 参考站点坐标数组 [(lat1, lon1), (lat2, lon2), ...]
    k: 最近邻站点数量

    Returns:
    nearest_indices: 最近站点的索引
    nearest_distances: 对应的距离
    """
    # 计算距离矩阵（使用欧几里得距离，单位：度）
    distances = cdist([target_coords], reference_coords, metric='euclidean')[0]

    # 找到最近的k个站点
    nearest_indices = np.argsort(distances)[:k]
    nearest_distances = distances[nearest_indices]

    return nearest_indices, nearest_distances


def calculate_visibility_by_humidity(target_rh, nearest_stations_data):
    """
    基于相对湿度计算能见度

    Parameters:
    target_rh: 目标站点的相对湿度
    nearest_stations_data: 最近邻站点的数据 (包含rh和vis列)

    Returns:
    vis_rh: 基于湿度权重的能见度估算值
    """
    # 过滤掉湿度或能见度为NaN的站点
    valid_stations = nearest_stations_data.dropna(subset=['rh', 'vis'])

    if len(valid_stations) == 0:
        return np.nan

    # 计算湿度差
    rh_diff = np.abs(target_rh - valid_stations['rh'].values)

    # 避免除零，设置最小差值
    rh_diff = np.maximum(rh_diff, 0.1)

    # 计算湿度权重 w_rh_i = 1/(d_rh_i)^2
    w_rh = 1.0 / (rh_diff ** 2)

    # 加权平均
    vis_rh = np.sum(w_rh * valid_stations['vis'].values) / np.sum(w_rh)

    return vis_rh


def calculate_visibility_by_distance(nearest_distances, nearest_stations_vis):
    """
    基于距离计算能见度

    Parameters:
    nearest_distances: 到最近邻站点的距离数组
    nearest_stations_vis: 最近邻站点的能见度数组

    Returns:
    vis_dis: 基于距离权重的能见度估算值
    """
    # 过滤掉能见度为NaN的站点
    valid_mask = ~np.isnan(nearest_stations_vis)
    valid_distances = nearest_distances[valid_mask]
    valid_vis = nearest_stations_vis[valid_mask]

    if len(valid_vis) == 0:
        return np.nan

    # 避免除零，设置最小距离
    valid_distances = np.maximum(valid_distances, 0.001)

    # 计算距离权重 w_dis_i = 1/(distance_i)^2
    w_dis = 1.0 / (valid_distances ** 2)

    # 加权平均
    vis_dis = np.sum(w_dis * valid_vis) / np.sum(w_dis)

    return vis_dis


def estimate_visibility_for_time(time_selected: datetime, output_dir: Path, use_cache: bool = True, type:str = "national"):
    """
    为指定时刻估算区域站能见度

    参数:
        time_selected: datetime, 指定时间
        output_dir: Path, 输出目录
        use_cache: bool, 是否使用缓存（默认True）
        type: str, 站点类型（默认"national"）

    返回:
        station_vis_all: pd.DataFrame, 合并后的数据（国家站实测+区域站估算）
    """
    time_str = time_selected.strftime('%Y%m%d%H%M')
    if type == "national":
        cache_file = output_dir / f"station_vis_all_estimated_{time_str}.csv"
    elif type == "national_and_regional":
        cache_file = output_dir / f"station_vis_all_estimated_{time_str}_national_and_regional.csv"
    else:
        print(f"⚠ 未知的站点类型: {type}")
        raise ValueError(f"未知的估算站点类型: {type}")

    # 检查缓存
    if use_cache and cache_file.exists():
        print(f"[缓存]", end=" ")
        try:
            station_vis_all = pd.read_csv(cache_file, encoding='utf-8-sig')
            return station_vis_all
        except Exception as e:
            print(f"⚠ 读取缓存失败: {e}")

    # 获取文件路径
    files_selected = build_filenames_and_urls(time_selected)
    file_path_nation = files_selected["national"]["fullpath"]
    file_path_region = files_selected["regional"]["fullpath"]

    try:
        # 读取国家站数据
        df_nation = pd.read_csv(file_path_nation, encoding='gbk', na_values=9999, usecols=fields)
        df_nation = df_nation.rename(columns=field_map)

        # 读取区域站数据
        df_region = pd.read_csv(file_path_region, encoding='gbk', na_values=9999, usecols=fields_2)
        df_region = df_region.rename(columns=field_map)
        df_region_vis_rh = df_region.dropna(subset=['vis','rh', 'county'])
        df_region_vis = df_region.dropna(subset=['vis', 'county'])
        common_columns = list(set(df_nation.columns) & set(df_region_vis_rh.columns))
        print("共同列:", common_columns)

        # 只保留共同列进行合并
        df_combined = pd.concat([
            df_nation[common_columns], 
            df_region_vis[common_columns]
        ], ignore_index=True)
        df_combined = df_combined.dropna(subset=['rh'])
        # 去除掉df_region中rh或county为NaN的条目
        df_region_rh = df_region.dropna(subset=['rh', 'county'])

    except Exception as e:
        print(f"⚠ 读取站点数据失败: {e}")
        return None
    if type == "national":
        station_vis = df_nation
        station_rh = df_region_rh
    elif type == "national_and_regional":
        station_vis = df_combined
        station_rh = df_region_rh
    else:
        print(f"⚠ 未知的站点类型: {type}")
        raise ValueError(f"未知的估算站点类型: {type}")

    # 检查数据是否有效
    if len(station_vis) == 0 or len(station_rh) == 0:
        print(f"⚠ 数据为空（国家站: {len(station_vis)}, 区域站: {len(station_rh)}）")
        return None

    # 为station_rh中的站点估算能见度
    station_rh_estimated = station_rh.copy()

    # 准备坐标数据
    vis_coords = station_vis[['lat', 'lon']].values
    rh_coords = station_rh[['lat', 'lon']].values

    # 初始化结果列表
    vis_rh_results = []
    vis_dis_results = []
    vis_final_results = []

    # 对每个station_rh站点进行处理
    for idx, row in station_rh.iterrows():
        target_coords = [row['lat'], row['lon']]
        target_rh = row['rh']

        # 找到最近的4个有能见度数据的站点
        nearest_indices, nearest_distances = find_nearest_stations(
            target_coords, vis_coords, k=4
        )

        # 获取最近邻站点的数据
        nearest_stations = station_vis.iloc[nearest_indices]

        # 基于相对湿度的能见度估算
        vis_rh = calculate_visibility_by_humidity(target_rh, nearest_stations)
        vis_rh_results.append(vis_rh)

        # 基于距离的能见度估算
        vis_dis = calculate_visibility_by_distance(
            nearest_distances, nearest_stations['vis'].values
        )
        vis_dis_results.append(vis_dis)

        # 最终能见度估算（两种方法的平均）
        if np.isnan(vis_rh) and np.isnan(vis_dis):
            vis_final = np.nan
        elif np.isnan(vis_rh):
            vis_final = vis_dis
        elif np.isnan(vis_dis):
            vis_final = vis_rh
        else:
            vis_final = 0.5 * vis_rh + 0.5 * vis_dis

        vis_final_results.append(vis_final)

    # 将结果添加到数据框中
    station_rh_estimated['vis_rh'] = vis_rh_results
    station_rh_estimated['vis_dis'] = vis_dis_results
    station_rh_estimated['vis'] = vis_final_results
    station_rh_estimated['is_vis_est'] = 1

    # 为station_vis添加标识列
    station_vis_final = station_vis.copy()
    station_vis_final['is_vis_est'] = 0  # 实测值

    # 确保station_rh_estimated有相同的列结构
    if 'vis_rh' not in station_vis_final.columns:
        station_vis_final['vis_rh'] = np.nan
    if 'vis_dis' not in station_vis_final.columns:
        station_vis_final['vis_dis'] = np.nan

    # 合并两个数据集
    station_vis_all = pd.concat([station_vis_final, station_rh_estimated],
                               ignore_index=True, sort=False)

    # 重新排列列的顺序
    column_order = ['code', 'name', 'city', 'county', 'lon', 'lat', 'altitude',
                    'rh', 'vis', 'vis_rh', 'vis_dis', 'is_vis_est']
    station_vis_all = station_vis_all[column_order]
    station_vis_all_nodup = station_vis_all.drop_duplicates(subset=['code'], keep='first')
    station_vis_all = station_vis_all_nodup

    # 保存结果到CSV文件
    station_vis_all.to_csv(cache_file, index=False, encoding='utf-8-sig')

    return station_vis_all


def process_single_time(args):
    """
    处理单个时次的能见度估算（用于多进程）

    参数:
        args: tuple, (time_selected, output_dir, type, index, total)

    返回:
        dict: 包含处理结果的字典
    """
    time_selected, output_dir, type, index, total = args
    time_str = time_selected.strftime('%Y-%m-%d %H:%M')

    try:
        # 执行能见度估算
        station_vis_all = estimate_visibility_for_time(
            time_selected, output_dir, use_cache=True, type=type
        )

        if station_vis_all is not None:
            n_national = len(station_vis_all[station_vis_all['is_vis_est'] == 0])
            n_regional = len(station_vis_all[station_vis_all['is_vis_est'] == 1])
            n_estimated = np.sum(~np.isnan(station_vis_all[station_vis_all['is_vis_est'] == 1]['vis']))

            print(f"✓ [{index}/{total}] {time_str} | 国家站: {n_national} | 区域站: {n_regional} | 成功估算: {n_estimated}")
            return {'success': True, 'time': time_str}
        else:
            print(f"✗ [{index}/{total}] {time_str} | 数据不可用")
            return {'success': False, 'time': time_str}
    except Exception as e:
        print(f"✗ [{index}/{total}] {time_str} | 错误: {str(e)}")
        return {'success': False, 'time': time_str, 'error': str(e)}


def main():
    """
    主函数：并发处理指定日期范围内的能见度估算
    """
    # 定义日期范围
    start_date = datetime(2024, 10, 11, 0, 0)
    end_date = datetime(2025, 10, 11, 0, 0)

    # 计算CPU核心数
    num_processes = max(1, cpu_count() - 1)

    for type in ["national_and_regional", "national"]:
        # 定义输出目录
        if type == "national":
            output_dir = Path(path.dirname(__file__)) / '../data/vis_estimated_base_nation_station'
        elif type == "national_and_regional":
            output_dir = Path(path.dirname(__file__)) / '../data/vis_estimated_base_nation_and_regional_station'

        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成所有需要处理的时间列表
        time_list = []
        current_date = start_date
        while current_date <= end_date:
            time_list.append(current_date)
            current_date += timedelta(hours=1)

        total_hours = len(time_list)

        print("=" * 80)
        print(f"能见度估算（基于国家站插值区域站）")
        print(f"站点类型: {type}")
        print(f"起始时间: {start_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"结束时间: {end_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"总时次数: {total_hours}")
        print(f"并发进程数: {num_processes}")
        print(f"输出目录: {output_dir}")
        print("=" * 80)

        # 准备多进程参数
        args_list = [
            (time, output_dir, type, idx + 1, total_hours)
            for idx, time in enumerate(time_list)
        ]

        # 使用多进程并发处理
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_single_time, args_list)

        # 统计结果
        processed = sum(1 for r in results if r['success'])
        skipped = len(results) - processed

        # 输出汇总信息
        print("\n" + "=" * 80)
        print("处理完成!")
        print(f"站点类型: {type}")
        print(f"成功处理: {processed} 个时次")
        print(f"跳过时次: {skipped} 个")
        print(f"输出目录: {output_dir}")
        print("=" * 80)



if __name__ == "__main__":
    main()
