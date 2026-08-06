"""
能见度预报检验脚本
从指定日期范围内循环检验能见度预报效果，并保存结果
"""

import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path, PureWindowsPath
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


def build_filenames_and_urls(ts: datetime, province: str = "广东") -> Dict[str, Dict[str, str]]:
    """
    根据指定时间 ts（datetime），返回：
      1) 国家站：文件名与完整 Windows 共享路径
      2) 区域站：文件名与完整 Windows 共享路径
      3) 国家局5km能见度实况融合产品：文件名与完整 URL

    参数
    ----
    ts : datetime
        目标时间。国家站/区域站精确到分钟；能见度产品按小时。
    province : str
        省份名称（文件名中的中文省份名），默认 "广东"。

    返回
    ----
    dict，结构如下：
    {
        "national": {"filename": str, "fullpath": str},
        "regional": {"filename": str, "fullpath": str},
        "visibility": {"filename": str, "url": str},
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

    # 国家局5km能见度实况融合产品（按小时）
    vis_dir_yyyymmdd = f"{YYYY}{MM}{DD}"
    vis_filename = f"VIS_{YYYY}{MM}{DD}{HH}.NC"
    vis_url = f"http://10.148.8.71:7080/thredds/dodsC/cldas/{vis_dir_yyyymmdd}/{vis_filename}"

    return {
        "national":  {"filename": national_filename, "fullpath": national_fullpath},
        "regional":  {"filename": regional_filename, "fullpath": regional_fullpath},
        "cldas_vis": {"filename": vis_filename, "url": vis_url},
    }


def load_visibility_data(data_path: str):
    """
    加载能见度数据

    参数:
        data_path: 数据路径（文件路径或URL）

    返回:
        vis_data: xarray.DataArray, 能见度数据
    """
    try:
        # 立即加载到内存后关闭文件，避免定时绘图长期占用本地文件或
        # OPeNDAP 连接。国家局产品的 vis000 维度为 time/level/lat/lon。
        with xr.open_dataset(data_path) as ds:
            if 'visibility' in ds:
                vis_data = ds['visibility'].load()
            elif 'vis000' in ds:
                vis_data = ds['vis000'][0, 0, :, :].load()
            else:
                # 尝试获取第一个数据变量
                var_name = list(ds.data_vars)[0]
                vis_data = ds[var_name].load()

        # 转换单位如果需要（从m转换为km）
        if float(vis_data.max(skipna=True).values) > 100:  # 如果最大值大于100，可能是米单位
            vis_data = vis_data / 1000

        return vis_data

    except Exception as e:
        print(f"  ⚠ 加载数据时出错: {e}")
        return None


def evaluate_visibility_score(vis_data, df_station):
    """
    评估能见度预报效果

    参数:
        vis_data: xarray.DataArray, 预报能见度格点数据 (单位: km)
        df_station: pd.DataFrame, 观测站数据，需包含 lon, lat, vis 列 (vis单位: m)

    返回:
        results_df: pd.DataFrame, 包含每个站点的对比结果
        stats: dict, 整体统计指标
    """
    # 过滤掉缺失能见度观测值的站点
    df_valid = df_station.dropna(subset=['vis', 'lon', 'lat']).copy()

    # 确保观测值单位为km
    df_valid['vis_obs_km'] = df_valid['vis'] / 1000.0

    # 初始化结果列表
    forecast_values = []
    obs_values = []
    station_info = []

    # 遍历每个站点
    for idx, row in df_valid.iterrows():
        lon_station = row['lon']
        lat_station = row['lat']
        vis_obs = row['vis_obs_km']

        # 使用最近邻方法查找格点值
        vis_forecast = vis_data.sel(lon=lon_station, lat=lat_station, method='nearest').values

        # 处理负值或异常值（将负值设为0）
        if vis_forecast < 0:
            vis_forecast = 0.0

        forecast_values.append(vis_forecast)
        obs_values.append(vis_obs)
        station_info.append({
            'code': row.get('code', ''),
            'name': row.get('name', ''),
            'lon': lon_station,
            'lat': lat_station,
            'vis_obs': vis_obs,
            'vis_forecast': vis_forecast
        })

    # 转换为numpy数组进行计算
    forecast_array = np.array(forecast_values)
    obs_array = np.array(obs_values)

    # 计算统计指标
    # 1. 误差 (Forecast - Observation)
    error = forecast_array - obs_array

    # 2. 绝对误差 (Absolute Error)
    abs_error = np.abs(error)

    # 3. 相对误差 (Relative Error, %)
    relative_error = np.where(obs_array > 0.001,
                              (error / obs_array) * 100,
                              np.nan)

    # 4. 均方误差 (Mean Squared Error)
    mse = np.mean(error ** 2)

    # 5. 均方根误差 (Root Mean Squared Error)
    rmse = np.sqrt(mse)

    # 6. 平均绝对误差 (Mean Absolute Error)
    mae = np.mean(abs_error)

    # 7. 平均偏差 (Mean Bias)
    bias = np.mean(error)

    # 8. 标准差
    std_error = np.std(error)

    # 9. 相关系数
    correlation = np.corrcoef(forecast_array, obs_array)[0, 1]

    # 10. 相对平均绝对误差 (Mean Absolute Percentage Error, MAPE)
    mape = np.nanmean(np.abs(relative_error))

    # 构建结果DataFrame
    results_df = pd.DataFrame(station_info)
    results_df['error'] = error
    results_df['abs_error'] = abs_error
    results_df['relative_error'] = relative_error

    # 统计指标字典
    stats = {
        'n_stations': len(obs_array),
        'mean_obs': np.mean(obs_array),
        'mean_forecast': np.mean(forecast_array),
        'bias': bias,
        'mae': mae,
        'rmse': rmse,
        'std_error': std_error,
        'correlation': correlation,
        'mape': mape,
        'mse': mse
    }

    return results_df, stats


def load_cached_results(cache_file: Path):
    """
    从缓存文件加载已有的检验结果

    参数:
        cache_file: Path, 缓存文件路径

    返回:
        results_df: pd.DataFrame, 详细结果
        stats: dict, 统计指标
    """
    try:
        results_df = pd.read_csv(cache_file, encoding='utf-8-sig')

        # 从详细结果计算统计指标
        obs_array = results_df['vis_obs'].values
        forecast_array = results_df['vis_forecast'].values
        error = forecast_array - obs_array
        abs_error = np.abs(error)
        relative_error = np.where(obs_array > 0.001,
                                  (error / obs_array) * 100,
                                  np.nan)

        stats = {
            'n_stations': len(obs_array),
            'mean_obs': np.mean(obs_array),
            'mean_forecast': np.mean(forecast_array),
            'bias': np.mean(error),
            'mae': np.mean(abs_error),
            'rmse': np.sqrt(np.mean(error ** 2)),
            'std_error': np.std(error),
            'correlation': np.corrcoef(forecast_array, obs_array)[0, 1],
            'mape': np.nanmean(np.abs(relative_error)),
            'mse': np.mean(error ** 2)
        }

        return results_df, stats
    except Exception as e:
        print(f"  ⚠ 读取缓存文件失败: {e}")
        return None, None


def get_vis_score_by_time(time_selected: datetime, fields: list, fields_2: list, field_map: dict,
                          station_type: str = "national", output_dir: Path = None, use_cache: bool = True):
    """
    根据指定时间，获取能见度预报得分

    参数:
        time_selected: datetime, 指定时间
        fields: list, 国家站字段列表
        fields_2: list, 区域站字段列表
        field_map: dict, 字段映射字典
        station_type: str, 站点类型 ("national" 或 "regional")
        output_dir: Path, 输出目录（用于检查缓存文件）
        use_cache: bool, 是否使用缓存（默认True）

    返回:
        results_df: pd.DataFrame, 包含每个站点的对比结果
        stats: dict, 整体统计指标
    """
    # 检查是否存在缓存文件
    if use_cache and output_dir is not None:
        time_str = time_selected.strftime('%Y%m%d%H')
        cache_file = output_dir / f"vis_score_detail_{station_type}_{time_str}.csv"

        if cache_file.exists():
            print(f"[缓存]", end=" ")
            return load_cached_results(cache_file)

    files_selected = build_filenames_and_urls(time_selected)
    file_path_nation = files_selected["national"]["fullpath"]
    file_path_region = files_selected["regional"]["fullpath"]
    url_path_cldas_vis = files_selected["cldas_vis"]["url"]

    try:
        if station_type == "national":
            # 读取国家站数据
            df_station = pd.read_csv(file_path_nation, encoding='gbk', na_values=9999, usecols=fields)
            df_station = df_station.rename(columns=field_map)
        elif station_type == "regional":
            # 读取区域站数据
            df_station = pd.read_csv(file_path_region, encoding='gbk', na_values=9999, usecols=fields_2)
            df_station = df_station.rename(columns=field_map)
            # 去除掉df_station中vis或county为NaN的条目
            df_station = df_station.dropna(subset=['vis', 'county'])
        else:
            print(f"  ⚠ 未知的站点类型: {station_type}")
            return None, None
    except Exception as e:
        print(f"  ⚠ 读取{station_type}站数据失败: {e}")
        return None, None

    # 读取能见度预报数据
    vis_data = load_visibility_data(url_path_cldas_vis)
    if vis_data is None:
        return None, None

    # 评估能见度预报效果
    results_df, stats = evaluate_visibility_score(vis_data, df_station)

    return results_df, stats


def main():
    """
    主函数：循环检验指定日期范围内的能见度预报
    """
    # 定义日期范围
    start_date = datetime(2024, 10, 11, 0, 0)
    end_date = datetime(2025, 10, 11, 0, 0)

    # 定义输出目录
    output_dir = Path(r"H:\github\python\vis_interpolate\data\cldas_score")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # 初始化统计结果列表
    all_stats_national = []
    all_stats_regional = []

    # 时间循环
    current_date = start_date
    total_hours = int((end_date - start_date).total_seconds() / 3600)
    processed_national = 0
    processed_regional = 0
    skipped = 0

    print("=" * 80)
    print(f"能见度预报检验 (国家站 + 区域站)")
    print(f"起始时间: {start_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"结束时间: {end_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"总时次数: {total_hours}")
    print("=" * 80)

    while current_date <= end_date:
        time_str = current_date.strftime('%Y%m%d%H')
        print(f"\n处理时次: {current_date.strftime('%Y-%m-%d %H:%M')} ({processed_national + processed_regional // 2 + skipped + 1}/{total_hours})")

        # 执行国家站检验
        print("  [国家站]", end=" ")
        results_df_national, stats_national = get_vis_score_by_time(
            current_date, fields, fields_2, field_map,
            station_type="national", output_dir=output_dir, use_cache=True
        )

        if results_df_national is not None and stats_national is not None:
            # 保存详细结果（仅当不是从缓存读取时）
            detail_file = output_dir / f"vis_score_detail_national_{time_str}.csv"
            if not detail_file.exists():
                results_df_national.to_csv(detail_file, index=False, encoding='utf-8-sig')

            # 添加时间信息到统计结果
            stats_national['datetime'] = current_date
            stats_national['time_str'] = time_str
            stats_national['station_type'] = 'national'
            all_stats_national.append(stats_national)

            print(f"✓ 样本数: {stats_national['n_stations']} | MAE: {stats_national['mae']:.2f} km | RMSE: {stats_national['rmse']:.2f} km | R: {stats_national['correlation']:.4f}")
            processed_national += 1
        else:
            print("✗ 数据不可用")

        # 执行区域站检验
        print("  [区域站]", end=" ")
        results_df_regional, stats_regional = get_vis_score_by_time(
            current_date, fields, fields_2, field_map,
            station_type="regional", output_dir=output_dir, use_cache=True
        )

        if results_df_regional is not None and stats_regional is not None:
            # 保存详细结果（仅当不是从缓存读取时）
            detail_file = output_dir / f"vis_score_detail_regional_{time_str}.csv"
            if not detail_file.exists():
                results_df_regional.to_csv(detail_file, index=False, encoding='utf-8-sig')

            # 添加时间信息到统计结果
            stats_regional['datetime'] = current_date
            stats_regional['time_str'] = time_str
            stats_regional['station_type'] = 'regional'
            all_stats_regional.append(stats_regional)

            print(f"✓ 样本数: {stats_regional['n_stations']} | MAE: {stats_regional['mae']:.2f} km | RMSE: {stats_regional['rmse']:.2f} km | R: {stats_regional['correlation']:.4f}")
            processed_regional += 1
        else:
            print("✗ 数据不可用")

        # 如果国家站和区域站都失败，计数跳过
        if (results_df_national is None or stats_national is None) and \
           (results_df_regional is None or stats_regional is None):
            skipped += 1

        # 移动到下一个时次（每小时）
        current_date += timedelta(hours=1)

    # 保存汇总统计结果
    print("\n" + "=" * 80)
    print("检验完成!")
    print(f"国家站成功处理: {processed_national} 个时次")
    print(f"区域站成功处理: {processed_regional} 个时次")
    print(f"完全跳过时次: {skipped} 个")
    print(f"详细结果保存至: {output_dir}")
    print("=" * 80)

    # 保存国家站汇总
    if all_stats_national:
        summary_df_national = pd.DataFrame(all_stats_national)
        summary_file_national = output_dir / "vis_score_summary_national.csv"
        summary_df_national.to_csv(summary_file_national, index=False, encoding='utf-8-sig')

        print("\n国家站整体统计 (所有时次平均):")
        print(f"  汇总文件: {summary_file_national}")
        print(f"  平均样本数: {summary_df_national['n_stations'].mean():.0f}")
        print(f"  平均观测值: {summary_df_national['mean_obs'].mean():.2f} km")
        print(f"  平均预报值: {summary_df_national['mean_forecast'].mean():.2f} km")
        print(f"  平均偏差 (Bias): {summary_df_national['bias'].mean():.2f} km")
        print(f"  平均绝对误差 (MAE): {summary_df_national['mae'].mean():.2f} km")
        print(f"  均方根误差 (RMSE): {summary_df_national['rmse'].mean():.2f} km")
        print(f"  平均相关系数 (R): {summary_df_national['correlation'].mean():.4f}")
        print(f"  平均相对误差 (MAPE): {summary_df_national['mape'].mean():.2f}%")

    # 保存区域站汇总
    if all_stats_regional:
        summary_df_regional = pd.DataFrame(all_stats_regional)
        summary_file_regional = output_dir / "vis_score_summary_regional.csv"
        summary_df_regional.to_csv(summary_file_regional, index=False, encoding='utf-8-sig')

        print("\n区域站整体统计 (所有时次平均):")
        print(f"  汇总文件: {summary_file_regional}")
        print(f"  平均样本数: {summary_df_regional['n_stations'].mean():.0f}")
        print(f"  平均观测值: {summary_df_regional['mean_obs'].mean():.2f} km")
        print(f"  平均预报值: {summary_df_regional['mean_forecast'].mean():.2f} km")
        print(f"  平均偏差 (Bias): {summary_df_regional['bias'].mean():.2f} km")
        print(f"  平均绝对误差 (MAE): {summary_df_regional['mae'].mean():.2f} km")
        print(f"  均方根误差 (RMSE): {summary_df_regional['rmse'].mean():.2f} km")
        print(f"  平均相关系数 (R): {summary_df_regional['correlation'].mean():.4f}")
        print(f"  平均相对误差 (MAPE): {summary_df_regional['mape'].mean():.2f}%")

    if not all_stats_national and not all_stats_regional:
        print("\n未能处理任何时次，请检查数据源。")


if __name__ == "__main__":
    main()
