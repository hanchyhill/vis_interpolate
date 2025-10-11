"""
测试区域站评估功能
"""

from datetime import datetime
from src.evaluate_visibility import get_vis_score_by_time

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

# 测试时间
test_time = datetime(2025, 2, 28, 0, 0)

print("=" * 80)
print("测试区域站评估功能")
print("=" * 80)

# 测试国家站
print("\n测试国家站评估:")
print("-" * 80)
results_national, stats_national = get_vis_score_by_time(
    test_time, fields, fields_2, field_map, station_type="national"
)

if results_national is not None and stats_national is not None:
    print(f"✓ 国家站评估成功")
    print(f"  样本数: {stats_national['n_stations']}")
    print(f"  平均观测值: {stats_national['mean_obs']:.2f} km")
    print(f"  平均预报值: {stats_national['mean_forecast']:.2f} km")
    print(f"  MAE: {stats_national['mae']:.2f} km")
    print(f"  RMSE: {stats_national['rmse']:.2f} km")
    print(f"  相关系数: {stats_national['correlation']:.4f}")
    print(f"\n前5个站点详细结果:")
    print(results_national[['name', 'vis_obs', 'vis_forecast', 'error', 'abs_error']].head())
else:
    print("✗ 国家站评估失败")

# 测试区域站
print("\n" + "-" * 80)
print("测试区域站评估:")
print("-" * 80)
results_regional, stats_regional = get_vis_score_by_time(
    test_time, fields, fields_2, field_map, station_type="regional"
)

if results_regional is not None and stats_regional is not None:
    print(f"✓ 区域站评估成功")
    print(f"  样本数: {stats_regional['n_stations']}")
    print(f"  平均观测值: {stats_regional['mean_obs']:.2f} km")
    print(f"  平均预报值: {stats_regional['mean_forecast']:.2f} km")
    print(f"  MAE: {stats_regional['mae']:.2f} km")
    print(f"  RMSE: {stats_regional['rmse']:.2f} km")
    print(f"  相关系数: {stats_regional['correlation']:.4f}")
    print(f"\n前5个站点详细结果:")
    print(results_regional[['name', 'vis_obs', 'vis_forecast', 'error', 'abs_error']].head())
else:
    print("✗ 区域站评估失败")

print("\n" + "=" * 80)
print("测试完成!")
print("=" * 80)
