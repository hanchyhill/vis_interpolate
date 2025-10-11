"""
测试优化后的各向异性IDW插值性能和正确性
"""

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from vis_dem_dis import deg2km, deg2km_vectorized, anisotropic_idw_interpolation

print("=" * 80)
print("测试优化后的各向异性IDW插值函数")
print("=" * 80)

# 测试1: deg2km_vectorized正确性测试
print("\n[测试1] deg2km_vectorized 函数正确性")
print("-" * 80)

lat1, lon1 = 23.0, 113.0
lat2_array = np.array([23.1, 23.2, 23.3, 23.5, 24.0])
lon2_array = np.array([113.1, 113.2, 113.3, 113.5, 114.0])

# 使用原函数逐个计算
distances_original = np.array([deg2km(lat1, lon1, lat2, lon2)
                               for lat2, lon2 in zip(lat2_array, lon2_array)])

# 使用向量化函数批量计算
distances_vectorized = deg2km_vectorized(lat1, lon1, lat2_array, lon2_array)

# 比较结果
diff = np.abs(distances_original - distances_vectorized)
max_diff = np.max(diff)
mean_diff = np.mean(diff)

print(f"原函数结果: {distances_original}")
print(f"向量化结果: {distances_vectorized}")
print(f"最大误差: {max_diff:.10f} km")
print(f"平均误差: {mean_diff:.10f} km")

if max_diff < 1e-10:
    print("[PASS] deg2km_vectorized 正确性测试通过!")
else:
    print("[FAIL] deg2km_vectorized 结果不匹配，需要检查!")

# 测试2: 性能对比
print("\n[测试2] 性能对比测试")
print("-" * 80)

# 生成测试数据
np.random.seed(42)
n_points = 10000

lat1_test = 23.0
lon1_test = 113.0
lat2_test = np.random.uniform(22.0, 24.0, n_points)
lon2_test = np.random.uniform(112.0, 114.0, n_points)

# 原函数性能
start = time.time()
for i in range(n_points):
    _ = deg2km(lat1_test, lon1_test, lat2_test[i], lon2_test[i])
time_original = time.time() - start

# 向量化函数性能
start = time.time()
_ = deg2km_vectorized(lat1_test, lon1_test, lat2_test, lon2_test)
time_vectorized = time.time() - start

speedup = time_original / time_vectorized

print(f"原函数耗时: {time_original:.4f} 秒 (计算{n_points}个点)")
print(f"向量化耗时: {time_vectorized:.4f} 秒")
print(f"加速比: {speedup:.2f}x")

if speedup > 2:
    print(f"[PASS] 性能提升显著 ({speedup:.1f}倍加速)!")
else:
    print(f"[WARN] 性能提升有限 ({speedup:.1f}倍加速)")

# 测试3: 完整插值函数测试（小规模）
print("\n[测试3] 完整插值函数测试（小规模）")
print("-" * 80)

# 创建模拟站点数据
n_stations = 50
df_station = pd.DataFrame({
    'lon': np.random.uniform(112.0, 114.0, n_stations),
    'lat': np.random.uniform(22.0, 24.0, n_stations),
    'altitude': np.random.uniform(0, 1000, n_stations),
    'vis': np.random.uniform(1.0, 30.0, n_stations)
})

# 创建目标网格（小规模测试）
n_grid = 100
target_lons = np.random.uniform(112.5, 113.5, n_grid)
target_lats = np.random.uniform(22.5, 23.5, n_grid)
target_elevations = np.random.uniform(0, 800, n_grid)

print(f"站点数量: {n_stations}")
print(f"目标点数量: {n_grid}")

# 执行插值
start = time.time()
result = anisotropic_idw_interpolation(
    df_station, target_lons, target_lats, target_elevations,
    beta=10.0, power=2.0, n_neighbors=6
)
elapsed = time.time() - start

# 检查结果
valid_results = np.sum(~np.isnan(result))
vis_mean = np.nanmean(result)
vis_min = np.nanmin(result)
vis_max = np.nanmax(result)

print(f"\n插值耗时: {elapsed:.4f} 秒")
print(f"有效结果: {valid_results}/{n_grid} ({valid_results/n_grid*100:.1f}%)")
print(f"能见度范围: {vis_min:.2f} - {vis_max:.2f} km")
print(f"平均能见度: {vis_mean:.2f} km")

if valid_results > 0:
    print("[PASS] 小规模插值测试成功!")
else:
    print("[FAIL] 插值失败，未生成有效结果!")

# 测试4: 大规模性能测试
print("\n[测试4] 大规模性能测试（模拟实际场景）")
print("-" * 80)

# 模拟实际场景：100个站点，50000个网格点
n_stations_large = 100
df_station_large = pd.DataFrame({
    'lon': np.random.uniform(112.0, 115.0, n_stations_large),
    'lat': np.random.uniform(22.0, 25.0, n_stations_large),
    'altitude': np.random.uniform(0, 1500, n_stations_large),
    'vis': np.random.uniform(1.0, 30.0, n_stations_large)
})

n_grid_large = 50000
target_lons_large = np.random.uniform(112.5, 114.5, n_grid_large)
target_lats_large = np.random.uniform(22.5, 24.5, n_grid_large)
target_elevations_large = np.random.uniform(0, 1200, n_grid_large)

print(f"站点数量: {n_stations_large}")
print(f"目标点数量: {n_grid_large}")

# 执行插值
start = time.time()
result_large = anisotropic_idw_interpolation(
    df_station_large, target_lons_large, target_lats_large, target_elevations_large,
    beta=10.0, power=2.0, n_neighbors=6
)
elapsed_large = time.time() - start

# 检查结果
valid_results_large = np.sum(~np.isnan(result_large))
throughput = n_grid_large / elapsed_large

print(f"\n插值耗时: {elapsed_large:.2f} 秒")
print(f"有效结果: {valid_results_large}/{n_grid_large} ({valid_results_large/n_grid_large*100:.1f}%)")
print(f"处理速度: {throughput:.0f} 点/秒")
print(f"预计100万网格点耗时: {1000000 / throughput:.1f} 秒 ({1000000 / throughput / 60:.1f} 分钟)")

if valid_results_large > n_grid_large * 0.95:
    print("[PASS] 大规模插值测试成功!")
else:
    print("[WARN] 部分插值失败，有效率低于95%")

print("\n" + "=" * 80)
print("所有测试完成!")
print("=" * 80)
print("\n性能总结:")
print(f"  deg2km 向量化加速: {speedup:.1f}x")
print(f"  大规模插值速度: {throughput:.0f} 点/秒")
print(f"  预计100万点耗时: {1000000 / throughput / 60:.1f} 分钟")
print("=" * 80)
