"""基于 0.01° DEM 生成山谷（辐射雾潜在沉积区）边界，多算法多阈值对比。

坐标系：WGS84 经纬度；海拔单位 m；距离/半径单位 km。

三类算法：
- tpi : TPI = DEM - 邻域均值，TPI <= -阈值 为山谷（基线方法）。
- fill: 形态学开运算得到局地谷底面 floor，局地山顶 peak = 邻域最大值，
        雾厚 = min(H, (peak - floor) * 1/3)，DEM - floor <= 雾厚 为山谷。
- unit: 以谷底核心为种子做分水岭分割得到山谷单元，每个单元
        雾顶 = 谷底平均海拔 + min(H, (单元山顶 - 谷底平均海拔) * 1/3)。

用法：
    uv run python -m src.valley_boundary
    uv run python -m src.valley_boundary --out-dir output/valley_boundary
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import shapes
from rasterio.transform import Affine
from scipy import ndimage as ndi
from shapely import contains_xy
from shapely.geometry import shape
from skimage.segmentation import watershed

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LAND_MIN_ELEVATION_M = 0.5
EARTH_KM_PER_DEG = 111.32
STRUCTURE_8 = np.ones((3, 3), dtype=bool)

# 卫星辐射雾示例图中的参考城市（经度, 纬度）
REFERENCE_CITIES = {
    "广州": (113.26, 23.13), "韶关": (113.60, 24.81), "清远": (113.06, 23.68),
    "河源": (114.70, 23.74), "梅州": (116.12, 24.29), "惠州": (114.42, 23.11),
    "肇庆": (112.47, 23.05), "云浮": (112.04, 22.92), "茂名": (110.93, 21.66),
    "阳江": (111.98, 21.86), "湛江": (110.36, 21.27), "江门": (113.08, 22.58),
    "汕头": (116.68, 23.35), "揭阳": (116.37, 23.55), "汕尾": (115.37, 22.79),
    "深圳": (114.06, 22.54),
}


@dataclass(frozen=True)
class ValleyConfig:
    """一组山谷识别参数。

    radius_km: 谷底/TPI 邻域半径 (km)；ridge_radius_km: 山顶搜索半径 (km)，默认 2*radius_km；
    threshold_m: tpi 为 TPI 阈值，fill/unit 为雾厚上限 H (m)；
    relief_fraction: 雾厚不超过 (山顶-谷底) 的比例；
    min_relief_m / min_enclosed_dirs: 在 ridge_km 距离内，至少 min_enclosed_dirs 个方向（共 8 个）
    有高出本点 min_relief_m 的山体才算封闭山谷，用于排除平原。
    min_area_km2: 剔除小于该面积的碎块，tpi/hybrid 同时填补小于该面积的内部空洞；
    hybrid: TPI 山谷按 unit 集水单元（unit_radius_km, unit_threshold_m）切分，
            保留封闭像元比例 >= min_overlap 的山谷；
    fusion: TPI 山谷外扩 widen_px 像元定边缘，限制在 unit 雾区外扩 reach_px 像元内，按 unit 编号分块；
            nearest_unit: 外扩像元归属最近的 unit 雾区（否则按集水单元）；
            merge_slack_m: 非 None 时，鞍部 <= 两谷雾顶较低者 + merge_slack_m 的相邻山谷合并，
            合并面积上限 max_merged_km2；
    group: 对比图分组名，默认同 method。
    """

    name: str
    method: str
    radius_km: float
    threshold_m: float
    ridge_radius_km: float | None = None
    relief_fraction: float = 1.0 / 3.0
    min_relief_m: float = 150.0
    min_enclosed_dirs: int = 6
    core_m: float = 30.0
    band_m: float = 100.0
    min_area_km2: float = 5.0
    unit_radius_km: float = 10.0
    unit_threshold_m: float = 100.0
    min_overlap: float = 0.5
    widen_px: int = 0
    reach_px: int = 3
    nearest_unit: bool = False
    merge_slack_m: float | None = None
    max_merged_km2: float = np.inf
    group: str = ""

    @property
    def ridge_km(self) -> float:
        return self.ridge_radius_km if self.ridge_radius_km is not None else 2.0 * self.radius_km

    @property
    def figure_group(self) -> str:
        return self.group or self.method

    @property
    def label(self) -> str:
        if self.method in ("tpi", "hybrid"):
            text = f"TPI R={self.radius_km:g}km, TPI≤-{self.threshold_m:g}m, 面积≥{self.min_area_km2:g}km2"
            if self.method == "hybrid":
                text += (f"\n按 unit(R={self.unit_radius_km:g}km) 集水单元切分, "
                         f"封闭≥{self.min_enclosed_dirs - 2}/8方向的像元≥{self.min_overlap:.0%}")
            return text
        if self.method == "fusion":
            return (f"TPI R={self.radius_km:g}km, TPI≤-{self.threshold_m:g}m, 外扩{self.widen_px}px\n"
                    f"限于 unit(R={self.unit_radius_km:g}km, H={self.unit_threshold_m:g}m) 雾区外扩"
                    f"{self.reach_px}px, 面积≥{self.min_area_km2:g}km2"
                    + (", 归属最近单元" if self.nearest_unit else "")
                    + (f"\n鞍部≤雾顶+{self.merge_slack_m:g}m 合并, 上限{self.max_merged_km2:g}km2"
                       if self.merge_slack_m is not None else ""))
        return (f"{self.method} R={self.radius_km:g}km, 雾厚≤min({self.threshold_m:g}m, 起伏×{self.relief_fraction:.2g}), "
                f"封闭≥{self.min_enclosed_dirs}/8方向(高差{self.min_relief_m:g}m, {self.ridge_km:g}km内)")


DEFAULT_CONFIGS: tuple[ValleyConfig, ...] = (
    ValleyConfig("tpi_r5_t30", "tpi", 5, 30),
    ValleyConfig("tpi_r10_t50", "tpi", 10, 50),
    ValleyConfig("tpi_r10_t100", "tpi", 10, 100),
    ValleyConfig("tpi_r20_t80", "tpi", 20, 80),
    ValleyConfig("fill_r10_h100_e6", "fill", 10, 100),
    ValleyConfig("fill_r10_h100_rise100_e6", "fill", 10, 100, min_relief_m=100),
    ValleyConfig("fill_r20_h100_e6", "fill", 20, 100),
    ValleyConfig("fill_r20_h100_e7", "fill", 20, 100, min_enclosed_dirs=7),
    ValleyConfig("unit_r10_h100_e6", "unit", 10, 100),
    ValleyConfig("unit_r10_h200_e6", "unit", 10, 200),
    ValleyConfig("unit_r20_h100_e6", "unit", 20, 100),
    ValleyConfig("unit_r20_h100_e7", "unit", 20, 100, min_enclosed_dirs=7),
    ValleyConfig("tpi_r5_t30_a20", "tpi", 5, 30, min_area_km2=20, group="tpi_refined"),
    ValleyConfig("tpi_r5_t30_a50", "tpi", 5, 30, min_area_km2=50, group="tpi_refined"),
    ValleyConfig("hybrid_r5_t30_a20", "hybrid", 5, 30, min_area_km2=20, group="tpi_refined"),
    ValleyConfig("hybrid_r5_t30_a20_e7", "hybrid", 5, 30, min_area_km2=20, min_enclosed_dirs=7,
                 group="tpi_refined"),
    ValleyConfig("fusion_t30_w0_r3", "fusion", 5, 30, min_area_km2=20),
    ValleyConfig("fusion_t30_w1_r3", "fusion", 5, 30, min_area_km2=20, widen_px=1),
    ValleyConfig("fusion_t20_w0_r3", "fusion", 5, 20, min_area_km2=20),
    ValleyConfig("fusion_t30_w1_r5", "fusion", 5, 30, min_area_km2=20, widen_px=1, reach_px=5),
    ValleyConfig("fusion_t20_near", "fusion", 5, 20, min_area_km2=20, nearest_unit=True, group="fusion_merge"),
    ValleyConfig("fusion_t20_near_m0", "fusion", 5, 20, min_area_km2=20, nearest_unit=True,
                 merge_slack_m=0, max_merged_km2=1500, group="fusion_merge"),
    ValleyConfig("fusion_t20_near_m50", "fusion", 5, 20, min_area_km2=20, nearest_unit=True,
                 merge_slack_m=50, max_merged_km2=1500, group="fusion_merge"),
    ValleyConfig("fusion_t20_near_m50_cap3000", "fusion", 5, 20, min_area_km2=20, nearest_unit=True,
                 merge_slack_m=50, max_merged_km2=3000, group="fusion_merge"),
)

# 粤北、粤东山区放大审查范围 (lon_min, lon_max, lat_min, lat_max)
ZOOM_EXTENT = (112.0, 117.0, 23.3, 25.6)
ZOOM_CONFIGS = ("tpi_r5_t30", "fusion_t20_w0_r3",
                "fusion_t20_near", "fusion_t20_near_m0", "fusion_t20_near_m50", "fusion_t20_near_m50_cap3000")


@dataclass
class DemGrid:
    elevation: np.ndarray  # 平滑后的海拔 (m)，非陆地为 0
    land: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    dy_km: float
    dx_km: float

    @property
    def pixel_area_km2(self) -> np.ndarray:
        """逐行像元面积 (km²)，形状 (lat, 1)。"""
        return (self.dy_km * self.dx_km * np.cos(np.deg2rad(self.lat))
                / np.cos(np.deg2rad(self.lat.mean())))[:, None]


def load_dem(path: Path, smooth_sigma_px: float = 1.0) -> DemGrid:
    with xr.open_dataset(path) as ds:
        dem = ds["elevation"].load()
    if dem.lat.values[0] > dem.lat.values[-1]:
        dem = dem.sortby("lat")
    values = dem.values.astype(np.float64)
    land = np.isfinite(values) & (values > LAND_MIN_ELEVATION_M)
    work = np.where(land, values, 0.0)
    if smooth_sigma_px > 0:
        work = np.where(land, ndi.gaussian_filter(work, smooth_sigma_px), 0.0)
    lat = dem.lat.values
    lon = dem.lon.values
    dy_km = float(np.abs(np.diff(lat)).mean()) * EARTH_KM_PER_DEG
    dx_km = float(np.abs(np.diff(lon)).mean()) * EARTH_KM_PER_DEG * np.cos(np.deg2rad(lat.mean()))
    return DemGrid(work, land, lat, lon, dy_km, dx_km)


def window_size(radius_km: float, grid: DemGrid) -> tuple[int, int]:
    """半径 (km) → 奇数窗口像元 (ny, nx)。"""
    ry = max(1, int(round(radius_km / grid.dy_km)))
    rx = max(1, int(round(radius_km / grid.dx_km)))
    return 2 * ry + 1, 2 * rx + 1


def compute_tpi(grid: DemGrid, radius_km: float) -> np.ndarray:
    """仅用陆地像元求邻域均值，避免海面 0 值拉低沿海 TPI。"""
    size = window_size(radius_km, grid)
    weight = grid.land.astype(np.float64)
    total = ndi.uniform_filter(grid.elevation * weight, size=size, mode="nearest")
    count = ndi.uniform_filter(weight, size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        tpi = grid.elevation - total / count
    return np.where(grid.land, tpi, np.nan)


def compute_floor_relief(grid: DemGrid, radius_km: float, ridge_radius_km: float
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (谷底面 floor, 高出谷底 height_above_floor, 局地起伏 relief)，单位 m。

    floor 为灰度开运算（去除窄于窗口的山体）再均值平滑；relief = 山顶搜索窗口内最大值 - floor。
    """
    size = window_size(radius_km, grid)
    floor = ndi.grey_opening(grid.elevation, size=size, mode="nearest")
    floor = ndi.uniform_filter(floor, size=(size[0] // 2 * 2 + 1, size[1] // 2 * 2 + 1), mode="nearest")
    floor = np.minimum(floor, grid.elevation)
    peak = ndi.maximum_filter(grid.elevation, size=window_size(ridge_radius_km, grid), mode="nearest")
    height_above_floor = grid.elevation - floor
    relief = peak - floor
    nan = np.nan
    return (np.where(grid.land, floor, nan), np.where(grid.land, height_above_floor, nan),
            np.where(grid.land, relief, nan))


def compute_enclosure(grid: DemGrid, distance_km: float, rise_m: float) -> np.ndarray:
    """封闭度：8 个方向中，distance_km 内地形比本点高出 >= rise_m 的方向数 (0-8)。

    冷空气只在多方向被山体围住的地方堆积，用于区分山谷与开阔平原。
    """
    elev = np.where(grid.land, grid.elevation, -np.inf)
    ny, nx = elev.shape
    pad = max(1, int(round(distance_km / min(grid.dy_km, grid.dx_km))))
    padded = np.pad(elev, pad, mode="constant", constant_values=-np.inf)
    count = np.zeros(elev.shape, dtype=np.int8)
    for dy, dx in ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)):
        step_km = float(np.hypot(dy * grid.dy_km, dx * grid.dx_km))
        steps = max(1, int(round(distance_km / step_km)))
        ray_max = np.full(elev.shape, -np.inf)
        for k in range(1, steps + 1):
            oy, ox = pad + dy * k, pad + dx * k
            np.maximum(ray_max, padded[oy:oy + ny, ox:ox + nx], out=ray_max)
        count += (ray_max - grid.elevation >= rise_m).astype(np.int8)
    return np.where(grid.land, count, 0).astype(np.int8)


def fog_thickness(relief_m: np.ndarray, max_thickness_m: float, relief_fraction: float) -> np.ndarray:
    """雾厚 = min(H, 起伏 * 比例)。"""
    return np.minimum(max_thickness_m, relief_m * relief_fraction)


def remove_small_regions(mask: np.ndarray, grid: DemGrid, min_area_km2: float) -> np.ndarray:
    labels, n = ndi.label(mask, structure=STRUCTURE_8)
    if n == 0:
        return mask
    area = np.bincount(labels.ravel(), weights=np.broadcast_to(grid.pixel_area_km2, mask.shape).ravel())
    keep = area >= min_area_km2
    keep[0] = False
    return keep[labels]


def fill_small_holes(mask: np.ndarray, grid: DemGrid, max_hole_km2: float) -> np.ndarray:
    holes = grid.land & ~mask
    small_holes = holes & ~remove_small_regions(holes, grid, max_hole_km2)
    return mask | small_holes


def tpi_valley_mask(grid: DemGrid, cfg: ValleyConfig) -> np.ndarray:
    tpi = compute_tpi(grid, cfg.radius_km)
    mask = grid.land & (np.nan_to_num(tpi, nan=np.inf) <= -cfg.threshold_m)
    mask = remove_small_regions(mask, grid, cfg.min_area_km2)
    return fill_small_holes(mask, grid, cfg.min_area_km2)


def hybrid_valley_mask(grid: DemGrid, cfg: ValleyConfig) -> tuple[np.ndarray, np.ndarray]:
    """TPI 山谷形状 + 分水岭山谷单元切分，返回 (山谷掩膜, 山谷编号)。

    TPI 山谷按 unit 集水单元切开（同一单元内的 TPI 支谷归为同一山谷），
    只保留封闭像元比例 >= min_overlap 且面积 >= min_area_km2 的山谷。
    河谷顺河方向敞开，封闭像元与 unit 雾区一致按 min_enclosed_dirs - 2 判定。
    """
    tpi_mask = tpi_valley_mask(grid, cfg)
    unit_cfg = ValleyConfig("_unit", "unit", cfg.unit_radius_km, cfg.unit_threshold_m,
                            relief_fraction=cfg.relief_fraction, min_relief_m=cfg.min_relief_m,
                            min_enclosed_dirs=cfg.min_enclosed_dirs)
    units, enclosure = valley_unit_partition(grid, unit_cfg)
    labels = np.where(tpi_mask, units, 0).astype(np.int32)
    n = int(labels.max())
    if n == 0:
        return labels > 0, labels
    area_px = np.broadcast_to(grid.pixel_area_km2, labels.shape).ravel()
    area = np.bincount(labels.ravel(), weights=area_px, minlength=n + 1)
    total = np.bincount(labels.ravel(), minlength=n + 1)
    enclosed = np.bincount(labels.ravel(), weights=(enclosure >= cfg.min_enclosed_dirs - 2).ravel(),
                           minlength=n + 1)
    keep = (area >= cfg.min_area_km2) & (enclosed >= cfg.min_overlap * np.maximum(total, 1))
    keep[0] = False
    labels = np.where(keep[labels], labels, 0)
    return labels > 0, labels


def fusion_valley_mask(grid: DemGrid, cfg: ValleyConfig) -> tuple[np.ndarray, np.ndarray]:
    """unit 定分块与范围、TPI 定边缘，返回 (山谷掩膜, 山谷编号=unit 集水单元编号)。

    山谷 = (TPI 山谷(外扩 widen_px) ∪ unit 谷底核心) ∩ (unit 雾区外扩 reach_px)，
    谷底核心用于补齐宽盆地中心 TPI≈0 的空洞；reach_px 限制 TPI 沿河网无限延伸。
    """
    tpi = compute_tpi(grid, cfg.radius_km)
    tpi_mask = grid.land & (np.nan_to_num(tpi, nan=np.inf) <= -cfg.threshold_m)
    if cfg.widen_px > 0:
        tpi_mask = ndi.binary_dilation(tpi_mask, structure=STRUCTURE_8, iterations=cfg.widen_px) & grid.land

    unit_cfg = ValleyConfig("_unit", "unit", cfg.unit_radius_km, cfg.unit_threshold_m,
                            relief_fraction=cfg.relief_fraction, min_relief_m=cfg.min_relief_m,
                            min_enclosed_dirs=cfg.min_enclosed_dirs)
    partition = _partition_with_markers(grid, unit_cfg)
    units, markers, _ = partition
    unit_mask, unit_ids, table = unit_valley_mask(grid, unit_cfg, partition)
    if not unit_mask.any():
        return unit_mask, unit_ids
    reach = unit_mask
    if cfg.reach_px > 0:
        reach = ndi.binary_dilation(unit_mask, structure=STRUCTURE_8, iterations=cfg.reach_px)

    mask = (tpi_mask | (markers > 0)) & reach & grid.land
    mask = fill_small_holes(mask, grid, cfg.min_area_km2)
    if cfg.nearest_unit:
        _, (iy, ix) = ndi.distance_transform_edt(unit_ids == 0, return_indices=True)
        owner = unit_ids[iy, ix]
    else:
        owner = units
    labels = np.where(mask, owner, 0).astype(np.int32)
    if cfg.merge_slack_m is not None:
        fog_top = np.full(int(labels.max()) + 1, -np.inf)
        valid = table["unit_id"].to_numpy() < fog_top.size
        fog_top[table["unit_id"].to_numpy()[valid]] = table["fog_top_m"].to_numpy()[valid]
        labels = merge_connected_valleys(labels, grid, fog_top, cfg.merge_slack_m, cfg.max_merged_km2)
    area_px = np.broadcast_to(grid.pixel_area_km2, labels.shape).ravel()
    area = np.bincount(labels.ravel(), weights=area_px, minlength=int(labels.max()) + 1)
    keep = area >= cfg.min_area_km2
    keep[0] = False
    labels = np.where(keep[labels], labels, 0)
    return labels > 0, labels


def merge_connected_valleys(labels: np.ndarray, grid: DemGrid, fog_top_m: np.ndarray,
                            slack_m: float, max_area_km2: float) -> np.ndarray:
    """合并雾层可连通的相邻山谷。

    鞍部 = 两山谷交界处像元对的 max(海拔) 的最小值；鞍部 <= min(两谷雾顶) + slack_m 时合并。
    按超出量从小到大贪心合并，合并后面积不超过 max_area_km2。
    """
    elev = grid.elevation
    lo_parts, hi_parts, saddle_parts = [], [], []
    ny, nx = labels.shape
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if dx >= 0:
            sa = (slice(0, ny - dy), slice(0, nx - dx))
            sb = (slice(dy, ny), slice(dx, nx))
        else:
            sa = (slice(0, ny - dy), slice(-dx, nx))
            sb = (slice(dy, ny), slice(0, nx + dx))
        a, b, ea, eb = labels[sa], labels[sb], elev[sa], elev[sb]
        touch = (a > 0) & (b > 0) & (a != b)
        lo_parts.append(np.minimum(a, b)[touch])
        hi_parts.append(np.maximum(a, b)[touch])
        saddle_parts.append(np.maximum(ea, eb)[touch])
    if not any(part.size for part in lo_parts):
        return labels
    pairs = (pd.DataFrame({"lo": np.concatenate(lo_parts), "hi": np.concatenate(hi_parts),
                           "saddle": np.concatenate(saddle_parts)})
             .groupby(["lo", "hi"], as_index=False)["saddle"].min())
    pairs["excess"] = pairs["saddle"] - np.minimum(fog_top_m[pairs["lo"]], fog_top_m[pairs["hi"]])
    pairs = pairs[pairs["excess"] <= slack_m].sort_values("excess")

    area_px = np.broadcast_to(grid.pixel_area_km2, labels.shape).ravel()
    area = np.bincount(labels.ravel(), weights=area_px, minlength=int(labels.max()) + 1)
    parent = np.arange(area.size)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for lo, hi in zip(pairs["lo"].to_numpy(), pairs["hi"].to_numpy()):
        ra, rb = find(lo), find(hi)
        if ra != rb and area[ra] + area[rb] <= max_area_km2:
            parent[rb] = ra
            area[ra] += area[rb]
    roots = np.array([find(i) for i in range(parent.size)])
    return roots[labels] * (labels > 0)


def fill_valley_mask(grid: DemGrid, cfg: ValleyConfig) -> np.ndarray:
    _, hab, relief = compute_floor_relief(grid, cfg.radius_km, cfg.ridge_km)
    relief0 = np.nan_to_num(relief, nan=0.0)
    thickness = fog_thickness(relief0, cfg.threshold_m, cfg.relief_fraction)
    enclosure = compute_enclosure(grid, cfg.ridge_km, cfg.min_relief_m)
    mask = (grid.land & (enclosure >= cfg.min_enclosed_dirs)
            & (np.nan_to_num(hab, nan=np.inf) <= thickness))
    return remove_small_regions(mask, grid, cfg.min_area_km2)


def valley_unit_partition(grid: DemGrid, cfg: ValleyConfig) -> tuple[np.ndarray, np.ndarray]:
    """以谷底核心为种子做分水岭分割，返回 (全陆地山谷单元编号, 封闭度)。"""
    units, _, enclosure = _partition_with_markers(grid, cfg)
    return units, enclosure


def _partition_with_markers(grid: DemGrid, cfg: ValleyConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """谷底核心 = 高出 radius_km 邻域最低点 <= core_m 的封闭像元；
    核心按谷底海拔分带（band_m）切分，避免长河谷上下游共用一个雾顶。
    """
    floor = ndi.minimum_filter(np.where(grid.land, grid.elevation, np.inf),
                               size=window_size(cfg.radius_km, grid), mode="nearest")
    enclosure = compute_enclosure(grid, cfg.ridge_km, cfg.min_relief_m)
    core = (grid.land & (enclosure >= cfg.min_enclosed_dirs)
            & (grid.elevation - floor <= cfg.core_m))
    core = remove_small_regions(core, grid, cfg.min_area_km2)

    band = np.floor(np.where(core, floor, 0.0) / cfg.band_m).astype(np.int32)
    markers = np.zeros(core.shape, dtype=np.int32)
    next_id = 0
    for b in np.unique(band[core]):
        labels, n = ndi.label(core & (band == b), structure=STRUCTURE_8)
        markers[labels > 0] = labels[labels > 0] + next_id
        next_id += n
    if next_id == 0:
        return markers, markers, enclosure

    area_px = np.broadcast_to(grid.pixel_area_km2, core.shape)
    core_area = np.bincount(markers.ravel(), weights=area_px.ravel(), minlength=next_id + 1)
    small = core_area < cfg.min_area_km2
    small[0] = False
    markers[small[markers]] = 0
    units = watershed(grid.elevation, markers=markers, mask=grid.land, connectivity=2)
    return units.astype(np.int32), markers, enclosure


def unit_valley_mask(grid: DemGrid, cfg: ValleyConfig,
                     partition: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
                     ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """返回 (山谷掩膜, 山谷单元编号, 单元属性表)。partition 为 _partition_with_markers 的结果，可复用。"""
    units, markers, enclosure = partition if partition is not None else _partition_with_markers(grid, cfg)
    if not markers.any():
        return np.zeros(markers.shape, dtype=bool), np.zeros(markers.shape, dtype=np.int32), pd.DataFrame()
    area_px = np.broadcast_to(grid.pixel_area_km2, markers.shape)
    core = markers > 0
    ids = np.unique(markers[markers > 0])
    valley_mean = np.asarray(ndi.mean(grid.elevation, labels=markers, index=ids))
    peak = np.asarray(ndi.labeled_comprehension(
        grid.elevation, units, ids, lambda v: np.percentile(v, 95), np.float64, np.nan))
    thickness = fog_thickness(peak - valley_mean, cfg.threshold_m, cfg.relief_fraction)
    fog_top = valley_mean + thickness

    top_lut = np.full(int(units.max()) + 1, -np.inf)
    top_lut[ids] = fog_top
    mask = (grid.land & (units > 0) & (grid.elevation <= top_lut[units])
            & (enclosure >= cfg.min_enclosed_dirs - 2))
    mask = remove_small_regions(mask, grid, cfg.min_area_km2)
    unit_ids = np.where(mask, units, 0).astype(np.int32)

    area = np.bincount(unit_ids.ravel(), weights=area_px.ravel(), minlength=top_lut.size)[ids]
    lat2d = np.broadcast_to(grid.lat[:, None], core.shape)
    lon2d = np.broadcast_to(grid.lon[None, :], core.shape)
    table = pd.DataFrame({
        "unit_id": ids,
        "center_lat": np.asarray(ndi.mean(lat2d, labels=markers, index=ids)),
        "center_lon": np.asarray(ndi.mean(lon2d, labels=markers, index=ids)),
        "valley_mean_m": valley_mean,
        "peak_p95_m": peak,
        "fog_thickness_m": thickness,
        "fog_top_m": fog_top,
        "fog_area_km2": area,
    })
    table = table[table["fog_area_km2"] > 0].reset_index(drop=True)
    return mask, unit_ids, table


def grid_transform(grid: DemGrid) -> Affine:
    dlon = float(grid.lon[1] - grid.lon[0])
    dlat = float(grid.lat[1] - grid.lat[0])
    return Affine(dlon, 0.0, float(grid.lon[0]) - dlon / 2, 0.0, dlat, float(grid.lat[0]) - dlat / 2)


def labels_to_polygons(labels: np.ndarray, grid: DemGrid, config_name: str) -> gpd.GeoDataFrame:
    """标号栅格 → 面矢量（EPSG:4326），每个连通山谷一个要素。"""
    transform = grid_transform(grid)
    records = [
        {"config": config_name, "valley_id": int(value), "geometry": shape(geom)}
        for geom, value in shapes(labels.astype(np.int32), mask=labels > 0, transform=transform)
    ]
    if not records:
        return gpd.GeoDataFrame(columns=["config", "valley_id", "geometry"], geometry="geometry", crs=4326)
    gdf = gpd.GeoDataFrame(records, crs=4326).dissolve(by="valley_id", as_index=False)
    return gdf


def hillshade(elevation: np.ndarray, grid: DemGrid, azimuth: float = 315, altitude: float = 45) -> np.ndarray:
    gy, gx = np.gradient(elevation, grid.dy_km * 1000, grid.dx_km * 1000)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.deg2rad(azimuth), np.deg2rad(altitude)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shade, 0, 1)


def _draw_base(ax, grid: DemGrid, shade: np.ndarray, boundary: gpd.GeoDataFrame | None, extent) -> None:
    extent_img = [grid.lon[0], grid.lon[-1], grid.lat[0], grid.lat[-1]]
    elev = np.where(grid.land, grid.elevation, np.nan)
    ax.imshow(shade, origin="lower", extent=extent_img, cmap="gray", vmin=0, vmax=1)
    ax.imshow(elev, origin="lower", extent=extent_img, cmap="terrain", vmin=-300, vmax=1600, alpha=0.35)
    if boundary is not None:
        boundary.boundary.plot(ax=ax, color="black", linewidth=0.8)
    for name, (lon, lat) in REFERENCE_CITIES.items():
        if not (extent[0] <= lon <= extent[1] and extent[2] <= lat <= extent[3]):
            continue
        ax.plot(lon, lat, "o", ms=2.5, color="red")
        ax.text(lon + 0.05, lat + 0.05, name, fontsize=7, color="darkred", clip_on=True)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(grid.lat.mean())))


def plot_method_panels(results: dict[str, np.ndarray], configs: list[ValleyConfig], grid: DemGrid,
                       boundary: gpd.GeoDataFrame | None, output: Path) -> None:
    shade = hillshade(grid.elevation, grid)
    if boundary is not None:
        b = boundary.total_bounds
        extent = (b[0] - 0.3, b[2] + 0.3, b[1] - 0.3, b[3] + 0.3)
    else:
        extent = (grid.lon[0], grid.lon[-1], grid.lat[0], grid.lat[-1])
    ncols = 2
    nrows = int(np.ceil(len(configs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5.6 * nrows), squeeze=False)
    extent_img = [grid.lon[0], grid.lon[-1], grid.lat[0], grid.lat[-1]]
    overlay_cmap = matplotlib.colors.ListedColormap(["#00a0ff"])
    for ax, cfg in zip(axes.ravel(), configs):
        _draw_base(ax, grid, shade, boundary, extent)
        mask = results[cfg.name]
        ax.imshow(np.where(mask, 1.0, np.nan), origin="lower", extent=extent_img,
                  cmap=overlay_cmap, alpha=0.75, interpolation="nearest")
        ax.set_title(f"{cfg.name}\n{cfg.label}", fontsize=10)
    for ax in axes.ravel()[len(configs):]:
        ax.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def _draw_labels(ax, labels: np.ndarray, grid: DemGrid) -> None:
    """按山谷编号随机着色叠加，便于检查碎片化程度和单块大小。"""
    rng = np.random.default_rng(0)
    lut = rng.random((int(labels.max()) + 1, 3)) * 0.8 + 0.1
    rgba = np.zeros(labels.shape + (4,))
    rgba[..., :3] = lut[labels]
    rgba[..., 3] = np.where(labels > 0, 0.8, 0.0)
    ax.imshow(rgba, origin="lower", extent=[grid.lon[0], grid.lon[-1], grid.lat[0], grid.lat[-1]],
              interpolation="nearest")


def plot_units(unit_ids: np.ndarray, cfg: ValleyConfig, grid: DemGrid,
               boundary: gpd.GeoDataFrame | None, output: Path) -> None:
    """山谷单元彩色分块图，便于检查单元切分是否合理。"""
    shade = hillshade(grid.elevation, grid)
    b = boundary.total_bounds if boundary is not None else (grid.lon[0], grid.lat[0], grid.lon[-1], grid.lat[-1])
    extent = (b[0] - 0.3, b[2] + 0.3, b[1] - 0.3, b[3] + 0.3)
    fig, ax = plt.subplots(figsize=(14, 9.5))
    _draw_base(ax, grid, shade, boundary, extent)
    _draw_labels(ax, unit_ids, grid)
    ax.set_title(f"{cfg.name} 山谷单元（颜色区分单元）\n{cfg.label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def plot_zoom_labels(labels: dict[str, np.ndarray], configs: list[ValleyConfig], grid: DemGrid,
                     boundary: gpd.GeoDataFrame | None, extent: tuple[float, float, float, float],
                     output: Path) -> None:
    """山区放大对比，每个山谷一种颜色。"""
    shade = hillshade(grid.elevation, grid)
    ncols = 2
    nrows = int(np.ceil(len(configs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 5 * nrows), squeeze=False)
    for ax, cfg in zip(axes.ravel(), configs):
        _draw_base(ax, grid, shade, boundary, extent)
        _draw_labels(ax, labels[cfg.name], grid)
        ax.set_title(f"{cfg.name}\n{cfg.label}", fontsize=10)
    for ax in axes.ravel()[len(configs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def guangdong_mask(grid: DemGrid, boundary: gpd.GeoDataFrame | None) -> np.ndarray:
    if boundary is None:
        return grid.land
    lon2d, lat2d = np.meshgrid(grid.lon, grid.lat)
    return contains_xy(boundary.geometry.union_all(), lon2d, lat2d) & grid.land


def run(dem_path: Path, boundary_path: Path | None, out_dir: Path,
        configs: tuple[ValleyConfig, ...] = DEFAULT_CONFIGS) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = load_dem(dem_path)
    boundary = None
    if boundary_path is not None and boundary_path.exists():
        boundary = gpd.read_file(boundary_path).to_crs(4326)
    gd = guangdong_mask(grid, boundary)
    area_px = np.broadcast_to(grid.pixel_area_km2, grid.land.shape)
    gd_area = float(area_px[gd].sum())

    masks: dict[str, np.ndarray] = {}
    all_labels: dict[str, np.ndarray] = {}
    data_vars: dict[str, tuple] = {
        "elevation": (("lat", "lon"), np.where(grid.land, grid.elevation, np.nan).astype(np.float32),
                      {"units": "m", "long_name": "smoothed elevation"}),
    }
    summary = []
    gpkg_path = out_dir / "valley_polygons.gpkg"
    gpkg_path.unlink(missing_ok=True)
    for cfg in configs:
        if cfg.method == "tpi":
            mask = tpi_valley_mask(grid, cfg)
            labels, _ = ndi.label(mask, structure=STRUCTURE_8)
        elif cfg.method == "fill":
            mask = fill_valley_mask(grid, cfg)
            labels, _ = ndi.label(mask, structure=STRUCTURE_8)
        elif cfg.method == "hybrid":
            mask, labels = hybrid_valley_mask(grid, cfg)
        elif cfg.method == "fusion":
            mask, labels = fusion_valley_mask(grid, cfg)
        elif cfg.method == "unit":
            mask, labels, table = unit_valley_mask(grid, cfg)
            table.to_csv(out_dir / f"units_{cfg.name}.csv", index=False, encoding="utf-8-sig")
            plot_units(labels, cfg, grid, boundary, out_dir / f"units_{cfg.name}.png")
        else:
            raise ValueError(f"未知算法: {cfg.method}")
        masks[cfg.name] = mask
        all_labels[cfg.name] = labels
        data_vars[f"valley_{cfg.name}"] = (("lat", "lon"), labels.astype(np.int32),
                                           {"long_name": cfg.label, "description": "0=非山谷, >0=山谷编号"})
        polygons = labels_to_polygons(labels, grid, cfg.name)
        if not polygons.empty:
            polygons.to_file(gpkg_path, layer=cfg.name, driver="GPKG")
        in_gd = mask & gd
        gd_labels = np.where(in_gd, labels, 0)
        valley_areas = np.bincount(gd_labels.ravel(), weights=area_px.ravel())[1:]
        valley_areas = valley_areas[valley_areas > 0]
        summary.append({
            "config": cfg.name,
            "method": cfg.method,
            "radius_km": cfg.radius_km,
            "threshold_m": cfg.threshold_m,
            "valley_count": int(valley_areas.size),
            "valley_area_median_km2": round(float(np.median(valley_areas)), 1) if valley_areas.size else np.nan,
            "valley_area_max_km2": round(float(valley_areas.max()), 1) if valley_areas.size else np.nan,
            "valley_area_km2_gd": round(float(area_px[in_gd].sum()), 1),
            "valley_fraction_gd": round(float(area_px[in_gd].sum()) / gd_area, 4) if gd_area else np.nan,
            "mean_elevation_m_gd": round(float(grid.elevation[in_gd].mean()), 1) if in_gd.any() else np.nan,
        })
        print(f"[OK] {cfg.name}: 广东境内山谷 {summary[-1]['valley_count']} 个，"
              f"面积 {summary[-1]['valley_area_km2_gd']} km2（{summary[-1]['valley_fraction_gd']:.1%}）")

    for group_name in dict.fromkeys(c.figure_group for c in configs):
        group = [c for c in configs if c.figure_group == group_name]
        plot_method_panels(masks, group, grid, boundary, out_dir / f"compare_{group_name}.png")
    by_name = {c.name: c for c in configs}
    zoom = [by_name[name] for name in ZOOM_CONFIGS if name in by_name]
    if zoom:
        plot_zoom_labels(all_labels, zoom, grid, boundary, ZOOM_EXTENT, out_dir / "compare_zoom_north.png")

    ds = xr.Dataset(data_vars, coords={"lat": grid.lat, "lon": grid.lon},
                    attrs={"title": "Valley boundaries for radiation fog correction",
                           "source_dem": str(dem_path), "crs": "EPSG:4326"})
    encoding = {name: {"zlib": True, "complevel": 4} for name in data_vars}
    ds.to_netcdf(out_dir / "valley_boundaries.nc", encoding=encoding)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] 输出目录: {out_dir.resolve()}")
    return summary_df


def _default_boundary() -> Path | None:
    candidates = sorted(Path("data/assets/gis/guangdong").glob("*.shp"))
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="基于 DEM 生成山谷边界（多算法多阈值对比）")
    parser.add_argument("--dem", type=Path, default=Path("data/assets/dem/merged_dem_data.nc"))
    parser.add_argument("--boundary", type=Path, default=_default_boundary(), help="广东边界 Shapefile")
    parser.add_argument("--out-dir", type=Path, default=Path("output/valley_boundary"))
    args = parser.parse_args()
    run(args.dem, args.boundary, args.out_dir)


if __name__ == "__main__":
    main()
