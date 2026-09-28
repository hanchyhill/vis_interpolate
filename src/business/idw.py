"""不依赖调试可视化的各向异性IDW实现。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from .radiation_fog import FogInfluence


def create_visibility_grid(
    stations: pd.DataFrame,
    dem: xr.Dataset,
    *,
    beta: float = 10.0,
    power: float = 2.0,
    n_neighbors: int = 6,
    fog_influence: FogInfluence | None = None,
) -> xr.DataArray:
    """fog_influence 非空时，stations 需含 fog_domain_id 列（0=非雾站），
    雾站对格点的权重乘以其影响域 g；为空时与原算法一致。fog_influence 须基于同一 DEM 网格。
    """
    required = ["lon", "lat", "vis", "altitude"]
    missing = [column for column in required if column not in stations]
    if missing:
        raise ValueError(f"IDW站点数据缺少字段: {', '.join(missing)}")
    valid = stations.dropna(subset=required).copy()
    if valid.empty:
        raise ValueError("没有可用于IDW插值的站点")
    if "elevation" not in dem:
        raise ValueError("DEM缺少 elevation 变量")

    lons = np.asarray(dem.lon.values)
    lats = np.asarray(dem.lat.values)
    elevations = np.asarray(dem["elevation"].values)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    weight_factor = None
    if fog_influence is not None and not fog_influence.empty:
        if "fog_domain_id" not in valid:
            raise ValueError("启用辐射雾订正时，IDW站点数据缺少 fog_domain_id 字段")
        domains = valid["fog_domain_id"].fillna(0).to_numpy(dtype=np.int64)

        def weight_factor(start: int, end: int, indices: np.ndarray) -> np.ndarray:
            return fog_influence.grid_factor(domains[indices], np.arange(start, end))

    result = _interpolate(
        valid, lon_grid.ravel(), lat_grid.ravel(), elevations.ravel(),
        beta=beta, power=power, n_neighbors=n_neighbors, weight_factor=weight_factor,
    ).reshape(lon_grid.shape)
    attrs = {}
    if weight_factor is not None:
        attrs["radiation_fog_correction"] = "fog station weights multiplied by influence-domain factor g"
    return xr.DataArray(
        result,
        coords={"lat": ("lat", lats), "lon": ("lon", lons)},
        dims=["lat", "lon"],
        name="visibility",
        attrs={
            "units": "m",
            "long_name": "Visibility",
            "interpolation_method": "Anisotropic IDW",
            "beta": beta,
            "power": power,
            "n_neighbors": n_neighbors,
            "description": "各向异性反距离权重插值得到的能见度，单位为米",
            **attrs,
        },
    )


def _interpolate(
    stations: pd.DataFrame,
    target_lons: np.ndarray,
    target_lats: np.ndarray,
    target_elevations: np.ndarray,
    *,
    beta: float,
    power: float,
    n_neighbors: int,
    weight_factor: Callable[[int, int, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """weight_factor(start, end, 候选站下标 (n, k)) 返回与原权重相乘的系数 (n, k)。"""
    station_lons = stations["lon"].to_numpy(float)
    station_lats = stations["lat"].to_numpy(float)
    station_vis = stations["vis"].to_numpy(float)
    station_alts = stations["altitude"].to_numpy(float)
    n_neighbors = min(n_neighbors, len(station_lons))
    search_count = min(max(n_neighbors * 2, n_neighbors), len(station_lons))
    finder = NearestNeighbors(n_neighbors=search_count, algorithm="ball_tree")
    finder.fit(np.column_stack([station_lons, station_lats]))
    results = np.full(target_lons.shape, np.nan, dtype=float)

    for start in range(0, len(target_lons), 10000):
        end = min(start + 10000, len(target_lons))
        target_coords = np.column_stack([target_lons[start:end], target_lats[start:end]])
        _, indices = finder.kneighbors(target_coords)
        candidate_lons, candidate_lats = station_lons[indices], station_lats[indices]
        candidate_alts, candidate_vis = station_alts[indices], station_vis[indices]
        target_lon = target_lons[start:end, None]
        target_lat = target_lats[start:end, None]
        target_alt = target_elevations[start:end, None]
        horizontal = _distance_km(target_lat, target_lon, candidate_lats, candidate_lons)
        vertical = np.abs(target_alt - candidate_alts) / 1000.0
        distance = np.sqrt(horizontal**2 + (beta * vertical) ** 2)
        finite = np.isfinite(distance) & (distance > 0) & np.isfinite(candidate_vis) & np.isfinite(target_alt)
        weights = np.where(finite, 1.0 / np.maximum(distance, 1e-12) ** power, 0.0)
        if weight_factor is not None:
            weights = weights * weight_factor(start, end, indices)
        weighted = np.sum(weights * np.where(np.isfinite(candidate_vis), candidate_vis, 0.0), axis=1)
        totals = np.sum(weights, axis=1)
        np.divide(weighted, totals, out=results[start:end], where=totals > 0)
    return results


def _distance_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    avg_lat = np.radians((lat1 + lat2) / 2.0)
    dx = 6371.0 * np.radians(lon2 - lon1) * np.cos(avg_lat)
    dy = 6371.0 * np.radians(lat2 - lat1)
    return np.sqrt(dx**2 + dy**2)
