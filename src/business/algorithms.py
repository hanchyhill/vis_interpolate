"""业务化能见度估算算法。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


OUTPUT_COLUMNS = [
    "code", "name", "city", "county", "lon", "lat", "altitude",
    "rh", "vis", "vis_rh", "vis_dis", "is_vis_est",
]
N_REFERENCE = 4
_BATCH_SIZE = 512

# (参考站下标 (n, k), 目标站行号 (n,)) -> 权重修正系数 g (n, k)，取值 [0, 1]。
WeightFactor = Callable[[np.ndarray, np.ndarray], np.ndarray]


def estimate_both(national: pd.DataFrame, regional: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """生成仅国家站参考和国家站+区域站参考两种估算结果。"""
    national_ref, combined_ref, regional_target = reference_sets(national, regional)
    return {
        "national": _estimate_one(national_ref, regional_target),
        "national_and_regional": _estimate_one(combined_ref, regional_target),
    }


def reference_sets(national: pd.DataFrame, regional: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """返回 (国家站参考, 国家站+区域站参考, 区域站估算目标)。"""
    national_ref = _usable(national, include_vis=True)
    regional_target = _usable(regional, include_vis=False)
    if national_ref.empty:
        raise ValueError("没有可用于能见度估算的国家站有效样本")
    if regional_target.empty:
        raise ValueError("没有可用于能见度估算的区域站有效湿度样本")
    combined_ref = pd.concat(
        [national_ref, _usable(regional, include_vis=True)], ignore_index=True, sort=False
    )
    return national_ref, combined_ref, regional_target


def _usable(frame: pd.DataFrame, *, include_vis: bool) -> pd.DataFrame:
    columns = ["code", "lon", "lat", "altitude", "rh"]
    if include_vis:
        columns.append("vis")
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = np.nan
    return result.dropna(subset=columns).copy()


def _estimate_one(
    reference: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    weight_factor: WeightFactor | None = None,
) -> pd.DataFrame:
    """每个目标站取最近 4 个参考站，0.5×湿度相似度加权 + 0.5×距离加权。

    weight_factor 为空时与 `_estimate_one_legacy` 结果一致；非空时在最近 8 站中
    按距离顺序取前 4 个 g > 0 的参考站，湿度权重和距离权重都乘以 g。
    """
    ref = reference.reset_index(drop=True)
    target_frame = targets.reset_index(drop=True)
    ref_coords = ref[["lat", "lon"]].to_numpy(dtype=float)
    ref_rh = ref["rh"].to_numpy(dtype=float)
    ref_vis = ref["vis"].to_numpy(dtype=float)
    target_coords = target_frame[["lat", "lon"]].to_numpy(dtype=float)
    target_rh = target_frame["rh"].to_numpy(dtype=float)
    k = min(N_REFERENCE, len(ref))
    search = min(2 * N_REFERENCE, len(ref)) if weight_factor is not None else k
    vis_rh = np.full(len(target_frame), np.nan)
    vis_dis = np.full(len(target_frame), np.nan)

    for start in range(0, len(target_frame), _BATCH_SIZE):
        rows = np.arange(start, min(start + _BATCH_SIZE, len(target_frame)))
        distances = cdist(target_coords[rows], ref_coords, metric="euclidean")
        nearest = np.argsort(distances, axis=1)[:, :search]
        nearest_distances = np.take_along_axis(distances, nearest, axis=1)
        if weight_factor is None:
            factor = np.ones(nearest.shape)
        else:
            factor = np.asarray(weight_factor(nearest, rows), dtype=float)
            order = np.argsort(factor <= 0, axis=1, kind="stable")[:, :k]
            nearest = np.take_along_axis(nearest, order, axis=1)
            nearest_distances = np.take_along_axis(nearest_distances, order, axis=1)
            factor = np.take_along_axis(factor, order, axis=1)

        vis = ref_vis[nearest]
        rh_diff = np.maximum(np.abs(target_rh[rows, None] - ref_rh[nearest]), 0.1)
        rh_weights = factor / (rh_diff**2)
        valid = np.isfinite(vis)
        distance_weights = np.where(valid, factor / (np.maximum(nearest_distances, 0.001) ** 2), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            vis_rh[rows] = np.sum(rh_weights * vis, axis=1) / np.sum(rh_weights, axis=1)
            vis_dis[rows] = (np.sum(distance_weights * np.where(valid, vis, 0.0), axis=1)
                             / np.sum(distance_weights, axis=1))

    final = np.where(
        np.isnan(vis_rh), vis_dis, np.where(np.isnan(vis_dis), vis_rh, 0.5 * vis_rh + 0.5 * vis_dis)
    )
    estimated = target_frame.copy()
    estimated["vis_rh"], estimated["vis_dis"], estimated["vis"], estimated["is_vis_est"] = vis_rh, vis_dis, final, 1
    return _combine_observed(ref, estimated)


def _estimate_one_legacy(reference: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """原始逐行实现，保留用于新旧算法对比和回归测试。"""
    ref = reference.reset_index(drop=True)
    ref_coords = ref[["lat", "lon"]].to_numpy(dtype=float)
    output_rows: list[pd.Series] = []
    for _, target in targets.iterrows():
        coords = np.asarray([[target["lat"], target["lon"]]], dtype=float)
        distances = cdist(coords, ref_coords, metric="euclidean")[0]
        nearest = np.argsort(distances)[: min(4, len(ref))]
        nearest_data = ref.iloc[nearest]
        nearest_distances = distances[nearest]

        rh_diff = np.maximum(np.abs(float(target["rh"]) - nearest_data["rh"].to_numpy()), 0.1)
        rh_weights = 1.0 / (rh_diff**2)
        vis_rh = float(np.sum(rh_weights * nearest_data["vis"].to_numpy()) / np.sum(rh_weights))

        valid = nearest_data["vis"].notna().to_numpy()
        valid_distances = np.maximum(nearest_distances[valid], 0.001)
        valid_vis = nearest_data.loc[valid, "vis"].to_numpy(dtype=float)
        if len(valid_vis):
            distance_weights = 1.0 / (valid_distances**2)
            vis_dis = float(np.sum(distance_weights * valid_vis) / np.sum(distance_weights))
        else:
            vis_dis = np.nan

        if np.isnan(vis_rh) and np.isnan(vis_dis):
            final = np.nan
        elif np.isnan(vis_rh):
            final = vis_dis
        elif np.isnan(vis_dis):
            final = vis_rh
        else:
            final = 0.5 * vis_rh + 0.5 * vis_dis
        row = target.copy()
        row["vis_rh"], row["vis_dis"], row["vis"], row["is_vis_est"] = vis_rh, vis_dis, final, 1
        output_rows.append(row)

    return _combine_observed(ref, pd.DataFrame(output_rows))


def _combine_observed(ref: pd.DataFrame, estimated: pd.DataFrame) -> pd.DataFrame:
    observed = ref.copy()
    observed["is_vis_est"] = 0
    observed["vis_rh"] = np.nan
    observed["vis_dis"] = np.nan
    for frame in (observed, estimated):
        for column in OUTPUT_COLUMNS:
            if column not in frame:
                frame[column] = np.nan
    combined = pd.concat([observed[OUTPUT_COLUMNS], estimated[OUTPUT_COLUMNS]], ignore_index=True)
    return combined.drop_duplicates("code", keep="first").reset_index(drop=True)
