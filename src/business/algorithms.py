"""业务化能见度估算算法。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


OUTPUT_COLUMNS = [
    "code", "name", "city", "county", "lon", "lat", "altitude",
    "rh", "vis", "vis_rh", "vis_dis", "is_vis_est",
]


def estimate_both(national: pd.DataFrame, regional: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """生成仅国家站参考和国家站+区域站参考两种估算结果。"""
    national_ref = _usable(national, include_vis=True)
    regional_target = _usable(regional, include_vis=False)
    if national_ref.empty:
        raise ValueError("没有可用于能见度估算的国家站有效样本")
    if regional_target.empty:
        raise ValueError("没有可用于能见度估算的区域站有效湿度样本")

    combined_ref = pd.concat(
        [national_ref, _usable(regional, include_vis=True)], ignore_index=True, sort=False
    )
    return {
        "national": _estimate_one(national_ref, regional_target),
        "national_and_regional": _estimate_one(combined_ref, regional_target),
    }


def _usable(frame: pd.DataFrame, *, include_vis: bool) -> pd.DataFrame:
    columns = ["code", "lon", "lat", "altitude", "rh"]
    if include_vis:
        columns.append("vis")
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = np.nan
    return result.dropna(subset=columns).copy()


def _estimate_one(reference: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
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

    estimated = pd.DataFrame(output_rows)
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
