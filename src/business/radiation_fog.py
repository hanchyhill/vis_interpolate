"""山谷辐射雾订正：雾站判别、影响域与权重修正系数 g。

坐标系 WGS84 经纬度；海拔 m；水平距离 km。

判定为辐射雾的站（雾站）只在其影响域内起作用：影响域 = 所在山谷中、
谷内最近站划分（Voronoi）下离雾站比离非雾站更近的部分。对任一目标点：

    g = exp(-Δz / σz) · exp(-d_out / σd)，g < gCutoff 时取 0

Δz 为目标点高出雾顶的米数（≤0 取 0），d_out 为目标点到影响域的距离（域内为 0），
所以影响域内、不高于雾顶处 g = 1。估算和 IDW 两个环节中，雾站的权重都乘以 g。
无雾站时 g 恒为 1，结果与原算法逐位一致。
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .algorithms import _estimate_one, reference_sets
from .config import RadiationFogSettings
from .idw import _distance_km, _interpolate, create_visibility_grid

FOG_NONE = 0
FOG_OBSERVED = 1
FOG_INFERRED = 2
EARTH_KM_PER_DEG = 111.32
FOG_COLUMNS = ["valley_id", "is_radiation_fog", "fog_domain_id", "pre_1h"]
# 对数混合时的能见度下限 (m)，避免 0 m 观测使整片过渡带变为 0。
_MIN_BLEND_VIS_M = 10.0
_MIN_DEPTH_RANGE_M = 10.0   # 雾顶与谷底高差的下限，避免浅谷中相对深度被放大
_MIN_SEED_DEPTH = 0.1       # 雾站相对深度下限，避免雾站贴近雾顶时指数失控
_MAX_DEPTH_EXPONENT = 2.0   # 谷底相对雾站最多再降低一个同样的对数幅度


@dataclass
class ValleyProduct:
    """山谷产品重采样到 IDW DEM 网格后的结果。"""

    valley_id: np.ndarray  # (lat, lon)，0=非山谷
    elevation: np.ndarray  # IDW 使用的 DEM 海拔 (m)
    lat: np.ndarray
    lon: np.ndarray
    fog_top_m: np.ndarray  # 按山谷编号索引的雾顶海拔，非山谷为 NaN
    dy_km: float
    dx_km: float

    @classmethod
    def load(cls, path: Path, dem: xr.Dataset) -> "ValleyProduct":
        with xr.open_dataset(path) as source:
            ds = source.load()
        lat = np.asarray(dem.lat.values, dtype=float)
        lon = np.asarray(dem.lon.values, dtype=float)
        tolerance = float(np.abs(np.diff(ds.lat.values)).mean()) * 0.6
        ids = ds["valley_id"].reindex(lat=lat, lon=lon, method="nearest", tolerance=tolerance)
        valley_id = np.nan_to_num(ids.values, nan=0).astype(np.int32)
        valleys = ds["valley"].values.astype(np.int64)
        size = int(max(valleys.max(initial=0), valley_id.max(initial=0))) + 1
        fog_top = np.full(size, np.nan)
        fog_top[valleys] = ds["fog_top_m"].values
        dy_km = float(np.abs(np.diff(lat)).mean()) * EARTH_KM_PER_DEG
        dx_km = float(np.abs(np.diff(lon)).mean()) * EARTH_KM_PER_DEG * float(np.cos(np.deg2rad(lat.mean())))
        return cls(valley_id, np.asarray(dem["elevation"].values, dtype=float), lat, lon, fog_top, dy_km, dx_km)

    def cell_index(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """站点最近格点的 (行, 列, 是否在网格内)。"""
        iy, in_y = _nearest_index(self.lat, np.asarray(lat, dtype=float))
        ix, in_x = _nearest_index(self.lon, np.asarray(lon, dtype=float))
        return iy, ix, in_y & in_x

    def station_valley(self, lon: np.ndarray, lat: np.ndarray, altitude: np.ndarray) -> np.ndarray:
        """站点所在山谷编号；站点海拔高于雾顶（山顶站）或缺测时视为非山谷 (0)。"""
        iy, ix, inside = self.cell_index(lon, lat)
        valley = np.where(inside, self.valley_id[iy, ix], 0)
        top = self.fog_top_m[valley]
        with np.errstate(invalid="ignore"):
            below_top = np.asarray(altitude, dtype=float) <= top
        return np.where((valley > 0) & below_top, valley, 0).astype(np.int32)


@dataclass
class FogDomain:
    domain_id: int
    fog_top_m: float
    y0: int
    x0: int
    distance_km: np.ndarray  # 窗口内到影响域的距离，域内为 0
    g: np.ndarray            # 窗口内格点的 g（按 DEM 海拔计算 Δz）

    def window_values(self, values: np.ndarray, iy: np.ndarray, ix: np.ndarray, fill: float) -> np.ndarray:
        wy, wx = iy - self.y0, ix - self.x0
        ny, nx = values.shape
        inside = (wy >= 0) & (wy < ny) & (wx >= 0) & (wx < nx)
        result = np.full(iy.shape, fill, dtype=float)
        result[inside] = values[wy[inside], wx[inside]]
        return result


class FogInfluence:
    """各影响域的 g 场，窗口外 g=0。"""

    def __init__(self, product: ValleyProduct, domains: dict[int, FogDomain], settings: RadiationFogSettings):
        self.product = product
        self.domains = domains
        self.settings = settings

    @property
    def empty(self) -> bool:
        return not self.domains

    def decay(self, dz_m: np.ndarray, distance_km: np.ndarray) -> np.ndarray:
        dz = np.clip(np.nan_to_num(np.asarray(dz_m, dtype=float), nan=0.0), 0.0, None)
        sigma_z = max(self.settings.sigma_z_m, 1e-6)
        sigma_d = max(self.settings.sigma_d_km, 1e-6)
        g = np.exp(-dz / sigma_z) * np.exp(-np.asarray(distance_km, dtype=float) / sigma_d)
        return np.where(g < self.settings.g_cutoff, 0.0, g)

    def grid_factor(self, domain_ids: np.ndarray, flat_index: np.ndarray) -> np.ndarray:
        """格点对候选站的 g：domain_ids (n, k)，flat_index 为格点在 (lat, lon) 展平后的序号 (n,)。"""
        factor = np.ones(domain_ids.shape, dtype=float)
        rows, cols = np.nonzero(domain_ids > 0)
        if rows.size == 0:
            return factor
        ids = domain_ids[rows, cols]
        iy, ix = np.divmod(np.asarray(flat_index)[rows], self.product.valley_id.shape[1])
        for domain_id in np.unique(ids):
            domain = self.domains.get(int(domain_id))
            if domain is None:
                continue
            sel = ids == domain_id
            factor[rows[sel], cols[sel]] = domain.window_values(domain.g, iy[sel], ix[sel], 0.0)
        return factor

    def station_factor(self, domain_ids: np.ndarray, lon: np.ndarray, lat: np.ndarray,
                       altitude: np.ndarray) -> np.ndarray:
        """站点对候选参考站的 g，Δz 用站点海拔：domain_ids (n, k)，其余为 (n,)。"""
        factor = np.ones(domain_ids.shape, dtype=float)
        iy, ix, inside = self.product.cell_index(lon, lat)
        altitude = np.asarray(altitude, dtype=float)
        for domain_id in np.unique(domain_ids[domain_ids > 0]):
            domain = self.domains.get(int(domain_id))
            if domain is None:
                continue
            rows, cols = np.nonzero(domain_ids == domain_id)
            distance = domain.window_values(domain.distance_km, iy[rows], ix[rows], np.inf)
            distance[~inside[rows]] = np.inf
            factor[rows, cols] = self.decay(altitude[rows] - domain.fog_top_m, distance)
        return factor

    def in_domain(self, domain_ids: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """站点最近格点是否落在 domain_ids 对应的影响域内。"""
        result = np.zeros(np.shape(domain_ids), dtype=bool)
        iy, ix, inside = self.product.cell_index(lon, lat)
        for domain_id in np.unique(domain_ids[domain_ids > 0]):
            domain = self.domains.get(int(domain_id))
            if domain is None:
                continue
            rows = np.flatnonzero((domain_ids == domain_id) & inside)
            result[rows] = domain.window_values(domain.distance_km, iy[rows], ix[rows], np.inf) == 0
        return result

    def max_field(self, *, fill_domain: bool = False) -> np.ndarray:
        """各格点受雾站影响的 g 最大值；fill_domain=True 时影响域内取 1（blend 模式）。"""
        field_ = np.zeros(self.product.valley_id.shape, dtype=np.float32)
        for domain in self.domains.values():
            ny, nx = domain.g.shape
            g = np.where(domain.distance_km == 0, 1.0, domain.g) if fill_domain else domain.g
            window = field_[domain.y0:domain.y0 + ny, domain.x0:domain.x0 + nx]
            np.maximum(window, g.astype(np.float32), out=window)
        return field_


@dataclass
class FogAnalysis:
    reference: pd.DataFrame  # 追加 valley_id, is_radiation_fog, fog_domain_id, _wet
    targets: pd.DataFrame    # 追加 valley_id, is_radiation_fog, fog_domain_id, _wet, virtual_vis
    influence: FogInfluence
    summary: dict[str, int] = field(default_factory=dict)


@dataclass
class CorrectedEstimate:
    frame: pd.DataFrame
    influence: FogInfluence
    summary: dict[str, int]


class RadiationFogModel:
    def __init__(self, product: ValleyProduct, settings: RadiationFogSettings):
        self.product = product
        self.settings = settings

    @classmethod
    def load(cls, settings: RadiationFogSettings, dem: xr.Dataset) -> "RadiationFogModel":
        return cls(ValleyProduct.load(settings.valley_path, dem), settings)

    def analyze(self, reference: pd.DataFrame, targets: pd.DataFrame) -> FogAnalysis:
        """reference 为有能见度观测的参考站，targets 为待估算的湿度站。"""
        s = self.settings
        ref = reference.reset_index(drop=True).copy()
        tgt = targets.reset_index(drop=True).copy()
        ref_valley = self._valley(ref)
        tgt_valley = self._valley(tgt)
        ref_wet, ref_pre_missing = self._wet(ref)
        tgt_wet, _ = self._wet(tgt)

        ref_vis = ref["vis"].to_numpy(dtype=float)
        low_in_valley = (ref_vis < s.vis_threshold_m) & (ref_valley > 0)
        ref_fog = low_in_valley & ~ref_wet
        observed_valleys = np.unique(ref_valley[ref_valley > 0])

        tgt_rh = tgt["rh"].to_numpy(dtype=float)
        candidate = ((tgt_valley > 0) & ~np.isin(tgt_valley, observed_valleys)
                     & (tgt_rh >= s.rh_threshold_pct) & ~tgt_wet)
        virtual_vis = np.full(len(tgt), np.nan)
        if candidate.any() and ref_fog.any():
            rows = np.flatnonzero(candidate)
            distance = _distance_km(
                tgt["lat"].to_numpy(float)[rows, None], tgt["lon"].to_numpy(float)[rows, None],
                ref["lat"].to_numpy(float)[None, ref_fog], ref["lon"].to_numpy(float)[None, ref_fog],
            )
            near = distance <= s.infer_radius_km
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(np.where(near, ref_vis[None, ref_fog], np.nan), axis=1)
            virtual_vis[rows] = median
        tgt_virtual = np.isfinite(virtual_vis)

        ref_domain = np.zeros(len(ref), dtype=np.int32)
        tgt_domain = np.zeros(len(tgt), dtype=np.int32)
        domains: dict[int, FogDomain] = {}
        fog_valleys = np.union1d(ref_valley[ref_fog], tgt_valley[tgt_virtual]).astype(int)
        for valley in fog_valleys:
            if valley in observed_valleys:
                seeds, seed_fog = ref[ref_valley == valley], ref_fog[ref_valley == valley]
            else:
                seeds, seed_fog = tgt[tgt_valley == valley], tgt_virtual[tgt_valley == valley]
            domain = self._build_domain(int(valley), seeds, seed_fog)
            if domain is not None:
                domains[int(valley)] = domain
        ref_domain[ref_fog] = ref_valley[ref_fog]
        tgt_domain[tgt_virtual] = tgt_valley[tgt_virtual]

        ref["valley_id"], ref["is_radiation_fog"] = ref_valley, np.where(ref_fog, FOG_OBSERVED, FOG_NONE)
        ref["fog_domain_id"], ref["_wet"] = ref_domain, ref_wet
        tgt["valley_id"], tgt["is_radiation_fog"] = tgt_valley, np.where(tgt_virtual, FOG_INFERRED, FOG_NONE)
        tgt["fog_domain_id"], tgt["_wet"], tgt["virtual_vis"] = tgt_domain, tgt_wet, virtual_vis
        summary = {
            "fog_observed": int(ref_fog.sum()),
            "fog_inferred": int(tgt_virtual.sum()),
            "fog_domains": len(domains),
            "low_vis_in_valley_wet": int((low_in_valley & ref_wet).sum()),
            "low_vis_in_valley_precip_missing": int((low_in_valley & ref_pre_missing).sum()),
        }
        return FogAnalysis(ref, tgt, FogInfluence(self.product, domains, s), summary)

    def estimate(self, reference: pd.DataFrame, targets: pd.DataFrame) -> CorrectedEstimate:
        """订正后的估算结果，列同原算法并追加 FOG_COLUMNS。"""
        analysis = self.analyze(reference, targets)
        influence = analysis.influence
        ref, tgt = analysis.reference, analysis.targets
        if influence.empty:
            frame = _estimate_one(ref, tgt)
        else:
            ref_domain = ref["fog_domain_id"].to_numpy()
            t_lon, t_lat = tgt["lon"].to_numpy(float), tgt["lat"].to_numpy(float)
            t_alt = tgt["altitude"].to_numpy(float)

            def factor(nearest: np.ndarray, rows: np.ndarray) -> np.ndarray:
                return influence.station_factor(ref_domain[nearest], t_lon[rows], t_lat[rows], t_alt[rows])

            frame = _estimate_one(ref, tgt, weight_factor=factor)

        columns = ["code", "valley_id", "is_radiation_fog", "fog_domain_id", "pre_1h", "_wet", "virtual_vis"]
        annotations = pd.concat(
            [frame_.reindex(columns=columns) for frame_ in (ref, tgt)], ignore_index=True
        ).drop_duplicates("code", keep="first")
        frame = frame.merge(annotations, on="code", how="left")
        estimated = frame["is_vis_est"].to_numpy() == 1
        virtual = estimated & (frame["is_radiation_fog"].to_numpy() == FOG_INFERRED)
        frame.loc[virtual, "vis"] = frame.loc[virtual, "virtual_vis"]

        # 雾谷内估算出的低能见度站同样只在影响域内起作用，避免在 IDW 中再次向外扩散。
        members = np.zeros(len(frame), dtype=bool)
        if not influence.empty:
            valley = frame["valley_id"].fillna(0).to_numpy(dtype=np.int64)
            candidate = (estimated & ~virtual & (valley > 0)
                         & (frame["vis"].to_numpy(float) < self.settings.vis_threshold_m)
                         & ~frame["_wet"].eq(True).to_numpy())
            domain_ids = np.where(candidate, valley, 0)
            members = candidate & influence.in_domain(
                domain_ids, frame["lon"].to_numpy(float), frame["lat"].to_numpy(float)
            )
            domain = frame["fog_domain_id"].fillna(0).to_numpy(dtype=np.int32)
            domain[members] = valley[members]
            frame["fog_domain_id"] = domain
        for column in ("valley_id", "is_radiation_fog", "fog_domain_id"):
            frame[column] = frame[column].fillna(0).astype(np.int32)
        frame = frame.drop(columns=["_wet", "virtual_vis"])
        summary = {**analysis.summary, "fog_domain_estimated_members": int(members.sum())}
        return CorrectedEstimate(frame, influence, summary)

    def _valley(self, frame: pd.DataFrame) -> np.ndarray:
        return self.product.station_valley(
            frame["lon"].to_numpy(float), frame["lat"].to_numpy(float), frame["altitude"].to_numpy(float)
        )

    def _wet(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(有降水, 降水缺测)；降水缺测按 precipMissingAsDry 处理。"""
        precip = (frame["pre_1h"] if "pre_1h" in frame else pd.Series(np.nan, index=frame.index))
        precip = pd.to_numeric(precip, errors="coerce").to_numpy(dtype=float)
        missing = ~np.isfinite(precip)
        with np.errstate(invalid="ignore"):
            wet = precip > self.settings.precip_threshold_mm
        if not self.settings.precip_missing_as_dry:
            wet |= missing
        return wet, missing

    def _build_domain(self, valley: int, seeds: pd.DataFrame, seed_fog: np.ndarray) -> FogDomain | None:
        p = self.product
        ys, xs = np.nonzero(p.valley_id == valley)
        if ys.size == 0:
            return None
        if not seed_fog.all():
            sy, sx, _ = p.cell_index(seeds["lon"].to_numpy(float), seeds["lat"].to_numpy(float))
            tree = cKDTree(np.column_stack([sy * p.dy_km, sx * p.dx_km]))
            _, nearest = tree.query(np.column_stack([ys * p.dy_km, xs * p.dx_km]))
            keep = seed_fog[nearest]
            ys, xs = ys[keep], xs[keep]
            if ys.size == 0:
                return None
        cutoff = min(max(self.settings.g_cutoff, 1e-6), 1.0)
        reach_km = self.settings.sigma_d_km * np.log(1.0 / cutoff)
        margin = int(np.ceil(reach_km / min(p.dy_km, p.dx_km))) + 1
        ny, nx = p.valley_id.shape
        y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, ny)
        x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, nx)
        outside = np.ones((y1 - y0, x1 - x0), dtype=bool)
        outside[ys - y0, xs - x0] = False
        distance = ndi.distance_transform_edt(outside, sampling=(p.dy_km, p.dx_km))
        fog_top = float(p.fog_top_m[valley])
        influence = FogInfluence(p, {}, self.settings)
        g = influence.decay(p.elevation[y0:y1, x0:x1] - fog_top, distance)
        return FogDomain(valley, fog_top, int(y0), int(x0), distance, g)


def estimate_both_corrected(
    national: pd.DataFrame, regional: pd.DataFrame, model: RadiationFogModel,
    sources: tuple[str, ...] = ("national", "national_and_regional"),
) -> dict[str, CorrectedEstimate]:
    """与 estimate_both 相同的两条参考路径，分别做辐射雾判别与订正。"""
    national_ref, combined_ref, regional_target = reference_sets(national, regional)
    references = {"national": national_ref, "national_and_regional": combined_ref}
    return {source: model.estimate(references[source], regional_target) for source in sources}


def _seed_idw(seeds: pd.DataFrame, values: np.ndarray, lons: np.ndarray, lats: np.ndarray,
              elevation: np.ndarray, *, beta: float, power: float) -> np.ndarray:
    """用 seeds 的位置对 values 做各向异性 IDW；与站点重合的格点取最近站的值。"""
    frame = seeds.assign(vis=values)
    result = _interpolate(frame, lons, lats, elevation, beta=beta, power=power, n_neighbors=len(frame))
    coincident = ~np.isfinite(result) & np.isfinite(elevation)
    if coincident.any():
        distance = _distance_km(lats[coincident, None], lons[coincident, None],
                                frame["lat"].to_numpy(float)[None, :], frame["lon"].to_numpy(float)[None, :])
        result[coincident] = np.asarray(values, dtype=float)[np.argmin(distance, axis=1)]
    return result


def _depth_profile(domain: FogDomain, seeds: pd.DataFrame, seed_z: np.ndarray, fog_vis: np.ndarray,
                   rows: np.ndarray, cols: np.ndarray, lons: np.ndarray, lats: np.ndarray,
                   elevation: np.ndarray, edge_vis_m: float, *, beta: float, power: float) -> np.ndarray:
    """按相对深度调整雾区能见度（blend_depth）。

    相对深度 r = (雾顶 − z) / (雾顶 − 谷底)，谷底取域内 DEM 海拔的 5% 分位数，截断到 [0, 1]；
    雾站深度 r_s 用站点所在格点的 DEM 海拔 seed_z 计算（站点海拔与 0.01° DEM 可相差上百米，
    混用会使雾站显得偏浅），再插值为 r̄。对数空间沿深度线性变化：
    V = V_edge · (V_fog / V_edge)^(r / r̄)，r = r̄ 时等于雾站插值，r = 0（雾顶/域外）时为
    V_edge = max(edge_vis_m, V_fog)，更深处继续降低，指数上限 _MAX_DEPTH_EXPONENT。
    """
    window = elevation[domain.y0:domain.y0 + domain.distance_km.shape[0],
                       domain.x0:domain.x0 + domain.distance_km.shape[1]]
    inside_window = (domain.distance_km == 0) & np.isfinite(window)
    floor = float(np.nanpercentile(window[inside_window], 5)) if inside_window.any() else domain.fog_top_m
    height = max(domain.fog_top_m - floor, _MIN_DEPTH_RANGE_M)
    seed_depth = np.clip((domain.fog_top_m - seed_z) / height, 0.0, 1.0)
    mean_depth = _seed_idw(seeds, seed_depth, lons, lats, elevation[rows, cols], beta=beta, power=power)
    mean_depth = np.maximum(np.nan_to_num(mean_depth, nan=1.0), _MIN_SEED_DEPTH)
    inside = domain.distance_km[rows - domain.y0, cols - domain.x0] == 0
    depth = np.where(inside, np.clip((domain.fog_top_m - elevation[rows, cols]) / height, 0.0, 1.0), 0.0)
    edge = np.maximum(edge_vis_m, fog_vis)
    exponent = np.minimum(depth / mean_depth, _MAX_DEPTH_EXPONENT)
    return edge * (np.maximum(fog_vis, _MIN_BLEND_VIS_M) / edge) ** exponent


def blend_fog_grid(stations: pd.DataFrame, dem: xr.Dataset, influence: FogInfluence,
                   *, beta: float = 10.0, power: float = 2.0, linear: bool = False,
                   depth: bool = False) -> xr.DataArray:
    """域内填充：默认在对数空间混合，V = V_fog^g · V_base^(1 − g)。

    V_base 为去掉全部雾域站（fog_domain_id > 0）后的原 IDW；V_fog 只用本影响域内的雾站
    （实测与虚拟）做各向异性 IDW。影响域内一律取 g = 1（不做垂直衰减），使雾区边缘与
    山谷边界一致；域外沿用 g 的衰减。格点受多个影响域覆盖时取 g 最大的影响域。
    linear=True 时为线性混合 V = g·V_fog + (1 − g)·V_base，且域内保留垂直衰减。
    depth=True 时 V_fog 再按相对深度调整（谷底最低、向雾顶升到 visThresholdM），见 _depth_profile。
    """
    domain_ids = stations["fog_domain_id"].fillna(0).to_numpy(dtype=np.int64)
    base = create_visibility_grid(stations[domain_ids == 0], dem, beta=beta, power=power)
    result = base.values.copy()
    best = np.zeros(result.shape)
    lons = np.asarray(dem.lon.values, dtype=float)
    lats = np.asarray(dem.lat.values, dtype=float)
    elevation = np.asarray(dem["elevation"].values, dtype=float)
    fog = stations["is_radiation_fog"].fillna(0).to_numpy() > 0
    for domain_id, domain in influence.domains.items():
        seeds = stations[(domain_ids == domain_id) & fog].dropna(subset=["lon", "lat", "vis", "altitude"])
        if seeds.empty:
            continue
        g_fill = domain.g if linear else np.where(domain.distance_km == 0, 1.0, domain.g)
        rows, cols = np.nonzero(g_fill > 0)
        gy, gx = rows + domain.y0, cols + domain.x0
        g = g_fill[rows, cols]
        replace = g > best[gy, gx]
        gy, gx, g = gy[replace], gx[replace], g[replace]
        if gy.size == 0:
            continue
        fog_vis = _seed_idw(seeds, seeds["vis"].to_numpy(float), lons[gx], lats[gy], elevation[gy, gx],
                            beta=beta, power=power)
        if depth:
            sy, sx, _ = influence.product.cell_index(seeds["lon"].to_numpy(float), seeds["lat"].to_numpy(float))
            seed_z = elevation[sy, sx]
            seed_z = np.where(np.isfinite(seed_z), seed_z, seeds["altitude"].to_numpy(float))
            fog_vis = _depth_profile(domain, seeds, seed_z, fog_vis, gy, gx, lons[gx], lats[gy], elevation,
                                     influence.settings.vis_threshold_m, beta=beta, power=power)
        valid = np.isfinite(fog_vis) & np.isfinite(result[gy, gx])
        gy, gx, g, fog_vis = gy[valid], gx[valid], g[valid], fog_vis[valid]
        if linear:
            result[gy, gx] = g * fog_vis + (1.0 - g) * base.values[gy, gx]
        else:
            log_fog = np.log(np.maximum(fog_vis, _MIN_BLEND_VIS_M))
            log_base = np.log(np.maximum(base.values[gy, gx], _MIN_BLEND_VIS_M))
            result[gy, gx] = np.exp(g * log_fog + (1.0 - g) * log_base)
        best[gy, gx] = g
    if linear:
        method = "blend_linear: g*V_fog + (1-g)*V_base"
    elif depth:
        method = "blend_depth: V_fog scaled by relative depth below fog top, then V_fog^g * V_base^(1-g)"
    else:
        method = "blend: V_fog^g * V_base^(1-g), g=1 inside domain"
    attrs = {**base.attrs, "radiation_fog_correction": method}
    return xr.DataArray(result, coords=base.coords, dims=base.dims, name="visibility", attrs=attrs)


@dataclass
class CorrectedProduct:
    frame: pd.DataFrame   # 订正后站点，追加 FOG_COLUMNS 与 vis_original
    dataset: xr.Dataset   # visibility（订正后）、visibility_original、fog_influence
    summary: dict[str, int]


def build_corrected_products(
    national: pd.DataFrame,
    regional: pd.DataFrame,
    dem: xr.Dataset,
    model: RadiationFogModel,
    original_estimates: dict[str, pd.DataFrame],
    original_grids: dict[str, xr.DataArray],
) -> dict[str, CorrectedProduct]:
    """在原算法结果基础上生成订正产品；某路径无雾站时订正结果直接复用原算法格点。"""
    products: dict[str, CorrectedProduct] = {}
    for source, corrected in estimate_both_corrected(national, regional, model, sources=tuple(original_estimates)).items():
        original = original_estimates[source][["code", "vis"]].rename(columns={"vis": "vis_original"})
        frame = corrected.frame.merge(original, on="code", how="left")
        grid = original_grids[source]
        if not corrected.influence.empty:
            if model.settings.fill_mode == "weight":
                grid = create_visibility_grid(frame, dem, fog_influence=corrected.influence)
            else:
                grid = blend_fog_grid(frame, dem, corrected.influence,
                                      linear=model.settings.fill_mode == "blend_linear",
                                      depth=model.settings.fill_mode == "blend_depth")
        influence = corrected.influence.max_field(fill_domain=model.settings.fill_mode in ("blend", "blend_depth"))
        dataset = xr.Dataset(
            {
                "visibility": grid.rename("visibility"),
                "visibility_original": original_grids[source].rename("visibility_original"),
                "fog_influence": (("lat", "lon"), influence, {
                    "units": "1",
                    "long_name": "各格点受辐射雾站影响的权重修正系数 g 的最大值",
                }),
            },
            attrs={"radiation_fog_correction": 1, "fill_mode": model.settings.fill_mode,
                   "valley_path": str(model.settings.valley_path),
                   **{f"fog_{key}": value for key, value in corrected.summary.items()}},
        )
        dataset["visibility_original"].attrs["description"] = "原算法（未做辐射雾订正）能见度，单位为米"
        products[source] = CorrectedProduct(frame, dataset, corrected.summary)
    return products


def load_model(settings: RadiationFogSettings, dem: xr.Dataset,
               logger: logging.Logger | None = None) -> RadiationFogModel | None:
    """未启用或山谷产品缺失时返回 None（退回原算法）。"""
    if not settings.enabled:
        return None
    if not Path(settings.valley_path).exists():
        if logger is not None:
            logger.warning("辐射雾订正已启用但山谷产品不存在，退回原算法: %s", settings.valley_path)
        return None
    return RadiationFogModel.load(settings, dem)


def _nearest_index(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(axis)
    sorted_axis = axis[order]
    half = float(np.abs(np.diff(sorted_axis)).mean()) / 2 if sorted_axis.size > 1 else 0.0
    finite = np.isfinite(values)
    safe = np.where(finite, values, sorted_axis[0])
    pos = np.clip(np.searchsorted(sorted_axis, safe), 1, max(sorted_axis.size - 1, 1))
    if sorted_axis.size == 1:
        pos = np.zeros_like(pos)
    else:
        left, right = sorted_axis[pos - 1], sorted_axis[pos]
        pos = pos - ((safe - left) <= (right - safe))
    inside = finite & (safe >= sorted_axis[0] - half) & (safe <= sorted_axis[-1] + half)
    return order[pos], inside
