"""辐射雾订正个例对比：同一时次同时运行原算法与订正算法，输出对比图和统计。

数据来源（优先级从高到低）：
1. --national-csv / --regional-csv：原始 SurfAuto / SurfAwst CSV（有无首行数量行均可）；
2. 缓存：<out-dir>/<时次>/input_{national,regional}.csv（首次从接口获取后写入）；
3. 业务接口：按 --config 中的账号获取 --time 时次数据。

用法：
    uv run python -m src.business.fog_compare --time 202502280000 \
        --national-csv data/SurfAuto_20250228000000.csv --regional-csv data/SurfAwst_20250228000000.csv
    uv run python -m src.business.fog_compare --time 202501150000 --sigma-d 3 --tag sd3

--time 为 UTC 时次；输出目录 <out-dir>/<时次>/<tag>/，tag 默认由参数生成，便于并排比较不同参数。
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .algorithms import estimate_both
from .api import (
    _NATIONAL_FIELDS, _REGIONAL_FIELDS, NATIONAL_INTERFACE, REGIONAL_INTERFACE,
    VisibilityApiClient, _parse_response,
)
from .config import FILL_MODES, BusinessConfig, RadiationFogSettings
from .idw import create_visibility_grid
from .plot import _visibility_colormap, _visibility_norm, apply_guangdong_mask, load_guangdong_boundary
from .radiation_fog import FOG_INFERRED, FOG_OBSERVED, RadiationFogModel, ValleyProduct, build_corrected_products

SOURCES = ("national", "national_and_regional")
_RAW_SPECS = {
    NATIONAL_INTERFACE: (_NATIONAL_FIELDS, "V20001", ("V20001_701_01",)),
    REGIONAL_INTERFACE: (_REGIONAL_FIELDS, "V20001_701_01", ("V20001",)),
}


def read_raw_csv(path: Path, interface_id: str) -> pd.DataFrame:
    """读取原始站点 CSV，字段规范化与业务接口一致。"""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gbk"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"无法识别文件编码: {path}")
    first = text.lstrip().split("\n", 1)[0].strip()
    if not first.isdigit():
        text = "0\n" + text
    fields, visibility_field, fallback = _RAW_SPECS[interface_id]
    return _parse_response(text, fields, visibility_field, fallback_visibility_fields=fallback)


def load_inputs(args: argparse.Namespace, case_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.national_csv or args.regional_csv:
        if not (args.national_csv and args.regional_csv):
            raise ValueError("--national-csv 与 --regional-csv 需同时提供")
        return read_raw_csv(args.national_csv, NATIONAL_INTERFACE), read_raw_csv(args.regional_csv, REGIONAL_INTERFACE)
    cache = case_dir / "input_national.csv", case_dir / "input_regional.csv"
    if all(path.exists() for path in cache) and not args.refresh:
        return tuple(pd.read_csv(path, dtype={"code": str}) for path in cache)  # type: ignore[return-value]
    config = BusinessConfig.from_file(args.config)
    batch = VisibilityApiClient(config.api).fetch(args.time)
    case_dir.mkdir(parents=True, exist_ok=True)
    for frame, path in zip((batch.national, batch.regional), cache):
        frame.drop(columns=["_update_time"], errors="ignore").to_csv(path, index=False, encoding="utf-8-sig")
    if batch.errors:
        print(f"[WARN] 部分接口失败: {batch.errors}")
    return batch.national, batch.regional


def settings_from_args(args: argparse.Namespace, base: RadiationFogSettings) -> RadiationFogSettings:
    overrides = {
        "valley_path": args.valley, "vis_threshold_m": args.vis_threshold, "rh_threshold_pct": args.rh_threshold,
        "precip_threshold_mm": args.precip_threshold, "sigma_z_m": args.sigma_z, "sigma_d_km": args.sigma_d,
        "g_cutoff": args.g_cutoff, "infer_radius_km": args.infer_radius, "fill_mode": args.fill_mode,
    }
    values = {key: value for key, value in overrides.items() if value is not None}
    if args.precip_missing_as_wet:
        values["precip_missing_as_dry"] = False
    return dataclasses.replace(base, enabled=True, **values)


def param_tag(s: RadiationFogSettings) -> str:
    return (f"v{s.vis_threshold_m:g}_rh{s.rh_threshold_pct:g}_sz{s.sigma_z_m:g}_sd{s.sigma_d_km:g}"
            f"_g{s.g_cutoff:g}_r{s.infer_radius_km:g}{'' if s.precip_missing_as_dry else '_wetmiss'}")


def default_tag(s: RadiationFogSettings) -> str:
    return f"{s.fill_mode}_{param_tag(s)}"


def low_vis_area(grid: xr.DataArray, threshold_m: float, region: np.ndarray) -> float:
    """区域内能见度 < threshold_m 的面积 (km²)。"""
    lat = grid.lat.values
    dy = float(np.abs(np.diff(lat)).mean()) * 111.32
    dx = float(np.abs(np.diff(grid.lon.values)).mean()) * 111.32
    cell = dy * dx * np.cos(np.deg2rad(lat))[:, None]
    low = region & np.isfinite(grid.values) & (grid.values < threshold_m)
    return float(np.broadcast_to(cell, low.shape)[low].sum())


def fog_extent(dataset: xr.Dataset, margin_deg: float = 0.3) -> tuple[float, float, float, float] | None:
    """g > 0 区域的经纬度范围 (lon0, lon1, lat0, lat1)，无影响域时返回 None。"""
    rows, cols = np.nonzero(dataset["fog_influence"].values > 0)
    if rows.size == 0:
        return None
    lon, lat = dataset.lon.values, dataset.lat.values
    return (float(lon[cols].min()) - margin_deg, float(lon[cols].max()) + margin_deg,
            float(lat[rows].min()) - margin_deg, float(lat[rows].max()) + margin_deg)


def _draw_panel(fig, ax, data: xr.DataArray, valley_id: np.ndarray, boundary, stations: pd.DataFrame,
                title: str, extent: tuple[float, float, float, float] | None, *, influence: bool = False) -> None:
    """单幅面板：能见度 (m) 或 g 场，叠加山谷边界、省界和雾站。"""
    lon, lat = data.lon.values, data.lat.values
    if boundary is not None:
        data = apply_guangdong_mask(data, boundary)
    if influence:
        image = ax.pcolormesh(lon, lat, np.where(data.values > 0, data.values, np.nan),
                              cmap="magma_r", vmin=0, vmax=1, shading="auto")
        fig.colorbar(image, ax=ax, shrink=0.8, label="g")
    else:
        image = ax.pcolormesh(lon, lat, data.values / 1000.0, cmap=_visibility_colormap(),
                              norm=_visibility_norm(), shading="auto")
        fig.colorbar(image, ax=ax, shrink=0.8, label="能见度 (km)", ticks=(0, 0.5, 1, 2, 5, 10, 20, 30))
    ax.contour(lon, lat, (valley_id > 0).astype(float), levels=[0.5], colors="#1f5fbf", linewidths=0.5)
    if boundary is not None:
        boundary.boundary.plot(ax=ax, color="black", linewidth=0.8)
        b = boundary.total_bounds
        ax.set_xlim(b[0] - 0.2, b[2] + 0.2)
        ax.set_ylim(b[1] - 0.2, b[3] + 0.2)
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    for flag, marker, color, text in ((FOG_OBSERVED, "o", "red", "实测雾站"), (FOG_INFERRED, "^", "magenta", "虚拟雾站")):
        chosen = stations[stations["is_radiation_fog"] == flag]
        if not chosen.empty:
            ax.scatter(chosen["lon"], chosen["lat"], s=14, marker=marker, facecolor=color,
                       edgecolor="white", linewidth=0.4, label=f"{text} ({len(chosen)})", zorder=5)
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    ax.set_title(title, fontsize=12)


def plot_compare(dataset: xr.Dataset, stations: pd.DataFrame, valley_id: np.ndarray, boundary, title: str,
                 output: Path, extent: tuple[float, float, float, float] | None = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5))
    panels = (("visibility_original", "原算法"), ("visibility", "辐射雾订正"), ("fog_influence", "雾站影响系数 g（最大值）"))
    for ax, (name, label) in zip(axes, panels):
        _draw_panel(fig, ax, dataset[name], valley_id, boundary, stations, label, extent,
                    influence=name == "fog_influence")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, facecolor="white")
    plt.close(fig)


MODE_ORDER = ("weight", "blend_linear", "blend", "blend_depth")
MODE_LABELS = {"weight": "weight（雾站权重×g）", "blend_linear": "blend_linear（线性混合）",
               "blend": "blend（域内填满 + 对数混合）", "blend_depth": "blend_depth（blend + 深度渐变）"}


def find_mode_files(case_dir: Path, suffix: str, source: str) -> dict[str, Path]:
    """各填充模式下 <source>.nc 的路径；weight 兼容早期不带模式名的目录。"""
    found: dict[str, Path] = {}
    for mode in MODE_ORDER:
        candidates = [case_dir / f"{mode}_{suffix}"] + ([case_dir / suffix] if mode == "weight" else [])
        for path in candidates:
            if (path / f"{source}.nc").exists():
                found[mode] = path / f"{source}.nc"
                break
    return found


def plot_modes(times: list[datetime], out_dir: Path, settings: RadiationFogSettings, dem: xr.Dataset,
               boundary, sources: tuple[str, ...]) -> list[Path]:
    """读取已有结果，把原算法与各候选模式放在同一张图上；每行一个（时次, 路径）。

    每个时次输出 <out-dir>/<时次>/modes_<参数>.png；多个时次时另输出汇总图。
    """
    suffix = param_tag(settings)
    valley_id = ValleyProduct.load(settings.valley_path, dem).valley_id
    region = None
    if boundary is not None:
        region = np.isfinite(apply_guangdong_mask(
            xr.DataArray(np.ones(valley_id.shape), coords={"lat": dem.lat, "lon": dem.lon}, dims=("lat", "lon")),
            boundary).values)
    all_rows: list[tuple[str, dict[str, xr.DataArray], pd.DataFrame, tuple | None]] = []
    outputs: list[Path] = []
    for time in times:
        stamp = time.strftime("%Y%m%d%H%M")
        rows = []
        for source in sources:
            mode_files = find_mode_files(out_dir / stamp, suffix, source)
            if not mode_files:
                print(f"[WARN] {stamp} {source} 没有参数为 {suffix} 的结果，请先运行各模式的对比")
                continue
            missing = [mode for mode in MODE_ORDER if mode not in mode_files]
            if missing:
                print(f"[WARN] {stamp} {source} 缺少模式 {missing} 的结果，对应子图留空")
            panels: dict[str, xr.DataArray] = {}
            stations, extent = pd.DataFrame(columns=["lon", "lat", "is_radiation_fog"]), None
            for mode, nc_path in mode_files.items():
                path = nc_path.parent
                with xr.open_dataset(nc_path) as source_ds:
                    ds = source_ds.load()
                panels.setdefault("original", ds["visibility_original"])
                panels[mode] = ds["visibility"]
                box = fog_extent(ds)
                if box is not None:
                    extent = box if extent is None else (min(extent[0], box[0]), max(extent[1], box[1]),
                                                         min(extent[2], box[2]), max(extent[3], box[3]))
                if stations.empty and (path / f"{source}_stations.csv").exists():
                    stations = pd.read_csv(path / f"{source}_stations.csv", dtype={"code": str})
            if panels:
                label = f"{time.strftime('%Y-%m-%d %H:%M UTC')}（北京时 {_beijing(time)}） {source}"
                rows.append((label, panels, stations, extent))
        if rows:
            output = out_dir / stamp / f"modes_{suffix}.png"
            _plot_mode_rows(rows, valley_id, boundary, region, output)
            outputs.append(output)
            all_rows.extend(rows)
    if len(times) > 1 and all_rows:
        stamps = f"{times[0].strftime('%Y%m%d%H%M')}-{times[-1].strftime('%Y%m%d%H%M')}"
        output = out_dir / f"modes_{suffix}_{stamps}.png"
        _plot_mode_rows(all_rows, valley_id, boundary, region, output)
        outputs.append(output)
    for output in outputs:
        print(f"[OK] 多模式对比图: {output.resolve()}")
    return outputs


def _plot_mode_rows(rows, valley_id: np.ndarray, boundary, region: np.ndarray | None, output: Path) -> None:
    columns = ("original", *MODE_ORDER)
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(6.2 * len(columns), 5.2 * len(rows)), squeeze=False)
    for row_axes, (label, panels, stations, extent) in zip(axes, rows):
        for ax, column in zip(row_axes, columns):
            if column not in panels:
                ax.set_title(f"{MODE_LABELS.get(column, column)}：无结果", fontsize=12)
                ax.axis("off")
                continue
            data = panels[column]
            area = low_vis_area(data, 1000, region if region is not None else np.ones(data.shape, dtype=bool))
            name = "原算法" if column == "original" else MODE_LABELS[column]
            _draw_panel(fig, ax, data, valley_id, boundary, stations, f"{name}\n<1 km 面积 {area:.0f} 平方千米", extent)
        row_axes[0].text(-0.12, 0.5, label, transform=row_axes[0].transAxes, rotation=90,
                         va="center", ha="center", fontsize=11)
        row_axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, facecolor="white")
    plt.close(fig)


def _beijing(time: datetime) -> str:
    return time.astimezone(timezone(timedelta(hours=8))).strftime("%m-%d %H时")


def _resources(args: argparse.Namespace) -> tuple[RadiationFogSettings, xr.Dataset, object]:
    config = _try_config(args.config)
    base = config.radiation_fog if config else RadiationFogSettings()
    dem_path = args.dem or (config.dem_path if config else Path("data/assets/dem/merged_dem_data.nc"))
    boundary_path = args.boundary or (
        config.guangdong_boundary_path if config else Path("data/assets/gis/guangdong/广东省_省界.shp")
    )
    with xr.open_dataset(dem_path) as source:
        dem = source.load()
    boundary = load_guangdong_boundary(boundary_path) if boundary_path.exists() else None
    return settings_from_args(args, base), dem, boundary


def run(args: argparse.Namespace) -> pd.DataFrame:
    stamp = args.time.strftime("%Y%m%d%H%M")
    case_dir = args.out_dir / stamp
    national, regional = load_inputs(args, case_dir)
    settings, dem, boundary = _resources(args)
    tag = args.tag or default_tag(settings)
    out = case_dir / tag
    out.mkdir(parents=True, exist_ok=True)

    model = RadiationFogModel.load(settings, dem)
    estimates = estimate_both(national, regional)
    selected = SOURCES if args.source == "both" else (args.source,)
    estimates = {source: estimates[source] for source in selected}
    grids = {source: create_visibility_grid(frame, dem) for source, frame in estimates.items()}
    products = build_corrected_products(national, regional, dem, model, estimates, grids)

    if boundary is not None:
        region = np.isfinite(apply_guangdong_mask(grids[selected[0]], boundary).values)
    else:
        region = np.ones(model.product.valley_id.shape, dtype=bool)
    rows = []
    time_label = args.time.strftime("%Y-%m-%d %H:%M UTC")
    for source in selected:
        product = products[source]
        product.dataset.to_netcdf(out / f"{source}.nc")
        product.frame.to_csv(out / f"{source}_stations.csv", index=False, encoding="utf-8-sig")
        ds = product.dataset
        row = {"time": stamp, "source": source, "tag": tag, **product.summary}
        for threshold in (500, 1000):
            before = low_vis_area(ds["visibility_original"], threshold, region)
            after = low_vis_area(ds["visibility"], threshold, region)
            row[f"area_lt{threshold}_original_km2"] = round(before, 1)
            row[f"area_lt{threshold}_corrected_km2"] = round(after, 1)
        changed = np.isfinite(ds["visibility"].values) & region
        row["max_abs_change_m"] = round(float(np.nanmax(np.abs(
            ds["visibility"].values[changed] - ds["visibility_original"].values[changed]), initial=0.0)), 1)
        rows.append(row)
        plot_compare(ds, product.frame, model.product.valley_id, boundary,
                     f"{source}  {time_label}  {tag}", out / f"{source}_compare.png")
        extent = fog_extent(ds)
        if extent is not None:
            plot_compare(ds, product.frame, model.product.valley_id, boundary,
                         f"{source}  {time_label}  {tag}（雾区放大）", out / f"{source}_compare_zoom.png", extent)
        print(f"[OK] {source}: 实测雾站 {row['fog_observed']}，虚拟雾站 {row['fog_inferred']}，"
              f"影响域 {row['fog_domains']}；<1km 面积 {row['area_lt1000_original_km2']} -> "
              f"{row['area_lt1000_corrected_km2']} km²")
    stats = pd.DataFrame(rows)
    stats.to_csv(out / "stats.csv", index=False, encoding="utf-8-sig")
    (out / "settings.txt").write_text(repr(settings), encoding="utf-8")
    print(f"[OK] 输出目录: {out.resolve()}")
    return stats


def _try_config(path: Path | None) -> BusinessConfig | None:
    try:
        return BusinessConfig.from_file(path)
    except (FileNotFoundError, ValueError):
        return None


def _parse_time(value: str) -> datetime:
    for fmt in ("%Y%m%d%H%M", "%Y%m%d%H", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"无法解析时次（UTC）: {value}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="辐射雾订正个例对比（原算法 vs 订正算法）")
    parser.add_argument("--time", type=_parse_time, nargs="+", required=True,
                        help="UTC 时次，如 202502280000；可给多个")
    parser.add_argument("--plot-modes", action="store_true",
                        help="不重新计算，读取已有各填充模式结果，与原算法画在同一张图上")
    parser.add_argument("--national-csv", type=Path)
    parser.add_argument("--regional-csv", type=Path)
    parser.add_argument("--config", type=Path, help="业务配置文件，默认按平台选择")
    parser.add_argument("--refresh", action="store_true", help="忽略缓存，重新从接口获取")
    parser.add_argument("--dem", type=Path)
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--valley", type=Path, help="山谷产品 NetCDF")
    parser.add_argument("--source", choices=("both", *SOURCES), default="both")
    parser.add_argument("--out-dir", type=Path, default=Path("output/fog_compare"))
    parser.add_argument("--tag", help="输出子目录名，默认由参数生成")
    parser.add_argument("--vis-threshold", type=float, help="雾站能见度阈值 (m)")
    parser.add_argument("--rh-threshold", type=float, help="推断起雾的相对湿度阈值 (%%)")
    parser.add_argument("--precip-threshold", type=float, help="1 小时降水阈值 (mm)")
    parser.add_argument("--precip-missing-as-wet", action="store_true", help="降水缺测视为有降水")
    parser.add_argument("--sigma-z", type=float, help="垂直衰减 σz (m)")
    parser.add_argument("--sigma-d", type=float, help="水平衰减 σd (km)")
    parser.add_argument("--g-cutoff", type=float, help="g 截断阈值")
    parser.add_argument("--infer-radius", type=float, help="推断起雾时实测雾站搜索半径 (km)")
    parser.add_argument("--fill-mode", choices=FILL_MODES,
                        help="blend_depth=blend 基础上按谷底深度渐变（默认）；blend=影响域内雾站填充+对数混合；"
                             "blend_linear=线性混合、域内保留垂直衰减；weight=仅给雾站权重乘 g")
    args = parser.parse_args(argv)
    times = args.time
    if args.plot_modes:
        settings, dem, boundary = _resources(args)
        sources = SOURCES if args.source == "both" else (args.source,)
        plot_modes(times, args.out_dir, settings, dem, boundary, sources)
        return
    if len(times) > 1 and (args.national_csv or args.regional_csv):
        parser.error("指定 --national-csv/--regional-csv 时只能给一个 --time")
    for time in times:
        run(argparse.Namespace(**{**vars(args), "time": time}))


if __name__ == "__main__":
    main()
