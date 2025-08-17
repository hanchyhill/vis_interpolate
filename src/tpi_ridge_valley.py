#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import warnings
import os
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
from rasterio.features import shapes
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

try:
    import fiona
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    SHAPELY_OK = True
except Exception:
    SHAPELY_OK = False


def is_projected_meters(crs):
    try:
        if crs is None:
            return False
        # 简单判断：常见投影单位为 metre
        return crs.axis_info and any(ax.unit_name.lower().startswith("metre") or ax.unit_name.lower().startswith("meter")
                                     for ax in crs.axis_info)
    except Exception:
        return False


def gaussian_smooth(arr, sigma_px):
    if sigma_px is None or sigma_px <= 0:
        return arr
    arr_filled = arr.copy()
    # 用局部均值填补 NaN 再高斯，避免泄漏
    nanmask = ~np.isfinite(arr_filled)
    if nanmask.any():
        # 简单用最近邻填补 NaN（更稳健可用距离变换，这里求稳且速度快）
        arr_fill = arr_filled.copy()
        arr_fill[nanmask] = 0.0
        w = (~nanmask).astype(np.float32)
        k = int(max(3, round(6 * sigma_px)))  # 足够大的核
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), dtype=np.float32)
        sum_arr = ndi.convolve(arr_fill, kernel, mode="nearest")
        sum_w = ndi.convolve(w, kernel, mode="nearest")
        with np.errstate(invalid='ignore', divide='ignore'):
            local_mean = sum_arr / sum_w
        arr_filled[nanmask] = local_mean[nanmask]
    return ndi.gaussian_filter(arr_filled, sigma=sigma_px)


def tpi_weighted(arr, size_px):
    """
    用权重卷积计算 TPI（中心像元减去邻域均值）。
    处理了边界与 NoData：均值= sum(values)/sum(weights)
    """
    assert size_px >= 3 and size_px % 2 == 1, "窗口必须为奇数且>=3"
    valid = np.isfinite(arr)
    arr0 = np.where(valid, arr, 0.0).astype(np.float32)
    w = valid.astype(np.float32)

    footprint = np.ones((size_px, size_px), dtype=np.float32)
    # 中心不参与均值（常见 TPI 定义）
    c = size_px // 2
    footprint[c, c] = 0.0

    sum_vals = ndi.convolve(arr0, footprint, mode="nearest")
    sum_w = ndi.convolve(w, footprint, mode="nearest")

    with np.errstate(invalid='ignore', divide='ignore'):
        neigh_mean = sum_vals / sum_w
    # TPI:
    tpi = arr - neigh_mean
    # 对无邻居（sum_w==0）设为 NaN
    tpi[sum_w == 0] = np.nan
    return tpi


def compute_thresholds(tpi, method="quantile", q_low=25, q_high=75):
    t = tpi[np.isfinite(tpi)]
    if t.size == 0:
        raise ValueError("TPI 全为无效值，检查输入 DEM。")

    if method == "quantile":
        low = np.percentile(t, q_low)
        high = np.percentile(t, q_high)
        return low, high, None
    elif method == "zscore":
        mu = np.nanmean(t)
        sd = np.nanstd(t)
        # 常见经验：|z| ≥ 1 → ridge/valley，可调
        return mu - 1*sd, mu + 1*sd, (mu, sd)
    elif method == "mad":
        med = np.nanmedian(t)
        mad = np.nanmedian(np.abs(t - med))
        # 将 MAD 近似换算为 σ：σ≈1.4826*MAD
        sigma = 1.4826 * mad if mad > 0 else np.nan
        return med - 1*sigma, med + 1*sigma, (med, sigma)
    else:
        raise ValueError(f"未知阈值方法: {method}")


def classify_from_tpi(tpi, low_thr, high_thr):
    cls = np.zeros_like(tpi, dtype=np.int8)
    cls[np.isfinite(tpi) & (tpi <= low_thr)] = -1  # valley
    cls[np.isfinite(tpi) & (tpi >= high_thr)] = 1   # ridge
    # 中间为 0（坡/平）
    cls[~np.isfinite(tpi)] = 0
    return cls


def label_blocks(cls, label_which="both"):
    """
    对分类结果做连通域标号。
    label_which: "ridge" | "valley" | "both"
    返回：
      labels_ridge, num_ridge, labels_valley, num_valley
    """
    structure = np.ones((3, 3), dtype=np.uint8)  # 8 邻域
    labels_ridge = np.zeros_like(cls, dtype=np.int32)
    labels_valley = np.zeros_like(cls, dtype=np.int32)
    num_ridge = num_valley = 0

    if label_which in ("ridge", "both"):
        ridge_mask = (cls == 1)
        labels_ridge, num_ridge = ndi.label(ridge_mask, structure=structure)

    if label_which in ("valley", "both"):
        valley_mask = (cls == -1)
        labels_valley, num_valley = ndi.label(valley_mask, structure=structure)

    return labels_ridge, num_ridge, labels_valley, num_valley


def save_raster(path, arr, profile, dtype=None, nodata=None):
    prof = profile.copy()
    if dtype is None:
        dtype = rasterio.float32 if arr.dtype.kind == 'f' else rasterio.int32
    prof.update(count=1, dtype=dtype, nodata=nodata)
    with rasterio.open(path, "w", **prof) as dst:
        if nodata is not None and np.isnan(nodata):
            # 如果 nodata 是 NaN，用掩膜写出
            dst.write(arr.astype(dtype), 1, masked=np.isnan(arr))
        else:
            dst.write(arr.astype(dtype), 1)


def export_polygons_gpkg(path_gpkg, cls, transform, crs):
    if not SHAPELY_OK:
        warnings.warn("未安装 fiona/shapely，无法导出矢量面；跳过。")
        return
    # 只导出山脊/山谷（排除 0）
    mask = cls != 0
    if not np.any(mask):
        warnings.warn("没有可导出的山脊/山谷像元；跳过。")
        return

    # shapes 会把同值区域转为多边形
    geoms = []
    for geom, value in shapes(cls.astype(np.int16), mask=mask, transform=transform):
        geoms.append((shape(geom), int(value)))

    # 写入 GPKG
    schema = {
        'geometry': 'Polygon',
        'properties': {'class': 'int', 'block_id': 'int'}
    }
    # 按类分别标号（与 label_blocks 一致）
    _, _, _, _ = None, None, None, None
    ridge_mask = cls == 1
    valley_mask = cls == -1
    ridge_labels, n_r, valley_labels, n_v = label_blocks(cls, "both")

    # 建立 block_id 查找
    def lookup_block_id(val, poly):
        # 取 polygon 内部一个点的像素，查其 label
        x, y = poly.representative_point().xy
        col = int((x[0] - transform.c) / transform.a)
        row = int((y[0] - transform.f) / transform.e)
        # 边界检查
        if 0 <= row < ridge_labels.shape[0] and 0 <= col < ridge_labels.shape[1]:
            if val == 1:
                return int(ridge_labels[row, col])
            elif val == -1:
                return int(valley_labels[row, col])
        return 0

    with fiona.open(path_gpkg, 'w',
                    driver='GPKG',
                    crs=crs.to_wkt() if crs else None,
                    layer='ridge_valley',
                    schema=schema) as dst:
        for geom, val in geoms:
            dst.write({
                'geometry': mapping(geom),
                'properties': {
                    'class': int(val),
                    'block_id': lookup_block_id(int(val), geom)
                }
            })


def main():
    parser = argparse.ArgumentParser(
        description="TPI 分析：基于 DEM 将地形分为山脊/山谷并分块标号（GeoTIFF & 可选 GPKG）。")
    parser.add_argument("dem", help="输入 DEM GeoTIFF 路径")
    parser.add_argument("--out-prefix", default=None, help="输出前缀（默认与 DEM 同目录同名）")
    # 窗口设置：可用像元或米
    parser.add_argument("--win-px", type=int, default=None, help="TPI 邻域窗口（像元，奇数，默认随 --win-m 或 11）")
    parser.add_argument("--win-m", type=float, default=None, help="TPI 邻域窗口（米，奇数像元近似），优先级低于 --win-px")
    parser.add_argument("--smooth-m", type=float, default=0.0, help="高斯平滑半径（米，对 DEM），默认 0 不平滑")
    parser.add_argument("--thr-method", choices=["quantile", "zscore", "mad"], default="quantile",
                        help="阈值方法：分位数/zscore/MAD（默认 quantile）")
    parser.add_argument("--q-low", type=float, default=25.0, help="低分位（quantile 模式），默认 25")
    parser.add_argument("--q-high", type=float, default=75.0, help="高分位（quantile 模式），默认 75")
    parser.add_argument("--label-which", choices=["both", "ridge", "valley"], default="both", help="分块对象")
    parser.add_argument("--export-gpkg", action="store_true", help="导出面到 GPKG（需 fiona/shapely）")
    parser.add_argument("--preview", action="store_true", help="导出 PNG 预览")
    args = parser.parse_args()

    # 输出前缀
    if args.out_prefix is None:
        stem = os.path.splitext(os.path.basename(args.dem))[0]
        out_prefix = os.path.join(os.path.dirname(args.dem), stem)
    else:
        out_prefix = args.out_prefix
    
    # 创建tpi_ridge_valley子目录用于保存PNG图片
    png_output_dir = os.path.join(os.path.dirname(args.dem), "tpi_ridge_valley")
    os.makedirs(png_output_dir, exist_ok=True)

    with rasterio.open(args.dem) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        if nodata is not None:
            dem[dem == nodata] = np.nan

        # 分辨率（像元宽/高，取绝对值）
        px_w = abs(transform.a)
        px_h = abs(transform.e)
        px_size = float((px_w + px_h) / 2.0)

        if not is_projected_meters(crs):
            warnings.warn("检测到坐标系可能不是米单位（或 CRS 缺失）。"
                          "建议先将 DEM 重投影到米单位投影（如 UTM），否则 --win-m / --smooth-m 将不准确。")

        # 窗口尺寸
        if args.win_px is not None:
            win_px = int(args.win_px)
        elif args.win_m is not None and px_size > 0:
            win_px = int(round(args.win_m / px_size))
        else:
            # 你的分辨率 ~30m，设默认 11px ≈ 330 m
            win_px = 11

        if win_px < 3:
            win_px = 3
        if win_px % 2 == 0:
            win_px += 1

        # 平滑（以像元为单位换算）
        sigma_px = 0.0
        if args.smooth_m and px_size > 0:
            # 以 1 sigma ≈ smooth_m / px_size / 2 只是经验；这里直接用半径≈sigma
            sigma_px = args.smooth_m / px_size

        dem_s = gaussian_smooth(dem, sigma_px=sigma_px if sigma_px > 0 else None)

        # 计算 TPI
        tpi = tpi_weighted(dem_s, size_px=win_px)

        # 阈值
        low_thr, high_thr, extra = compute_thresholds(
            tpi, method=args.thr_method, q_low=args.q_low, q_high=args.q_high
        )

        # 分类
        cls = classify_from_tpi(tpi, low_thr, high_thr)

        # 分块
        labels_ridge, n_ridge, labels_valley, n_valley = label_blocks(cls, args.label_which)

        # 合并为一个 labels 图层（可选：分别输出更清晰）
        labels = np.zeros_like(cls, dtype=np.int32)
        # 为避免 ID 冲突，山谷 ID 加上一个偏移
        if n_ridge > 0:
            labels[labels_ridge > 0] = labels_ridge[labels_ridge > 0]
        if n_valley > 0:
            offset = int(labels.max())
            labels[labels_valley > 0] = labels_valley[labels_valley > 0] + offset

        # 保存
        save_raster(out_prefix + "_tpi.tif", tpi, profile, dtype=rasterio.float32, nodata=np.nan)
        save_raster(out_prefix + "_class.tif", cls, profile, dtype=rasterio.int8, nodata=0)
        save_raster(out_prefix + "_labels.tif", labels, profile, dtype=rasterio.int32, nodata=0)

        if args.export_gpkg:
            export_polygons_gpkg(out_prefix + "_polygons.gpkg", cls, transform, crs)

        if args.preview:
            # 简易预览：分类
            plt.figure(figsize=(8, 6))
            show = cls.copy().astype(np.float32)
            show[show == 0] = np.nan
            plt.imshow(show, interpolation="nearest")
            plt.title("Classification (−1 valley / 0 slope / +1 ridge)")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(png_output_dir, os.path.basename(out_prefix) + "_class.png"), dpi=200)
            plt.close()

            # TPI 直方图（了解阈值）
            plt.figure(figsize=(6, 4))
            vals = tpi[np.isfinite(tpi)]
            plt.hist(vals, bins=100)
            plt.axvline(low_thr, linestyle="--")
            plt.axvline(high_thr, linestyle="--")
            plt.title("TPI histogram with thresholds")
            plt.tight_layout()
            plt.savefig(os.path.join(png_output_dir, os.path.basename(out_prefix) + "_tpi_hist.png"), dpi=200)
            plt.close()

        print(f"[OK] 窗口={win_px}px，px≈{px_size:.3f}m；"
              f"阈值方法={args.thr_method}；低阈={low_thr:.3f}，高阈={high_thr:.3f}")
        print(f"[OK] 输出：\n  {out_prefix}_tpi.tif\n  {out_prefix}_class.tif\n  {out_prefix}_labels.tif")
        if args.export_gpkg:
            print(f"  {out_prefix}_polygons.gpkg")
        if args.preview:
            print(f"  {os.path.join(png_output_dir, os.path.basename(out_prefix) + '_class.png')}\n  {os.path.join(png_output_dir, os.path.basename(out_prefix) + '_tpi_hist.png')}")


if __name__ == "__main__":
    main()
# uv run tpi_ridge_valley.py H:\data\DEM\ASTGTM2_N23E111_dem.tif --win-px 31 --preview
