"""业务能见度绘图：广东省边界遮罩和IDW结果输出。"""

from __future__ import annotations

from pathlib import Path
import argparse

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, Normalize
from shapely import contains_xy


def _configure_fonts() -> None:
    """优先选择系统中文字体，避免省名和色标中文变成DejaVu缺字警告。"""
    candidates = ("SimHei", "Microsoft YaHei", "Noto Sans SC", "STHeiti", "Arial Unicode MS")
    selected = None
    for name in candidates:
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=name), fallback_to_default=False
            )
            selected = name
            break
        except (FileNotFoundError, ValueError):
            continue
    plt.rcParams["font.sans-serif"] = [selected or "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


_configure_fonts()


def load_guangdong_boundary(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"广东边界文件不存在: {path}")
    boundary = gpd.read_file(path)
    if boundary.empty:
        raise ValueError(f"广东边界文件为空: {path}")
    if boundary.crs is not None and boundary.crs.to_epsg() != 4326:
        boundary = boundary.to_crs(4326)
    return boundary


def apply_guangdong_mask(vis_data: xr.DataArray, boundary: gpd.GeoDataFrame) -> xr.DataArray:
    if "lat" not in vis_data.coords or "lon" not in vis_data.coords:
        raise ValueError("绘图数据必须包含lat和lon坐标")
    geometry = boundary.geometry.union_all()
    lon_grid, lat_grid = np.meshgrid(vis_data.lon.values, vis_data.lat.values)
    mask = contains_xy(geometry, lon_grid, lat_grid)
    return vis_data.where(mask)


def plot_visibility(
    nc_path: Path,
    boundary_path: Path,
    output_path: Path,
    *,
    title: str | None = None,
) -> Path:
    """读取一个IDW NetCDF，应用广东遮罩并保存PNG。"""
    boundary = load_guangdong_boundary(boundary_path)
    with xr.open_dataset(nc_path) as dataset:
        if "visibility" not in dataset:
            raise ValueError(f"NetCDF缺少visibility变量: {nc_path}")
        visibility = dataset["visibility"].load()
        units = str(visibility.attrs.get("units", "m")).lower()
    if units in {"m", "meter", "meters"}:
        visibility = visibility / 1000.0
    masked = apply_guangdong_mask(visibility, boundary)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmap = _visibility_colormap()
    # 保留原有低能见度红色、高能见度蓝色的色系，但使用连续归一化，
    # 避免BoundaryNorm造成分段跳色，使图面和色标均为平滑渐变。
    norm = Normalize(vmin=0, vmax=30, clip=True)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    image = masked.plot.pcolormesh(
        ax=ax,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        shading="auto",
    )
    boundary.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=1.5,
        transform=ccrs.PlateCarree(),
    )
    bounds = boundary.total_bounds
    ax.set_extent([bounds[0] - 0.2, bounds[2] + 0.2, bounds[1] - 0.2, bounds[3] + 0.2], ccrs.PlateCarree())
    gridlines = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    gridlines.top_labels = False
    gridlines.right_labels = False
    ax.set_title(title or nc_path.stem, fontsize=15, pad=12)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.04)
    colorbar.set_label("能见度 (km)")
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _visibility_colormap() -> LinearSegmentedColormap:
    colors = [
        "#8B0000", "#DC143C", "#FF4500", "#FFA500", "#FFD700",
        "#ADFF2F", "#32CD32", "#00CED1", "#00BFFF", "#87CEEB", "#F0F8FF",
    ]
    return LinearSegmentedColormap.from_list("visibility_business", colors, N=256)


def main() -> None:
    parser = argparse.ArgumentParser(description="对已有能见度IDW NetCDF绘制广东省遮罩图")
    parser.add_argument("--nc", required=True, type=Path, help="输入IDW NetCDF路径")
    parser.add_argument("--boundary", required=True, type=Path, help="广东省边界Shapefile/GeoJSON路径")
    parser.add_argument("--output", required=True, type=Path, help="输出PNG路径")
    args = parser.parse_args()
    plot_visibility(args.nc, args.boundary, args.output)


if __name__ == "__main__":
    main()
