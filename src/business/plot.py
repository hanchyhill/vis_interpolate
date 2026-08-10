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
from matplotlib.ft2font import FT2Font
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap
from shapely import contains_xy

from src.evaluate_visibility import load_visibility_data


def _configure_fonts() -> bool:
    """选择支持中文的系统字体；没有时由调用方使用英文标签。"""
    candidates = (
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans CN",
        "WenQuanYi Zen Hei",
        "STHeiti",
        "Arial Unicode MS",
    )
    for name in candidates:
        try:
            font_path = font_manager.findfont(
                font_manager.FontProperties(family=name), fallback_to_default=False
            )
            font = FT2Font(font_path)
            if all(font.get_char_index(ord(char)) for char in "广东省能见度"):
                plt.rcParams["font.sans-serif"] = [name]
                plt.rcParams["axes.unicode_minus"] = False
                return True
        except (FileNotFoundError, ValueError):
            continue
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return False


_HAS_CJK_FONT = _configure_fonts()
_VISIBILITY_TICKS = (0, 0.5, 1, 2, 4, 6, 8, 10, 15, 20, 25, 30)


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
    norm = _visibility_norm()
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
    plot_title = title or nc_path.stem
    if not _HAS_CJK_FONT:
        plot_title = nc_path.stem
    ax.set_title(plot_title, fontsize=15, pad=12)
    colorbar = fig.colorbar(
        image,
        ax=ax,
        shrink=0.82,
        pad=0.04,
        boundaries=norm.boundaries,
        ticks=_VISIBILITY_TICKS,
        spacing="proportional",
    )
    # 只保留有对应文字的显式主刻度，关闭BoundaryNorm可能带来的额外次刻度。
    colorbar.ax.yaxis.set_major_locator(FixedLocator(_VISIBILITY_TICKS))
    colorbar.ax.yaxis.set_minor_locator(NullLocator())
    colorbar.ax.set_yticklabels(
        [
            "0", "0.5", "1\n大雾" if _HAS_CJK_FONT else "1\nDense fog", "2", "4", "6",
            "8", "10\n轻雾" if _HAS_CJK_FONT else "10\nLight fog", "15", "20", "25", "30",
        ],
        fontsize=9,
    )
    colorbar.set_label("能见度 (km)" if _HAS_CJK_FONT else "Visibility (km)")
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_cldas_visibility(
    data_path: str,
    boundary_path: Path,
    output_path: Path,
    *,
    title: str | None = None,
) -> Path:
    """使用 contourf 绘制国家局 5km 能见度实况融合产品。

    ``data_path`` 可以是本地 NetCDF 文件或国家局 THREDDS OPeNDAP URL。
    产品通常覆盖全国范围，因此先按广东省边界裁剪，再执行边界遮罩，
    避免对整张 5km 全国网格进行绘图。
    """
    boundary = load_guangdong_boundary(boundary_path)
    visibility = load_visibility_data(data_path)
    if visibility is None:
        raise FileNotFoundError(f"国家局5km能见度产品尚未生成或无法打开: {data_path}")
    if "lat" not in visibility.coords or "lon" not in visibility.coords:
        raise ValueError(f"国家局5km能见度产品缺少lat/lon坐标: {data_path}")
    if visibility.ndim != 2:
        raise ValueError(f"国家局5km能见度产品必须是二维网格，实际维度为{visibility.dims}")

    bounds = boundary.total_bounds
    visibility = visibility.sel(
        lon=_coordinate_slice(visibility.lon, bounds[0], bounds[2]),
        lat=_coordinate_slice(visibility.lat, bounds[1], bounds[3]),
    )
    # 国家局产品以负值表示缺测；缺测点不应参与等值线填色。
    visibility = visibility.where(visibility >= 0)
    masked = apply_guangdong_mask(visibility, boundary)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmap = _visibility_colormap()
    norm = _visibility_norm()
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    image = ax.contourf(
        masked.lon.values,
        masked.lat.values,
        masked.values,
        levels=norm.boundaries,
        cmap=cmap,
        norm=norm,
        extend="max",
        transform=ccrs.PlateCarree(),
    )
    boundary.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=1.5,
        transform=ccrs.PlateCarree(),
    )
    ax.set_extent([bounds[0] - 0.2, bounds[2] + 0.2, bounds[1] - 0.2, bounds[3] + 0.2], ccrs.PlateCarree())
    gridlines = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    gridlines.top_labels = False
    gridlines.right_labels = False
    plot_title = title or Path(data_path).stem
    if not _HAS_CJK_FONT:
        plot_title = Path(data_path).stem
    ax.set_title(plot_title, fontsize=15, pad=12)
    colorbar = fig.colorbar(
        image,
        ax=ax,
        shrink=0.82,
        pad=0.04,
        boundaries=norm.boundaries,
        ticks=_VISIBILITY_TICKS,
        spacing="proportional",
    )
    colorbar.ax.yaxis.set_major_locator(FixedLocator(_VISIBILITY_TICKS))
    colorbar.ax.yaxis.set_minor_locator(NullLocator())
    colorbar.ax.set_yticklabels(
        [
            "0", "0.5", "1\n大雾" if _HAS_CJK_FONT else "1\nDense fog", "2", "4", "6",
            "8", "10\n轻雾" if _HAS_CJK_FONT else "10\nLight fog", "15", "20", "25", "30",
        ],
        fontsize=9,
    )
    colorbar.set_label("能见度 (km)" if _HAS_CJK_FONT else "Visibility (km)")
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _visibility_colormap() -> ListedColormap:
    """创建分段渐变色标，并在1、10、20 km处保留明显色系跳变。"""
    fog = _sample_gradient(("#6B0000", "#FF4D4D"), 64)
    light_fog = _sample_gradient(("#FF8C00", "#FFE66D"), 96)
    green = _sample_gradient(("#006400", "#32CD32", "#ADFF2F"), 48)
    blue = _sample_gradient(("#008B8B", "#2F80ED", "#D9F0FF"), 48)
    return ListedColormap(np.vstack([fog, light_fog, green, blue]), name="visibility_business")


def _visibility_norm(vmax: float = 30.0) -> BoundaryNorm:
    """将三个能见度等级分别分配颜色空间，阈值处产生突变。"""
    boundaries = np.concatenate(
        [
            np.linspace(0.0, 1.0, 65),
            np.linspace(1.0, 10.0, 97)[1:],
            np.linspace(10.0, 20.0, 49)[1:],
            np.linspace(20.0, vmax, 49)[1:],
        ]
    )
    return BoundaryNorm(boundaries, ncolors=256, clip=True)


def _sample_gradient(colors: tuple[str, ...], count: int) -> np.ndarray:
    cmap = LinearSegmentedColormap.from_list("visibility_segment", list(colors), N=count)
    return cmap(np.linspace(0.0, 1.0, count))[:, :3]


def _coordinate_slice(coord: xr.DataArray, lower: float, upper: float) -> slice:
    """为升序或降序经纬度坐标生成正确的裁剪切片。"""
    values = coord.values
    if values.size < 2 or values[0] <= values[-1]:
        return slice(lower, upper)
    return slice(upper, lower)


def main() -> None:
    parser = argparse.ArgumentParser(description="对已有能见度IDW NetCDF绘制广东省遮罩图")
    parser.add_argument("--nc", required=True, type=Path, help="输入IDW NetCDF路径")
    parser.add_argument("--boundary", required=True, type=Path, help="广东省边界Shapefile/GeoJSON路径")
    parser.add_argument("--output", required=True, type=Path, help="输出PNG路径")
    args = parser.parse_args()
    plot_visibility(args.nc, args.boundary, args.output)


if __name__ == "__main__":
    main()
