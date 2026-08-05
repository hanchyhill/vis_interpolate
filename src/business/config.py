"""业务流程配置。"""

from __future__ import annotations

import json
import os
import platform
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_PROVINCES = ("广东", "广西", "湖南", "江西", "福建", "海南")


@dataclass(frozen=True)
class ApiSettings:
    base_url: str = "http://172.22.1.175/di/http.action"
    user_id: str = ""
    password: str = ""
    # province保留用于兼容旧调用；未指定时使用provinces中的六省范围。
    province: str | None = None
    provinces: tuple[str, ...] = DEFAULT_PROVINCES
    timeout_seconds: int = 60
    retries: int = 3

    def requested_provinces(self) -> tuple[str, ...]:
        values = (self.province,) if self.province else self.provinces
        return tuple(dict.fromkeys(value.strip() for value in values if value and value.strip()))


@dataclass(frozen=True)
class BusinessConfig:
    repo_root: Path
    api: ApiSettings
    dem_path: Path
    state_path: Path
    lock_path: Path
    log_path: Path
    csv_national_root: Path
    csv_combined_root: Path
    nc_national_root: Path
    nc_combined_root: Path
    vis_img_root: Path = Path("data/vis_img")
    guangdong_boundary_path: Path = Path(
        "data/assets/gis/guangdong/广东省_省界.shp"
    )
    schedule_minutes: tuple[int, ...] = field(
        default=(2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57)
    )

    @classmethod
    def from_file(
        cls,
        config_path: str | Path | None = None,
        *,
        repo_root: str | Path | None = None,
        dem_path: str | Path | None = None,
        province: str | None = None,
        provinces: str | tuple[str, ...] | list[str] | None = None,
    ) -> "BusinessConfig":
        root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        root = root.resolve()
        configured_path = config_path or os.environ.get("VIS_BUSINESS_CONFIG")
        credential_path = (
            Path(configured_path)
            if configured_path
            else _default_config_path(root)
        )
        if not credential_path.is_absolute():
            credential_path = (root / credential_path).resolve()
        values: dict[str, Any] = json.loads(credential_path.read_text(encoding="utf-8"))
        user_id = values.get("userId") or values.get("user_id")
        password = values.get("pwd") or values.get("password")
        if not user_id or not password:
            raise ValueError(f"配置文件缺少 userId 或 pwd: {credential_path}")

        configured = values.get("provinces")
        if provinces:
            selected_provinces = _normalize_provinces(provinces)
            selected_province = None
        elif province:
            selected_provinces = (province,)
            selected_province = province
        elif configured:
            selected_provinces = _normalize_provinces(configured)
            selected_province = None
        elif values.get("province"):
            selected_provinces = (str(values["province"]),)
            selected_province = str(values["province"])
        else:
            selected_provinces = DEFAULT_PROVINCES
            selected_province = None

        api = ApiSettings(
            base_url=str(values.get("baseUrl", ApiSettings.base_url)),
            user_id=str(user_id),
            password=str(password),
            province=selected_province,
            provinces=selected_provinces,
            timeout_seconds=int(values.get("timeoutSeconds", 60)),
            retries=max(1, int(values.get("retries", 3))),
        )
        data_root = _resolve_path(values.get("dataRoot", "data"), root)
        default_dem = _resolve_path("data/assets/dem/merged_dem_data.nc", root)
        default_boundary = _resolve_path(
            "data/assets/gis/guangdong/广东省_省界.shp",
            root,
        )
        selected_dem = _resolve_path(dem_path, root) if dem_path else _resolve_path(
            values.get("demPath", default_dem), root
        )
        return cls(
            repo_root=root,
            api=api,
            dem_path=selected_dem,
            state_path=_configured_data_path(values, "statePath", data_root / "business" / "pipeline_state.sqlite", data_root),
            lock_path=_configured_data_path(values, "lockPath", data_root / "business" / "pipeline.lock", data_root),
            log_path=_configured_data_path(values, "logPath", data_root / "business" / "business.log", data_root),
            csv_national_root=_configured_data_path(values, "csvNationalRoot", data_root / "vis_estimated_base_nation_station", data_root),
            csv_combined_root=_configured_data_path(values, "csvCombinedRoot", data_root / "vis_estimated_base_nation_and_regional_station", data_root),
            nc_national_root=_configured_data_path(values, "ncNationalRoot", data_root / "idw_nc" / "national", data_root),
            nc_combined_root=_configured_data_path(values, "ncCombinedRoot", data_root / "idw_nc" / "national_and_regional", data_root),
            vis_img_root=_configured_data_path(values, "visImgRoot", data_root / "vis_img", data_root),
            guangdong_boundary_path=_resolve_path(
                values.get("guangdongBoundaryPath", default_boundary), root
            ),
        )


def _normalize_provinces(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        items = re.split(r"[,，]", value)
    else:
        items = value
    result = tuple(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))
    if not result:
        raise ValueError("省份列表不能为空")
    return result


def _default_config_path(root: Path) -> Path:
    """根据运行平台选择默认配置文件。"""
    filename = "local.config.json" if platform.system() == "Windows" else "server.config.json"
    return root / "src" / "config" / filename


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _configured_data_path(values: dict[str, Any], key: str, default: Path, data_root: Path) -> Path:
    value = values.get(key)
    return _resolve_path(value, data_root) if value else default
