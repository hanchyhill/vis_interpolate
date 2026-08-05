"""能见度接口访问与站点数据规范化。"""

from __future__ import annotations

import io
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .config import ApiSettings


NATIONAL_INTERFACE = "getSurfAutoOrg4Prov"
REGIONAL_INTERFACE = "getSurfAwstOrg4Prov"

_COMMON_FIELDS = [
    "V01301", "VF01015_CN", "V_CITY", "V_COUNTY", "V06001", "V05001", "V07001", "V13003"
]
_NATIONAL_FIELDS = _COMMON_FIELDS + ["V20001"]
_REGIONAL_FIELDS = _COMMON_FIELDS + ["V20001_701_01"]


@dataclass
class StationBatch:
    national: pd.DataFrame
    regional: pd.DataFrame
    marker_counts: dict[str, int | None]

    @property
    def sample_counts(self) -> dict[str, int]:
        n = self.national.dropna(subset=["lon", "lat", "altitude", "rh", "vis"])
        r_base = self.regional.dropna(subset=["lon", "lat", "altitude", "rh"])
        r_vis = r_base.dropna(subset=["vis"])
        return {
            "national_valid": int(n["code"].nunique()),
            "regional_valid_rh": int(r_base["code"].nunique()),
            "regional_valid_vis": int(r_vis["code"].nunique()),
        }


class VisibilityApiClient:
    def __init__(self, settings: ApiSettings):
        self.settings = settings

    def fetch(self, timestamp: datetime) -> StationBatch:
        timestamp = _as_utc(timestamp)
        national_frames: list[pd.DataFrame] = []
        regional_frames: list[pd.DataFrame] = []
        national_markers: list[int | None] = []
        regional_markers: list[int | None] = []
        for province in self.settings.requested_provinces():
            national_text, national_marker = self._request(NATIONAL_INTERFACE, timestamp, province)
            regional_text, regional_marker = self._request(REGIONAL_INTERFACE, timestamp, province)
            national_frames.append(_parse_response(national_text, _NATIONAL_FIELDS, "V20001"))
            regional_frames.append(_parse_response(regional_text, _REGIONAL_FIELDS, "V20001_701_01"))
            national_markers.append(national_marker)
            regional_markers.append(regional_marker)
        if not national_frames or not regional_frames:
            raise ValueError("没有配置可请求的省份")
        national = _merge_province_frames(national_frames)
        regional = _merge_province_frames(regional_frames)
        return StationBatch(
            national=national,
            regional=regional,
            marker_counts={"national": _sum_markers(national_markers), "regional": _sum_markers(regional_markers)},
        )

    def _request(self, interface_id: str, timestamp: datetime, province: str) -> tuple[str, int | None]:
        query = urllib.parse.urlencode(
            {
                "userId": self.settings.user_id,
                "pwd": self.settings.password,
                "interfaceId": interface_id,
                "dataFormat": "csv",
                "ymdhms": timestamp.strftime("%Y%m%d%H%M%S"),
                "prov": province,
            }
        )
        request = urllib.request.Request(
            f"{self.settings.base_url}?{query}",
            headers={"User-Agent": "vis-interpolate-business/1.0", "Accept": "text/csv,*/*"},
        )
        last_error: Exception | None = None
        for attempt in range(self.settings.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                    raw = response.read()
                text = raw.decode("utf-8-sig")
                marker = _response_marker(text)
                return text, marker
            except Exception as exc:  # noqa: BLE001 - retry boundary
                last_error = exc
                if attempt + 1 < self.settings.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"接口 {interface_id} 请求失败: {last_error}") from last_error


def _merge_province_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(frames, ignore_index=True, sort=False)
    return merged.drop_duplicates("code", keep="first").reset_index(drop=True)


def _sum_markers(values: list[int | None]) -> int | None:
    return sum(values) if values and all(value is not None for value in values) else None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _response_marker(text: str) -> int | None:
    for line in text.splitlines():
        if line.strip():
            try:
                return int(line.strip().lstrip("\ufeff"))
            except ValueError:
                return None
    return None


def _parse_response(text: str, fields: list[str], visibility_field: str) -> pd.DataFrame:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("接口响应缺少数量行或CSV表头")
    try:
        frame = pd.read_csv(
            io.StringIO("\n".join(lines[1:])),
            na_values=[9999, 999999, "9999", "999999"],
            keep_default_na=True,
        )
    except Exception as exc:  # noqa: BLE001 - normalize parser errors
        raise ValueError(f"接口CSV解析失败: {exc}") from exc
    missing = sorted(set(fields) - set(frame.columns))
    if missing:
        raise ValueError(f"接口CSV缺少字段: {', '.join(missing)}")

    selected = frame[fields].copy()
    if "D_UPDATE_TIME" in frame.columns:
        selected["_update_time"] = pd.to_datetime(
            frame["D_UPDATE_TIME"], format="mixed", errors="coerce"
        )
    frame = selected.rename(
        columns={
            "V01301": "code", "VF01015_CN": "name", "V_CITY": "city", "V_COUNTY": "county",
            "V06001": "lon", "V05001": "lat", "V07001": "altitude", "V13003": "rh",
            visibility_field: "vis",
        }
    )
    frame["code"] = _normalize_codes(frame["code"])
    for col in ["lon", "lat", "altitude", "rh", "vis"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame.loc[frame[col].isin([9999, 999999]), col] = np.nan
    frame = frame.dropna(subset=["code"]).reset_index(drop=True)
    frame["_quality"] = frame[["lon", "lat", "altitude", "rh", "vis"]].notna().sum(axis=1)
    sort_columns = ["_quality"]
    ascending = [False]
    if "_update_time" in frame:
        sort_columns.append("_update_time")
        ascending.append(False)
    frame = frame.sort_values(sort_columns, ascending=ascending, kind="stable")
    frame = frame.drop_duplicates("code", keep="first").drop(
        columns=[column for column in ["_quality", "_update_time"] if column in frame]
    )
    return frame.reset_index(drop=True)


def _normalize_codes(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip()
    return result.str.replace(r"\.0$", "", regex=True)
