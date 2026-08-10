"""能见度接口访问与站点数据规范化。"""

from __future__ import annotations

import io
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    errors: tuple[str, ...] = ()
    request_timings: dict[str, float] | None = None
    parse_seconds: float = 0.0
    source_update_time: datetime | None = None

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
        self._last_request_url: str | None = None
        self._request_context = threading.local()

    def fetch(self, timestamp: datetime) -> StationBatch:
        timestamp = _as_utc(timestamp)
        national_frames: list[pd.DataFrame] = []
        regional_frames: list[pd.DataFrame] = []
        national_markers: list[int | None] = []
        regional_markers: list[int | None] = []
        errors: list[str] = []
        request_timings: dict[str, float] = {}
        parse_seconds = 0.0
        tasks = [
            (NATIONAL_INTERFACE, province, _NATIONAL_FIELDS, "V20001", ("V20001_701_01",))
            for province in self.settings.requested_provinces()
        ] + [
            (REGIONAL_INTERFACE, province, _REGIONAL_FIELDS, "V20001_701_01", ("V20001",))
            for province in self.settings.requested_provinces()
        ]
        with ThreadPoolExecutor(
            max_workers=min(self.settings.request_concurrency, max(1, len(tasks))),
            thread_name_prefix="visibility-api",
        ) as executor:
            futures = {
                executor.submit(self._fetch_one, timestamp, *task): task[:2]
                for task in tasks
            }
            for future in as_completed(futures):
                interface_id, province = futures[future]
                key = f"{interface_id}:{province}"
                frame, marker, elapsed, request_elapsed, parsed_elapsed, error = future.result()
                request_timings[key] = request_elapsed
                parse_seconds += parsed_elapsed
                if error:
                    errors.append(error)
                    continue
                if interface_id == NATIONAL_INTERFACE:
                    national_frames.append(frame)
                    national_markers.append(marker)
                else:
                    regional_frames.append(frame)
                    regional_markers.append(marker)
        if not national_frames or not regional_frames:
            detail = "\n".join(errors)
            raise ValueError(f"没有获得国家站或区域站有效响应{': ' + detail if detail else ''}")
        national = _merge_province_frames(national_frames)
        regional = _merge_province_frames(regional_frames)
        return StationBatch(
            national=national,
            regional=regional,
            marker_counts={"national": _sum_markers(national_markers), "regional": _sum_markers(regional_markers)},
            errors=tuple(sorted(errors)),
            request_timings=request_timings,
            parse_seconds=parse_seconds,
            source_update_time=_latest_update_time(national, regional),
        )

    def _fetch_one(
        self,
        timestamp: datetime,
        interface_id: str,
        province: str,
        fields: list[str],
        visibility_field: str,
        fallback_visibility_fields: tuple[str, ...],
    ) -> tuple[pd.DataFrame | None, int | None, float, float, float, str | None]:
        started = time.perf_counter()
        try:
            request_started = time.perf_counter()
            text, marker = self._request(interface_id, timestamp, province)
            request_elapsed = time.perf_counter() - request_started
            parse_started = time.perf_counter()
            try:
                frame = _parse_response(
                    text,
                    fields,
                    visibility_field,
                    fallback_visibility_fields=fallback_visibility_fields,
                )
            except Exception as exc:  # noqa: BLE001 - include raw response for diagnosis
                error = _format_response_error(
                    exc,
                    interface_id,
                    province,
                    getattr(self._request_context, "last_request_url", None) or self._last_request_url,
                    text,
                )
                return None, None, time.perf_counter() - started, request_elapsed, time.perf_counter() - parse_started, error
            return frame, marker, time.perf_counter() - started, request_elapsed, time.perf_counter() - parse_started, None
        except Exception as exc:  # noqa: BLE001 - isolate one province/interface failure
            return None, None, time.perf_counter() - started, time.perf_counter() - started, 0.0, (
                f"接口请求失败（接口={interface_id}, 省份={province}）: {exc}"
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
        request_url = request.full_url
        self._last_request_url = request_url
        self._request_context.last_request_url = request_url
        last_error: Exception | None = None
        last_body: str | None = None
        for attempt in range(self.settings.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                    raw = response.read()
                text = raw.decode("utf-8-sig")
                marker = _response_marker(text)
                return text, marker
            except urllib.error.HTTPError as exc:
                last_error = exc
                try:
                    last_body = exc.read().decode("utf-8-sig", errors="replace")
                except Exception:  # noqa: BLE001 - retain the HTTP error if body cannot be read
                    last_body = None
            except Exception as exc:  # noqa: BLE001 - retry boundary
                last_error = exc
            if attempt + 1 < self.settings.retries:
                time.sleep(2**attempt)
        raise RuntimeError(
            f"接口 {interface_id} 请求失败\n"
            f"请求 URL: {_mask_url(request_url)}\n"
            f"响应 body:\n{last_body or '<empty>'}\n"
            f"错误: {last_error}"
        ) from last_error


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


def _mask_url(url: str | None) -> str:
    """隐藏 URL 中的 API 密码，保留其余请求参数用于排查。"""
    if not url:
        return "<unknown>"
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    masked_query = urllib.parse.urlencode([
        (key, "***" if key.lower() in {"pwd", "password"} else value)
        for key, value in query
    ])
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, masked_query, parsed.fragment)
    )


def _format_response_error(
    error: Exception,
    interface_id: str,
    province: str,
    request_url: str | None,
    body: str,
) -> str:
    return (
        f"{error}（接口={interface_id}, 省份={province}）\n"
        f"请求 URL: {_mask_url(request_url)}\n"
        f"响应 body:\n{body or '<empty>'}"
    )


def _parse_response(
    text: str,
    fields: list[str],
    visibility_field: str,
    fallback_visibility_fields: tuple[str, ...] = (),
) -> pd.DataFrame:
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
    # 部分时次主能见度字段会整列缺测，但同一响应仍提供另一种能见度产品；
    # 仅对主字段缺测的行回退，主字段有效值保持不变。
    visibility = frame[visibility_field].copy()
    for fallback_field in fallback_visibility_fields:
        if fallback_field in frame.columns:
            visibility = visibility.fillna(frame[fallback_field])
    selected[visibility_field] = visibility
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
    frame = frame.drop_duplicates("code", keep="first").drop(columns=["_quality"])
    return frame.reset_index(drop=True)


def _latest_update_time(*frames: pd.DataFrame) -> datetime | None:
    values: list[pd.Timestamp] = []
    for frame in frames:
        if "_update_time" not in frame:
            continue
        parsed = pd.to_datetime(frame["_update_time"], errors="coerce", utc=True).dropna()
        values.extend(parsed.tolist())
    if not values:
        return None
    return max(values).to_pydatetime()


def _normalize_codes(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip()
    return result.str.replace(r"\.0$", "", regex=True)
