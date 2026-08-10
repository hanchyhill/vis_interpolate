from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from src.business.algorithms import estimate_both
from src.business.api import (
    NATIONAL_INTERFACE,
    REGIONAL_INTERFACE,
    VisibilityApiClient,
    _mask_url,
    _parse_response,
)
from src.business.config import DEFAULT_PROVINCES, ApiSettings, BusinessConfig
from src.business.idw import create_visibility_grid
from src.business.plot import _visibility_colormap, _visibility_norm, plot_cldas_visibility, plot_visibility
from src.business.pipeline import (
    close_logging,
    hourly_visibility_times,
    run_cldas_visibility_backfill,
    run_once,
    _beijing_timestamp,
    window_times,
)
from src.business.state import PipelineState, process_lock


class BusinessPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        close_logging()

    def tearDown(self) -> None:
        close_logging()

    def test_api_response_parser_skips_count_line_and_normalizes_fields(self) -> None:
        text = "2\n\"V01301\",\"VF01015_CN\",\"V_CITY\",\"V_COUNTY\",\"V06001\",\"V05001\",\"V07001\",\"V13003\",\"V20001\"\n\"N1\",\"站1\",\"广州\",\"县\",113,23,10,80,1000\n\"N1\",\"站1\",\"广州\",\"县\",113,23,10,80,9999\n"
        frame = _parse_response(
            text,
            ["V01301", "VF01015_CN", "V_CITY", "V_COUNTY", "V06001", "V05001", "V07001", "V13003", "V20001"],
            "V20001",
        )
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["code"], "N1")
        self.assertEqual(frame.iloc[0]["vis"], 1000)

    def test_api_parser_falls_back_when_primary_visibility_is_missing(self) -> None:
        text = "1\nV01301,VF01015_CN,V_CITY,V_COUNTY,V06001,V05001,V07001,V13003,V20001,V20001_701_01\nN1,站1,广州,县,113,23,10,80,9999,25000\n"
        frame = _parse_response(
            text,
            ["V01301", "VF01015_CN", "V_CITY", "V_COUNTY", "V06001", "V05001", "V07001", "V13003", "V20001"],
            "V20001",
            fallback_visibility_fields=("V20001_701_01",),
        )
        self.assertEqual(frame.iloc[0]["vis"], 25000)

    def test_api_parse_error_includes_request_url_and_response_body(self) -> None:
        class InvalidResponseClient(VisibilityApiClient):
            def _request(self, interface_id, timestamp, province):
                self._last_request_url = (
                    "http://example.test/api?userId=u&pwd=secret&interfaceId="
                    f"{interface_id}&prov={province}"
                )
                return "upstream error", None

        client = InvalidResponseClient(ApiSettings(user_id="u", password="p"),)
        with self.assertRaisesRegex(ValueError, "响应 body:\\nupstream error") as context:
            client.fetch(datetime(2026, 8, 5, 8, 0, tzinfo=timezone.utc))
        message = str(context.exception)
        self.assertIn("请求 URL: http://example.test/api", message)
        self.assertIn("pwd=%2A%2A%2A", message)
        self.assertNotIn("secret", message)

    def test_mask_url_preserves_request_parameters_without_password(self) -> None:
        masked = _mask_url("http://example.test?a=1&pwd=secret&prov=%E5%B9%BF%E4%B8%9C")
        self.assertEqual(masked, "http://example.test?a=1&pwd=%2A%2A%2A&prov=%E5%B9%BF%E4%B8%9C")

    def test_api_client_queries_and_merges_all_six_provinces(self) -> None:
        calls = []

        class FakeClient(VisibilityApiClient):
            def _request(self, interface_id, timestamp, province):
                calls.append((interface_id, province))
                if interface_id == NATIONAL_INTERFACE:
                    return _api_csv("V20001", f"N-{province}", 1000), 1
                return _api_csv("V20001_701_01", f"R-{province}", 2000), 1

        client = FakeClient(ApiSettings(user_id="u", password="p"))
        batch = client.fetch(datetime(2026, 8, 5, 8, 0, tzinfo=timezone.utc))
        self.assertEqual(client.settings.requested_provinces(), DEFAULT_PROVINCES)
        self.assertEqual(len(calls), len(DEFAULT_PROVINCES) * 2)
        self.assertEqual(len(batch.national), len(DEFAULT_PROVINCES))
        self.assertEqual(len(batch.regional), len(DEFAULT_PROVINCES))
        self.assertEqual(batch.marker_counts, {"national": 6, "regional": 6})

    def test_api_client_uses_bounded_concurrency_and_records_timings(self) -> None:
        active = 0
        maximum = 0
        guard = threading.Lock()

        class SlowClient(VisibilityApiClient):
            def _request(self, interface_id, timestamp, province):
                nonlocal active, maximum
                with guard:
                    active += 1
                    maximum = max(maximum, active)
                try:
                    time.sleep(0.02)
                    field = "V20001" if interface_id == NATIONAL_INTERFACE else "V20001_701_01"
                    return _api_csv(field, f"{interface_id}-{province}", 1000), 1
                finally:
                    with guard:
                        active -= 1

        settings = ApiSettings(
            user_id="u",
            password="p",
            provinces=("广东", "广西", "湖南"),
            request_concurrency=2,
        )
        batch = SlowClient(settings).fetch(datetime(2026, 8, 5, 8, 0, tzinfo=timezone.utc))
        self.assertGreater(maximum, 1)
        self.assertLessEqual(maximum, 2)
        self.assertEqual(len(batch.request_timings or {}), 6)
        self.assertGreater(batch.parse_seconds, 0)

    def test_state_backfill_queue_claims_oldest_pending_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = PipelineState(Path(directory) / "state.sqlite")
            times = [
                datetime(2026, 8, 5, 7, 50, tzinfo=timezone.utc),
                datetime(2026, 8, 5, 7, 55, tzinfo=timezone.utc),
                datetime(2026, 8, 5, 8, 0, tzinfo=timezone.utc),
            ]
            state.enqueue(times)
            claimed = state.claim_backfill(exclude=times[-1], limit=1)
            self.assertEqual(claimed, [times[0]])
            state.queue_result(times[0], success=True)
            state.enqueue(times)
            claimed_next = state.claim_backfill(exclude=times[-1], limit=1)
            self.assertEqual(claimed_next, [times[0]])

    def test_province_override_accepts_chinese_comma(self) -> None:
        config = BusinessConfig.from_file(provinces="广东，广西")
        self.assertEqual(config.api.requested_provinces(), ("广东", "广西"))

    def test_json_config_controls_data_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "business.config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "userId": "u",
                        "pwd": "p",
                        "dataRoot": "server-data",
                        "demPath": "dem/merged_dem_data.nc",
                        "csvNationalRoot": "csv-national",
                        "logPath": "logs/business.log",
                    }
                ),
                encoding="utf-8",
            )
            config = BusinessConfig.from_file(config_path, repo_root=root)
            self.assertEqual(config.dem_path, (root / "dem/merged_dem_data.nc").resolve())
            self.assertEqual(config.csv_national_root, (root / "server-data/csv-national").resolve())
            self.assertEqual(config.log_path, (root / "server-data/logs/business.log").resolve())
            self.assertEqual(config.nc_national_root, (root / "server-data/idw_nc/national").resolve())

    def test_default_config_follows_platform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "src" / "config"
            config_dir.mkdir(parents=True)
            (config_dir / "local.config.json").write_text(
                json.dumps({"userId": "windows-user", "pwd": "p"}),
                encoding="utf-8",
            )
            (config_dir / "server.config.json").write_text(
                json.dumps({"userId": "linux-user", "pwd": "p"}),
                encoding="utf-8",
            )

            with patch("src.business.config.platform.system", return_value="Windows"):
                windows_config = BusinessConfig.from_file(repo_root=root)
            with patch("src.business.config.platform.system", return_value="Linux"):
                linux_config = BusinessConfig.from_file(repo_root=root)

            self.assertEqual(windows_config.api.user_id, "windows-user")
            self.assertEqual(linux_config.api.user_id, "linux-user")

    def test_window_is_five_minute_aligned_and_excludes_recent_data(self) -> None:
        now = datetime(2026, 8, 5, 8, 2, tzinfo=timezone.utc)
        self.assertEqual(
            [item.strftime("%H:%M") for item in window_times(now)],
            ["07:35", "07:40", "07:45", "07:50", "07:55"],
        )

    def test_hourly_visibility_times_returns_two_completed_utc_hours(self) -> None:
        now = datetime(2026, 8, 5, 8, 2, tzinfo=timezone.utc)
        self.assertEqual(
            [item.strftime("%Y%m%d%H") for item in hourly_visibility_times(now)],
            ["2026080507", "2026080506"],
        )

    def test_plot_title_timestamp_uses_beijing_time(self) -> None:
        utc_time = datetime(2026, 8, 5, 16, 2, tzinfo=timezone.utc)
        self.assertEqual(_beijing_timestamp(utc_time, "%Y%m%d%H%M"), "202608060002")

    def test_cldas_contourf_plot_uses_boundary_and_saves_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundary_path = root / "guangdong.geojson"
            boundary_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [{
                            "type": "Feature",
                            "properties": {},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[113, 23], [114, 23], [114, 24], [113, 24], [113, 23]]],
                            },
                        }],
                    }
                ),
                encoding="utf-8",
            )
            visibility = xr.DataArray(
                np.array([[500.0, 1000.0, 2000.0], [3000.0, 4000.0, 5000.0], [6000.0, 7000.0, 8000.0]]),
                dims=("lat", "lon"),
                coords={"lat": [23.1, 23.5, 23.9], "lon": [113.1, 113.5, 113.9]},
            )
            output = root / "images" / "cldas.png"
            with patch("src.business.plot.load_visibility_data", return_value=visibility):
                result = plot_cldas_visibility("VIS_2026080507.NC", boundary_path, output)
            self.assertEqual(result, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_missing_cldas_product_is_skipped_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundary_path = root / "guangdong.geojson"
            boundary_path.write_text(
                json.dumps({
                    "type": "FeatureCollection",
                    "features": [{
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[113, 23], [114, 23], [114, 24], [113, 24], [113, 23]]],
                        },
                    }],
                }),
                encoding="utf-8",
            )
            config = BusinessConfig(
                repo_root=root,
                api=ApiSettings(user_id="u", password="p"),
                dem_path=root / "dem.nc",
                state_path=root / "state.sqlite",
                lock_path=root / "pipeline.lock",
                log_path=root / "business.log",
                csv_national_root=root / "csv-national",
                csv_combined_root=root / "csv-combined",
                nc_national_root=root / "nc-national",
                nc_combined_root=root / "nc-combined",
                guangdong_boundary_path=boundary_path,
                vis_img_root=root / "images",
            )
            with patch(
                "src.business.pipeline.plot_cldas_visibility",
                side_effect=FileNotFoundError("数据尚未生成"),
            ):
                outputs = run_cldas_visibility_backfill(
                    datetime(2026, 8, 5, 8, 2, tzinfo=timezone.utc),
                    config,
                    async_plot=False,
                )
            self.assertEqual(outputs, [])
            self.assertEqual(list((root / "images").rglob("*.png")), [])

    def test_state_reprocesses_only_when_count_increases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = PipelineState(Path(directory) / "state.sqlite")
            timestamp = datetime(2026, 8, 5, 8, 0, tzinfo=timezone.utc)
            counts = {"national_valid": 2, "regional_valid_rh": 3, "regional_valid_vis": 1}
            self.assertTrue(state.should_process(timestamp, counts))
            state.success(timestamp, counts, [])
            self.assertFalse(state.should_process(timestamp, counts))
            self.assertTrue(state.should_process(timestamp, {**counts, "regional_valid_vis": 2}))

    def test_two_estimation_paths_have_different_reference_data(self) -> None:
        national = _national_frame()
        regional = _regional_frame()
        outputs = estimate_both(national, regional)
        self.assertEqual(outputs["national"].query("code == 'R1'").iloc[0]["is_vis_est"], 1)
        combined_r1 = outputs["national_and_regional"].query("code == 'R1'").iloc[0]
        self.assertEqual(combined_r1["vis"], 3000)
        self.assertEqual(combined_r1["is_vis_est"], 0)

    def test_idw_returns_meter_units(self) -> None:
        dem = xr.Dataset(
            {"elevation": (("lat", "lon"), np.ones((2, 2)) * 10)},
            coords={"lat": [23.0, 24.0], "lon": [113.0, 114.0]},
        )
        result = create_visibility_grid(_national_frame(), dem)
        self.assertEqual(result.attrs["units"], "m")
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(np.isfinite(result.values).any())

    def test_plot_masks_to_guangdong_boundary_and_saves_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundary_path = root / "guangdong.geojson"
            boundary_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [{
                            "type": "Feature",
                            "properties": {},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[113, 23], [114, 23], [114, 24], [113, 24], [113, 23]]],
                            },
                        }],
                    }
                ),
                encoding="utf-8",
            )
            nc_path = root / "visibility.nc"
            xr.Dataset(
                {"visibility": (("lat", "lon"), np.ones((3, 3)) * 1000)},
                coords={"lat": [22.5, 23.5, 24.5], "lon": [112.5, 113.5, 114.5]},
            ).to_netcdf(nc_path)
            output = plot_visibility(nc_path, boundary_path, root / "images" / "result.png")
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_visibility_colormap_has_breaks_at_fog_thresholds(self) -> None:
        cmap = _visibility_colormap()
        norm = _visibility_norm()
        self.assertNotEqual(tuple(cmap(norm(0.99))), tuple(cmap(norm(1.01))))
        self.assertNotEqual(tuple(cmap(norm(9.99))), tuple(cmap(norm(10.01))))
        self.assertNotEqual(tuple(cmap(norm(19.99))), tuple(cmap(norm(20.01))))
        self.assertEqual(len(norm.boundaries), 257)

    def test_file_lock_skips_second_holder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.lock"
            with process_lock(path) as first:
                self.assertTrue(first)
                with process_lock(path) as second:
                    self.assertFalse(second)

    def test_run_once_publishes_four_outputs_and_same_counts_skip(self) -> None:
        import src.business.pipeline as pipeline
        from src.business.api import StationBatch

        batch = StationBatch(_national_frame(), _regional_frame(), {"national": 2, "regional": 2})
        original_client = pipeline.VisibilityApiClient

        class FakeClient:
            def __init__(self, settings):
                pass

            def fetch(self, timestamp):
                return batch

        pipeline.VisibilityApiClient = FakeClient
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dem_path = root / "dem.nc"
                xr.Dataset(
                    {"elevation": (("lat", "lon"), np.ones((2, 2)) * 10)},
                    coords={"lat": [23.0, 24.0], "lon": [113.0, 114.0]},
                ).to_netcdf(dem_path)
                boundary_path = root / "guangdong.geojson"
                boundary_path.write_text(
                    json.dumps(
                        {
                            "type": "FeatureCollection",
                            "features": [{
                                "type": "Feature",
                                "properties": {},
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [[[112.9, 22.9], [114.1, 22.9], [114.1, 24.1], [112.9, 24.1], [112.9, 22.9]]],
                                },
                            }],
                        }
                    ),
                    encoding="utf-8",
                )
                data = root / "data"
                config = BusinessConfig(
                    repo_root=root,
                    api=ApiSettings(user_id="u", password="p"),
                    dem_path=dem_path,
                    state_path=data / "business" / "state.sqlite",
                    lock_path=data / "business" / "pipeline.lock",
                    log_path=data / "business" / "business.log",
                    csv_national_root=data / "csv-national",
                    csv_combined_root=data / "csv-combined",
                    nc_national_root=data / "nc-national",
                    nc_combined_root=data / "nc-combined",
                    vis_img_root=data / "vis-img",
                    guangdong_boundary_path=boundary_path,
                    source_ready_delay_minutes=5,
                    max_backfill_slots_per_cycle=4,
                    async_plots=False,
                )
                timestamp = datetime(2026, 8, 5, 8, 2, tzinfo=timezone.utc)
                first = run_once(timestamp, config)
                self.assertEqual(sum(item["status"] == "success" for item in first), 5)
                self.assertEqual(len(list((data / "csv-national").rglob("*.csv"))), 5)
                self.assertEqual(len(list((data / "csv-combined").rglob("*.csv"))), 5)
                self.assertEqual(len(list((data / "nc-national").rglob("*.nc"))), 5)
                self.assertEqual(len(list((data / "nc-combined").rglob("*.nc"))), 5)
                self.assertEqual(len(list((data / "vis-img").rglob("*.png"))), 10)
                close_logging()
                second = run_once(timestamp, config)
                self.assertEqual(sum(item["status"] == "skipped" for item in second), 5)
                close_logging()
        finally:
            close_logging()
            pipeline.VisibilityApiClient = original_client


def _national_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["N1", "N2"], "name": ["n1", "n2"], "city": ["c", "c"], "county": ["x", "x"],
            "lon": [113.0, 114.0], "lat": [23.0, 24.0], "altitude": [10.0, 20.0],
            "rh": [80.0, 90.0], "vis": [1000.0, 2000.0],
        }
    )


def _api_csv(visibility_field: str, code: str, visibility: int) -> str:
    return (
        f"1\nV01301,VF01015_CN,V_CITY,V_COUNTY,V06001,V05001,V07001,V13003,{visibility_field}\n"
        f"{code},站,城市,县,113,23,10,80,{visibility}\n"
    )


def _regional_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["R1", "R2"], "name": ["r1", "r2"], "city": ["c", "c"], "county": ["x", "x"],
            "lon": [113.2, 113.8], "lat": [23.2, 23.8], "altitude": [12.0, 18.0],
            "rh": [82.0, 88.0], "vis": [3000.0, np.nan],
        }
    )


if __name__ == "__main__":
    unittest.main()
