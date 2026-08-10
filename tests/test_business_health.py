from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.business.config import ApiSettings, BusinessConfig
from src.business.health import HealthState, start_health_server


class BusinessHealthTests(unittest.TestCase):
    def test_health_endpoints_report_ready_and_degraded_without_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dem_path = root / "dem.nc"
            dem_path.write_bytes(b"test")
            state = HealthState(_config(root, dem_path))
            state.mark_started()
            server = start_health_server("127.0.0.1", 0, state)
            try:
                status, payload = _request(server.port, "/health/ready")
                self.assertEqual(status, 200)
                self.assertEqual(payload["status"], "degraded")
                self.assertNotIn("secret-password", json.dumps(payload))

                status, payload = _request(server.port, "/health/live")
                self.assertEqual(status, 200)
                self.assertEqual(payload["status"], "live")

                with self.assertRaises(HTTPError) as missing:
                    urlopen(f"http://127.0.0.1:{server.port}/missing")
                self.assertEqual(missing.exception.code, 404)

                with self.assertRaises(HTTPError) as method:
                    urlopen(Request(f"http://127.0.0.1:{server.port}/health/live", method="POST"))
                self.assertEqual(method.exception.code, 405)
            finally:
                server.close()

    def test_readiness_is_unavailable_when_dem_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = HealthState(_config(root, root / "missing.nc"))
            state.mark_started()
            server = start_health_server("127.0.0.1", 0, state)
            try:
                with self.assertRaises(HTTPError) as unavailable:
                    urlopen(f"http://127.0.0.1:{server.port}/health/ready")
                self.assertEqual(unavailable.exception.code, 503)
            finally:
                server.close()


def _request(port: int, path: str) -> tuple[int, dict[str, object]]:
    with urlopen(f"http://127.0.0.1:{port}{path}") as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _config(root: Path, dem_path: Path) -> BusinessConfig:
    data = root / "data"
    return BusinessConfig(
        repo_root=root,
        api=ApiSettings(user_id="user", password="secret-password"),
        dem_path=dem_path,
        state_path=data / "business" / "state.sqlite",
        lock_path=data / "business" / "pipeline.lock",
        log_path=data / "business" / "business.log",
        csv_national_root=data / "csv-national",
        csv_combined_root=data / "csv-combined",
        nc_national_root=data / "nc-national",
        nc_combined_root=data / "nc-combined",
        vis_img_root=data / "images",
        guangdong_boundary_path=data / "missing-boundary.shp",
    )
