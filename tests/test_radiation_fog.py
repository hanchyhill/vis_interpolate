from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.business.algorithms import _estimate_one, _estimate_one_legacy, estimate_both
from src.business.api import StationBatch, _parse_response
from src.business.config import ApiSettings, BusinessConfig, RadiationFogSettings
from src.business.idw import create_visibility_grid
from src.business.radiation_fog import (
    FOG_INFERRED,
    FOG_OBSERVED,
    RadiationFogModel,
    ValleyProduct,
    build_corrected_products,
)

LAT = np.round(np.arange(23.0, 23.505, 0.01), 2)   # 51 行
LON = np.round(np.arange(113.0, 113.805, 0.01), 2)  # 81 列
FOG_TOP = 200.0


def _cell(row: int, col: int) -> tuple[float, float]:
    return float(LON[col]), float(LAT[row])


def _write_inputs(directory: Path) -> tuple[Path, xr.Dataset]:
    """两个矩形山谷：5 号 (行 10-40, 列 5-25) 与 9 号 (行 10-40, 列 55-75)，雾顶 200 m。

    谷内海拔 100 m，谷外 150 m（低于雾顶），因此谷外 g 只随水平距离衰减。
    """
    valley = np.zeros((LAT.size, LON.size), dtype=np.int32)
    valley[10:41, 5:26] = 5
    valley[10:41, 55:76] = 9
    elevation = np.where(valley > 0, 100.0, 150.0)
    path = directory / "valley.nc"
    xr.Dataset(
        {
            "valley_id": (("lat", "lon"), valley),
            "fog_top_m": ("valley", [FOG_TOP, FOG_TOP]),
            "valley_mean_m": ("valley", [100.0, 100.0]),
        },
        coords={"lat": LAT, "lon": LON, "valley": [5, 9]},
    ).to_netcdf(path)
    dem = xr.Dataset({"elevation": (("lat", "lon"), elevation)}, coords={"lat": LAT, "lon": LON})
    return path, dem


def _stations(rows: list[tuple]) -> pd.DataFrame:
    """rows: (code, 行, 列, 海拔, 相对湿度, 能见度, 1 小时降水)。"""
    records = []
    for code, row, col, altitude, rh, vis, pre in rows:
        lon, lat = _cell(row, col)
        records.append({"code": code, "name": code, "city": "c", "county": "x", "lon": lon, "lat": lat,
                        "altitude": altitude, "rh": rh, "vis": vis, "pre_1h": pre})
    return pd.DataFrame(records)


GOOD = [  # 远离山谷的能见度良好站
    ("G1", 2, 40, 150.0, 80.0, 10000.0, 0.0),
    ("G2", 48, 40, 150.0, 80.0, 10000.0, 0.0),
    ("G3", 25, 45, 150.0, 80.0, 10000.0, 0.0),
    ("G4", 2, 2, 150.0, 80.0, 10000.0, 0.0),
]


class RadiationFogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.valley_path, self.dem = _write_inputs(Path(self.tmp.name))
        self.settings = RadiationFogSettings(enabled=True, valley_path=self.valley_path)
        self.model = RadiationFogModel.load(self.settings, self.dem)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _model(self, **changes) -> RadiationFogModel:
        return RadiationFogModel.load(dataclasses.replace(self.settings, **changes), self.dem)

    def test_station_valley_excludes_mountain_top_and_outside(self) -> None:
        frame = _stations([("A", 20, 15, 100.0, 90, 1, 0), ("B", 20, 15, 300.0, 90, 1, 0),
                           ("C", 20, 40, 100.0, 90, 1, 0), ("D", 20, 60, 150.0, 90, 1, 0)])
        valley = self.model.product.station_valley(frame.lon, frame.lat, frame.altitude)
        np.testing.assert_array_equal(valley, [5, 0, 0, 9])

    def test_fog_requires_valley_low_visibility_and_no_precipitation(self) -> None:
        reference = _stations([
            ("F1", 20, 15, 100.0, 98, 200.0, 0.0),
            ("WET", 20, 60, 100.0, 98, 200.0, 0.5),
            ("PLAIN", 25, 40, 150.0, 98, 200.0, 0.0),
            ("MISS", 30, 15, 100.0, 98, 300.0, np.nan),
            *GOOD,
        ])
        flags = dict(zip(reference.code, self.model.analyze(reference, reference.iloc[:0]).reference.is_radiation_fog))
        self.assertEqual(flags["F1"], FOG_OBSERVED)
        self.assertEqual(flags["WET"], 0)
        self.assertEqual(flags["PLAIN"], 0)
        self.assertEqual(flags["MISS"], FOG_OBSERVED)
        wet_missing = self._model(precip_missing_as_dry=False).analyze(reference, reference.iloc[:0])
        self.assertEqual(dict(zip(wet_missing.reference.code, wet_missing.reference.is_radiation_fog))["MISS"], 0)
        self.assertEqual(wet_missing.summary["low_vis_in_valley_precip_missing"], 1)

    def test_influence_is_one_in_domain_and_decays_outside(self) -> None:
        reference = _stations([("F1", 25, 15, 100.0, 98, 200.0, 0.0), *GOOD])
        influence = self.model.analyze(reference, reference.iloc[:0]).influence
        g = influence.max_field()
        self.assertEqual(g[25, 15], 1.0)
        self.assertEqual(g[25, 25], 1.0)
        distance_km = 2 * self.model.product.dx_km
        self.assertAlmostEqual(float(g[25, 27]), np.exp(-distance_km / 2.0), places=5)
        self.assertEqual(g[25, 40], 0.0)
        self.assertEqual(g[25, 65], 0.0)

    def test_vertical_decay_uses_height_above_fog_top(self) -> None:
        reference = _stations([("F1", 25, 15, 100.0, 98, 200.0, 0.0)])
        influence = self.model.analyze(reference, reference.iloc[:0]).influence
        lon, lat = _cell(25, 15)
        domain = np.array([[5, 5]])
        g = influence.station_factor(domain, np.array([lon]), np.array([lat]), np.array([150.0]))
        np.testing.assert_allclose(g, [[1.0, 1.0]])
        g = influence.station_factor(domain, np.array([lon]), np.array([lat]), np.array([FOG_TOP + 50.0]))
        np.testing.assert_allclose(g, [[np.exp(-1.0)] * 2])
        g = influence.station_factor(domain, np.array([lon]), np.array([lat]), np.array([FOG_TOP + 200.0]))
        np.testing.assert_array_equal(g, [[0.0, 0.0]])

    def test_voronoi_keeps_only_part_closer_to_fog_station(self) -> None:
        reference = _stations([("F1", 12, 15, 100.0, 98, 200.0, 0.0),
                               ("OK", 38, 15, 100.0, 80, 15000.0, 0.0), *GOOD])
        influence = self.model.analyze(reference, reference.iloc[:0]).influence
        lon = np.array([_cell(13, 15)[0], _cell(37, 15)[0]])
        lat = np.array([_cell(13, 15)[1], _cell(37, 15)[1]])
        np.testing.assert_array_equal(influence.in_domain(np.array([5, 5]), lon, lat), [True, False])

    def test_virtual_fog_station_in_valley_without_visibility(self) -> None:
        reference = _stations([("F1", 20, 15, 100.0, 98, 200.0, 0.0),
                               ("F2", 30, 15, 100.0, 98, 400.0, 0.0), *GOOD])
        targets = _stations([("T1", 25, 56, 120.0, 97.0, np.nan, 0.0),
                             ("T2", 30, 65, 120.0, 90.0, np.nan, 0.0),
                             ("T3", 35, 65, 120.0, 99.0, np.nan, 1.0)])
        analysis = self.model.analyze(reference, targets)
        flags = dict(zip(analysis.targets.code, analysis.targets.is_radiation_fog))
        self.assertEqual(flags, {"T1": FOG_INFERRED, "T2": 0, "T3": 0})
        self.assertEqual(analysis.targets.set_index("code").loc["T1", "virtual_vis"], 300.0)
        self.assertEqual(set(analysis.influence.domains), {5, 9})
        corrected = self.model.estimate(reference, targets).frame.set_index("code")
        self.assertEqual(corrected.loc["T1", "vis"], 300.0)
        self.assertEqual(corrected.loc["T1", "fog_domain_id"], 9)

        far = self._model(infer_radius_km=10).analyze(reference, targets)
        self.assertFalse((far.targets.is_radiation_fog == FOG_INFERRED).any())

    def test_estimate_avoids_fog_station_outside_domain(self) -> None:
        reference = _stations([("F1", 25, 20, 100.0, 98, 200.0, 0.0), *GOOD])
        targets = _stations([("OUT", 25, 35, 150.0, 98.0, np.nan, 0.0),
                             ("IN", 25, 18, 100.0, 98.0, np.nan, 0.0)])
        original = _estimate_one(reference, targets).set_index("code")
        corrected = self.model.estimate(reference, targets).frame.set_index("code")
        self.assertLess(original.loc["OUT", "vis"], 10000.0)
        self.assertAlmostEqual(corrected.loc["OUT", "vis"], 10000.0)
        self.assertEqual(corrected.loc["IN", "vis"], original.loc["IN", "vis"])
        self.assertLess(corrected.loc["IN", "vis"], 500.0)
        self.assertEqual(corrected.loc["IN", "fog_domain_id"], 5)
        self.assertEqual(corrected.loc["OUT", "fog_domain_id"], 0)
        self.assertEqual(corrected.loc["F1", "is_radiation_fog"], FOG_OBSERVED)

    def test_idw_fog_station_has_no_weight_far_away(self) -> None:
        reference = _stations([("F1", 25, 15, 100.0, 98, 200.0, 0.0), *GOOD])
        corrected = self.model.estimate(reference, reference.iloc[:0])
        stations = corrected.frame
        with_fog = create_visibility_grid(stations, self.dem, fog_influence=corrected.influence).values
        original = create_visibility_grid(stations, self.dem).values
        without = create_visibility_grid(stations[stations.code != "F1"], self.dem).values
        self.assertAlmostEqual(with_fog[25, 50], without[25, 50])
        self.assertLess(original[25, 50], without[25, 50])
        self.assertEqual(with_fog[25, 15], original[25, 15])

    def test_no_fog_matches_original_bitwise(self) -> None:
        national = _stations(GOOD)
        regional = _stations([("R1", 25, 15, 100.0, 97.0, np.nan, 0.0), ("R2", 20, 60, 100.0, 60.0, 8000.0, 0.0)])
        estimates = estimate_both(national, regional)
        grids = {source: create_visibility_grid(frame, self.dem) for source, frame in estimates.items()}
        products = build_corrected_products(national, regional, self.dem, self.model, estimates, grids)
        for source, product in products.items():
            np.testing.assert_array_equal(product.dataset["visibility"].values, grids[source].values)
            np.testing.assert_array_equal(product.frame["vis"].to_numpy(), estimates[source]["vis"].to_numpy())
            self.assertEqual(float(product.dataset["fog_influence"].max()), 0.0)

    def test_vectorized_estimate_matches_legacy(self) -> None:
        rng = np.random.default_rng(1)
        reference = pd.DataFrame({
            "code": [f"N{i}" for i in range(60)],
            "lon": np.round(rng.uniform(113, 114, 60), 2), "lat": np.round(rng.uniform(23, 24, 60), 2),
            "altitude": rng.uniform(0, 500, 60), "rh": rng.integers(60, 100, 60).astype(float),
            "vis": rng.uniform(100, 30000, 60),
        })
        reference.loc[5:9, ["lon", "lat"]] = reference.loc[0:4, ["lon", "lat"]].to_numpy()
        targets = reference.drop(columns="vis").assign(code=lambda f: "R" + f.code.str[1:])
        targets = pd.concat([targets, targets.assign(code=lambda f: f.code + "b", lon=113.5)], ignore_index=True)
        new = _estimate_one(reference, targets)
        old = _estimate_one_legacy(reference, targets)
        self.assertEqual(new.code.tolist(), old.code.tolist())
        for column in ("vis", "vis_rh", "vis_dis", "is_vis_est"):
            np.testing.assert_array_equal(new[column].to_numpy(float), old[column].to_numpy(float))

    def test_parser_maps_precipitation_and_tolerates_missing_field(self) -> None:
        fields = ["V01301", "VF01015_CN", "V_CITY", "V_COUNTY", "V06001", "V05001", "V07001", "V13003", "V20001"]
        header = ",".join(fields)
        with_precip = _parse_response(
            f"2\n{header},V13019\nA,a,c,x,113,23,10,90,300,0.5\nB,b,c,x,113,23,10,90,300,9999\n", fields, "V20001"
        ).set_index("code")
        self.assertEqual(with_precip.loc["A", "pre_1h"], 0.5)
        self.assertTrue(np.isnan(with_precip.loc["B", "pre_1h"]))
        without = _parse_response(f"1\n{header}\nA,a,c,x,113,23,10,90,300\n", fields, "V20001")
        self.assertTrue(without["pre_1h"].isna().all())

    def test_config_reads_radiation_fog_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "config.json"
            path.write_text(json.dumps({"userId": "u", "pwd": "p", "radiationFog": {
                "enabled": True, "valleyPath": "v.nc", "sigmaDKm": 3, "rhThresholdPct": 97,
                "precipMissingAsDry": False,
            }}), encoding="utf-8")
            settings = BusinessConfig.from_file(path, repo_root=root).radiation_fog
            self.assertTrue(settings.enabled)
            self.assertEqual(settings.valley_path, (root / "v.nc").resolve())
            self.assertEqual((settings.sigma_d_km, settings.rh_threshold_pct), (3.0, 97.0))
            self.assertFalse(settings.precip_missing_as_dry)
            self.assertEqual(settings.sigma_z_m, 50.0)
            path.write_text(json.dumps({"userId": "u", "pwd": "p"}), encoding="utf-8")
            self.assertFalse(BusinessConfig.from_file(path, repo_root=root).radiation_fog.enabled)

    def test_build_outputs_writes_original_and_corrected_fields(self) -> None:
        from src.business import pipeline

        root = Path(self.tmp.name)
        dem_path = root / "dem.nc"
        self.dem.to_netcdf(dem_path)
        data = root / "data"
        config = BusinessConfig(
            repo_root=root, api=ApiSettings(user_id="u", password="p"), dem_path=dem_path,
            state_path=data / "state.sqlite", lock_path=data / "lock", log_path=data / "log.txt",
            csv_national_root=data / "csv-n", csv_combined_root=data / "csv-c",
            nc_national_root=data / "nc-n", nc_combined_root=data / "nc-c", vis_img_root=data / "img",
            guangdong_boundary_path=root / "missing.shp", async_plots=False, radiation_fog=self.settings,
        )
        data.mkdir(parents=True)
        national = _stations([("F1", 25, 15, 100.0, 98, 200.0, 0.0), *GOOD])
        regional = _stations([("R1", 25, 35, 150.0, 98.0, np.nan, 0.0), ("R2", 25, 65, 120.0, 97.0, np.nan, 0.0)])
        batch = StationBatch(national, regional, {"national": 5, "regional": 2})
        timings: dict = {}
        outputs = pipeline._build_outputs(datetime(2026, 1, 5, 22, 0, tzinfo=timezone.utc), batch, config, timings)
        nc_path = next(Path(p) for p in outputs if p.endswith(".nc") and "nc-c" in p)
        with xr.open_dataset(nc_path) as ds:
            self.assertEqual(set(ds.data_vars), {"visibility", "visibility_original", "fog_influence"})
            self.assertEqual(ds["visibility"].attrs["units"], "m")
            self.assertEqual(float(ds["fog_influence"].max()), 1.0)
        csv_path = next(Path(p) for p in outputs if p.endswith(".csv") and "csv-c" in p)
        frame = pd.read_csv(csv_path)
        for column in ("valley_id", "is_radiation_fog", "fog_domain_id", "vis_original"):
            self.assertIn(column, frame)
        self.assertEqual(timings["radiation_fog"]["national_and_regional"]["fog_observed"], 1)
        pipeline.close_logging()


if __name__ == "__main__":
    unittest.main()
