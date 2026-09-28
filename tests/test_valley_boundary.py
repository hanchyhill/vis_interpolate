import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage as ndi

from src.valley_boundary import (
    DemGrid,
    ValleyConfig,
    merge_connected_valleys,
    compute_enclosure,
    fill_valley_mask,
    fog_thickness,
    fusion_valley_mask,
    hybrid_valley_mask,
    load_dem,
    run,
    tpi_valley_mask,
    unit_valley_mask,
)


def _write_synthetic_dem(path: Path) -> None:
    """左半部为碗状盆地（谷底 50 m，四周山脊 850 m），右半部为 10 m 平原，外圈为海 (0 m)。"""
    lat = np.round(np.arange(23.0, 23.81, 0.01), 2)
    lon = np.round(np.arange(113.0, 114.21, 0.01), 2)
    lon2d, lat2d = np.meshgrid(lon, lat)
    r = np.hypot((lon2d - 113.4) * 100, (lat2d - 23.4) * 100)
    elev = np.where(r < 30, 50 + 800 * (r / 30) ** 2, 850.0)
    elev = np.where(lon2d > 113.75, 10.0, elev)
    elev[:2, :] = 0.0
    xr.Dataset({"elevation": (("lat", "lon"), elev)}, coords={"lat": lat, "lon": lon}).to_netcdf(path)


class ValleyBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dem_path = Path(cls.tmp.name) / "dem.nc"
        _write_synthetic_dem(cls.dem_path)
        cls.grid = load_dem(cls.dem_path, smooth_sigma_px=0)
        cls.basin = (40, 40)
        cls.plain = (40, 100)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fog_thickness_uses_smaller_of_cap_and_relief_fraction(self):
        result = fog_thickness(np.array([150.0, 600.0]), 100.0, 1 / 3)
        np.testing.assert_allclose(result, [50.0, 100.0])

    def test_enclosure_separates_basin_from_plain(self):
        enclosure = compute_enclosure(self.grid, distance_km=40, rise_m=150)
        self.assertEqual(enclosure[self.basin], 8)
        self.assertLess(enclosure[self.plain], 6)
        self.assertEqual(enclosure[0, 0], 0)

    def test_fill_mask_covers_basin_floor_only(self):
        mask = fill_valley_mask(self.grid, ValleyConfig("f", "fill", 10, 100, min_area_km2=1))
        self.assertTrue(mask[self.basin])
        self.assertFalse(mask[self.plain])
        self.assertFalse(mask[40, 60])  # 850 m 山脊

    def test_unit_fog_top_follows_thickness_rule(self):
        cfg = ValleyConfig("u", "unit", 10, 100, min_area_km2=1)
        mask, units, table = unit_valley_mask(self.grid, cfg)
        self.assertTrue(mask[self.basin])
        self.assertFalse(mask[self.plain])
        self.assertEqual(len(table), 1)
        row = table.iloc[0]
        expected = min(100.0, (row.peak_p95_m - row.valley_mean_m) / 3)
        self.assertAlmostEqual(row.fog_thickness_m, expected)
        self.assertLessEqual(self.grid.elevation[mask].max(), row.fog_top_m + 1e-6)
        self.assertEqual(set(np.unique(units[mask])), {int(row.unit_id)})

    def test_hybrid_keeps_enclosed_tpi_valley_and_drops_plain(self):
        cfg = ValleyConfig("h", "hybrid", 5, 10, min_area_km2=1)
        mask, labels = hybrid_valley_mask(self.grid, cfg)
        self.assertTrue(mask[self.basin])
        self.assertFalse(mask[self.plain])
        self.assertTrue(np.array_equal(mask, labels > 0))

    def test_fusion_uses_unit_blocks_and_stays_near_unit_fog_area(self):
        cfg = ValleyConfig("fu", "fusion", 5, 10, min_area_km2=1)
        mask, labels = fusion_valley_mask(self.grid, cfg)
        unit_mask, _, _ = unit_valley_mask(self.grid, ValleyConfig("u", "unit", 10, 100, min_area_km2=1))
        self.assertTrue(mask[self.basin])
        self.assertFalse(mask[self.plain])
        self.assertEqual(len(np.unique(labels[mask])), 1)
        reach = ndi.binary_dilation(unit_mask, structure=np.ones((3, 3)), iterations=cfg.reach_px)
        self.assertFalse((mask & ~reach).any())

    def test_merge_connected_valleys_uses_saddle_and_area_cap(self):
        elevation = np.full((3, 4), 100.0)
        grid = DemGrid(elevation, np.ones((3, 4), bool), np.array([23.0, 23.01, 23.02]),
                       np.array([113.0, 113.01, 113.02, 113.03]), 1.0, 1.0)
        labels = np.array([[1, 1, 2, 2]] * 3, dtype=np.int32)
        merged = merge_connected_valleys(labels, grid, np.array([-np.inf, 150.0, 150.0]), 0, np.inf)
        self.assertEqual(len(np.unique(merged)), 1)
        high_saddle = merge_connected_valleys(labels, grid, np.array([-np.inf, 50.0, 50.0]), 0, np.inf)
        self.assertEqual(len(np.unique(high_saddle)), 2)
        slack = merge_connected_valleys(labels, grid, np.array([-np.inf, 50.0, 50.0]), 50, np.inf)
        self.assertEqual(len(np.unique(slack)), 1)
        capped = merge_connected_valleys(labels, grid, np.array([-np.inf, 150.0, 150.0]), 0, 8.0)
        self.assertEqual(len(np.unique(capped)), 2)

    def test_tpi_min_area_removes_fragments(self):
        small = tpi_valley_mask(self.grid, ValleyConfig("t", "tpi", 5, 10, min_area_km2=1))
        large = tpi_valley_mask(self.grid, ValleyConfig("t", "tpi", 5, 10, min_area_km2=10_000))
        self.assertTrue(small.any())
        self.assertFalse(large.any())

    def test_run_writes_netcdf_with_masks(self):
        out_dir = Path(self.tmp.name) / "out"
        configs = (ValleyConfig("fill_test", "fill", 10, 100, min_area_km2=1),
                   ValleyConfig("unit_test", "unit", 10, 100, min_area_km2=1))
        summary = run(self.dem_path, None, out_dir, configs)
        self.assertEqual(list(summary["config"]), ["fill_test", "unit_test"])
        with xr.open_dataset(out_dir / "valley_boundaries.nc") as ds:
            self.assertIn("valley_fill_test", ds)
            self.assertEqual(ds["valley_unit_test"].dims, ("lat", "lon"))
            self.assertGreater(int((ds["valley_fill_test"] > 0).sum()), 0)
        for name in ("summary.csv", "compare_fill.png", "compare_unit.png", "valley_polygons.gpkg"):
            self.assertTrue((out_dir / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
