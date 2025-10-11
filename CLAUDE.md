# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python tool for geospatial data interpolation and processing, focused on:
1. **DEM (Digital Elevation Model) data** - Downsampling high-resolution DEM data to 0.01°×0.01° grids
2. **Visibility interpolation** - Anisotropic Inverse Distance Weighting (IDW) for meteorological station visibility data
3. **Terrain analysis** - TPI (Topographic Position Index) based ridge/valley classification

Python version: >=3.13

## Common Commands

### Environment Setup
```bash
# The project uses a virtual environment at .venv/
# Activate it if needed (it may auto-activate)
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Dependencies are managed via pyproject.toml
# Install/update dependencies (if using pip):
pip install -e .
```

### Running the Tools

**Interactive main program:**
```bash
python main.py
# Options:
# 1 - Single DEM interpolation
# 2 - Batch DEM processing and merging
# 3 - Help information
```

**Direct script execution:**
```bash
# Single DEM file interpolation
python src/dem_interpolation.py

# Batch processing
python src/dem_interpolation.py batch

# Visibility interpolation
python src/vis_dem_dis.py

# TPI terrain analysis
python src/tpi_ridge_valley.py
```

### Testing
```bash
# Test individual components
python test_dem_interpolation.py
python test_visibility_interpolation.py
python test_batch_processing.py
python test_boundary_optimization.py

# Debug visualization
python debug_visibility_visualization.py
```

## Architecture and Key Concepts

### DEM Processing Pipeline

The DEM workflow consists of three main phases:

1. **Single-file interpolation** ([src/dem_interpolation.py](src/dem_interpolation.py))
   - Loads GeoTIFF DEM files (format: `ASTGTM2_NyyExxx_dem.tif`)
   - Creates 0.01° resolution target grids
   - Multiple interpolation methods available (vectorized is fastest)
   - Outputs NetCDF format

2. **Batch processing** ([src/dem_interpolation.py](src/dem_interpolation.py):596-683)
   - Scans directory for multiple DEM tiles
   - Parses filenames to extract lat/lon (`parse_dem_filename()`)
   - Processes each tile independently
   - Saves intermediate NetCDF files to `output_dir`

3. **Boundary-optimized merging** ([src/dem_interpolation.py](src/dem_interpolation.py):978-1109)
   - **Critical:** The `create_unified_grid()` function handles merging with advanced boundary processing
   - Fills gaps between DEM tiles using linear/nearest interpolation
   - Smooths transitions at tile boundaries using Gaussian filtering
   - Handles overlapping regions with weighted averaging
   - This is crucial because raw DEM tiles often have gaps at boundaries

**Key functions:**
- `interpolate_dem_to_grid()` - Core interpolation with multiple method support
- `batch_process_and_merge()` - Complete pipeline from TIF files to merged NetCDF
- `fill_boundary_gaps()` - Fills missing values at tile boundaries
- `smooth_boundaries()` - Applies Gaussian smoothing to reduce discontinuities

### Interpolation Methods

Performance comparison (fastest to slowest):
1. `vectorized` - Uses xarray's native interpolation (RECOMMENDED)
2. `fast_nearest` - Custom nearest-neighbor via direct indexing
3. `block_average` - Block averaging for downsampling
4. `nearest`, `linear`, `cubic` - scipy.interpolate.griddata methods

### Visibility Interpolation

The visibility workflow ([src/vis_dem_dis.py](src/vis_dem_dis.py)) uses **anisotropic IDW** to account for terrain:

- **Distance function**: `d = sqrt((x-x0)^2 + (y-y0)^2 + β^2(z-z0)^2)`
  - β (default 10.0) amplifies vertical distance influence
  - Higher β = terrain plays bigger role in interpolation

- **Key function**: `anisotropic_idw_interpolation()` (line 59+)
  - Uses sklearn's NearestNeighbors for efficient spatial queries
  - Converts lat/lon to km via `deg2km()` for proper Euclidean distance
  - Default: 6 nearest neighbors

- **Integration with DEM**: Reads merged DEM data to get elevation values for each grid point, enabling 3D-aware interpolation

### File Naming Conventions

**DEM files:** `ASTGTM2_N{lat:02d}E{lon:03d}_dem.tif`
- Example: `ASTGTM2_N23E111_dem.tif` = 23°N, 111°E

**Station data:** CSV with required columns:
- `lon`, `lat`, `vis` (visibility), `altitude`, `rh` (relative humidity)

**Output paths** (hardcoded in scripts):
- Input: `h:\data\DEM\`
- Temp NetCDF: `h:\data\DEM\netcdf_output\`
- Merged: `h:\data\DEM\merged_dem_data.nc`
- Station data: `data/station_vis_all_estimated.csv`

### Terrain Analysis (TPI)

[src/tpi_ridge_valley.py](src/tpi_ridge_valley.py) implements ridge/valley detection:
- Calculates TPI (Topographic Position Index) using weighted convolution
- Classifies terrain based on TPI thresholds (quantile, zscore, or MAD methods)
- Can export results as vector polygons (requires fiona/shapely)

## Development Notes

### Memory Management
- For large files, use `chunk_size` parameter in merge functions
- Example: `chunk_size={'lat': 1000, 'lon': 1000}`
- Adjust based on available RAM

### Coordinate Systems
- All inputs must use geographic coordinates (lat/lon)
- Internal processing maintains consistent WGS84/geographic projection
- Output NetCDF files include coordinate metadata

### Visualization
- Uses matplotlib with cartopy for geographic plots
- Chinese font support configured: `plt.rcParams['font.sans-serif'] = ['SimHei']`
- Cartopy requires coastline data (downloads automatically on first use)

### Error Handling
- Boundary optimization is critical for multi-tile DEM merging
- Enable with `enable_boundary_optimization=True` in `merge_netcdf_files_optimized()`
- If traditional xarray merge fails, falls back to custom `create_unified_grid()`

## Important Implementation Details

1. **DEM tile boundaries:** Real-world DEM tiles often have 1-2 pixel gaps at edges. The boundary optimization algorithms (fill + smooth) are essential for seamless merged datasets.

2. **Anisotropic IDW β parameter:** The vertical weighting factor β=10 means elevation differences are amplified 10x compared to horizontal distance. This reflects how atmospheric visibility varies more strongly with altitude than horizontal distance.

3. **Interpolation method selection:** Always use 'vectorized' for production unless you need specific interpolation characteristics (e.g., cubic for smoothness). It's 10-100x faster than scipy methods.

4. **Coordinate precision:** The code uses tolerance-based coordinate matching when merging grids to handle floating-point precision issues.
