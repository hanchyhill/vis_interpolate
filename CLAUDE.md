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

uv run main.py
```

### Running the Tools

**Interactive main program:**
```bash
uv run main.py
# Options:
# 1 - Single DEM interpolation
# 2 - Batch DEM processing and merging
# 3 - Help information
```

**Direct script execution:**
```bash
# Single DEM file interpolation
uv run src/dem_interpolation.py

# Batch processing
uv run src/dem_interpolation.py batch

# Visibility interpolation
uv run src/vis_dem_dis.py

# TPI terrain analysis
uv run src/tpi_ridge_valley.py
```

### Testing
```bash
# Test individual components
uv run test_dem_interpolation.py
uv run test_visibility_interpolation.py
uv run test_batch_processing.py
uv run test_boundary_optimization.py
uv run test_idw_optimization.py

# Debug visualization
uv run debug_visibility_visualization.py

# Model evaluation
uv run src/evaluate_visibility_model.py
uv run test_regional_evaluation.py
```

### Data Processing Pipeline
```bash
# Complete visibility processing workflow:

# Step 1: Estimate visibility from RH data (national & regional stations)
uv run src/get_vis_estimated_by_rh.py

# Step 2: Perform anisotropic IDW interpolation with DEM
uv run src/vis_dem_dis.py

# Step 3: Evaluate model performance
uv run src/evaluate_visibility_model.py
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

The visibility workflow consists of three main scripts:

#### 1. Visibility Estimation ([src/get_vis_estimated_by_rh.py](src/get_vis_estimated_by_rh.py))
- Estimates visibility for regional stations using national station data
- Uses **weighted interpolation based on RH (relative humidity) similarity**
- Key function: `find_nearest_stations()` - finds k nearest neighbors
- Processes time series data from Windows network shares
- Output: CSV files with estimated visibility for all stations

#### 2. Anisotropic IDW Interpolation ([src/vis_dem_dis.py](src/vis_dem_dis.py))
Uses **anisotropic IDW** to account for terrain:

- **Distance function**: `d = sqrt((x-x0)^2 + (y-y0)^2 + β^2(z-z0)^2)`
  - β (default 10.0) amplifies vertical distance influence
  - Higher β = terrain plays bigger role in interpolation

- **Key functions**:
  - `anisotropic_idw_interpolation()` - Core interpolation (fully vectorized)
  - `deg2km_batch()` - Batch distance calculation with broadcasting
  - `create_visibility_grid()` - Main pipeline function

- **Performance**: Fully vectorized implementation achieves **1,364x speedup** over original version
  - Processes 50,000 grid points in ~0.22 seconds (was ~300 seconds)
  - Uses batch processing (default batch_size=10000) to control memory
  - Supports multiprocessing for time series processing

- **Integration with DEM**: Reads merged DEM data to get elevation values for each grid point, enabling 3D-aware interpolation

#### 3. Model Evaluation ([src/evaluate_visibility_model.py](src/evaluate_visibility_model.py))
- Validates interpolated results against observations
- Compares with national 5km CLDAS visibility product (from http://10.148.8.71:7080/thredds/dodsC/)
- Processes time series for comprehensive evaluation
- Output: Evaluation metrics and comparison statistics

### File Naming Conventions

**DEM files:** `ASTGTM2_N{lat:02d}E{lon:03d}_dem.tif`
- Example: `ASTGTM2_N23E111_dem.tif` = 23°N, 111°E

**Station data:** CSV with required columns:
- `lon`, `lat`, `vis` (visibility), `altitude`, `rh` (relative humidity)

**Data Sources:**

1. **National stations:**
   - Path: `\\10.148.44.81\surf\idea\getSurfAutoOrg4Prov\{YYYY}\{MM}\SurfAuto_广东_{YYYYMMDDHHmm}00.csv`
   - Example: `\\10.148.44.81\surf\idea\getSurfAutoOrg4Prov\2021\09\SurfAuto_广东_20210901014500.csv`

2. **Regional stations:**
   - Path: `\\10.148.44.81\surf\idea\getSurfAwstOrg4Prov\{YYYY}\{MM}\SurfAwst_广东_{YYYYMMDDHHmm}00.csv`
   - Example: `\\10.148.44.81\surf\idea\getSurfAwstOrg4Prov\2024\11\SurfAwst_广东_20241101025000.csv`

3. **CLDAS 5km visibility product:**
   - URL: `http://10.148.8.71:7080/thredds/dodsC/cldas/{YYYYMMDD}/VIS_{YYYYMMDDHH}.NC`
   - Example: `http://10.148.8.71:7080/thredds/dodsC/cldas/20251007/VIS_2025100705.NC`

**Output paths** (hardcoded in scripts):
- DEM input: `h:\data\DEM\`
- Temp NetCDF: `h:\data\DEM\netcdf_output\`
- Merged DEM: `h:\data\DEM\merged_dem_data.nc`
- Estimated station data: `data/vis_estimated_base_nation_station/station_vis_all_estimated_{YYYYMMDDHHmm}.csv`
- IDW interpolation output: `data/idw_nc/visibility_anisotropic_idw_{YYYYMMDDHH}.nc`

### Terrain Analysis (TPI)

[src/tpi_ridge_valley.py](src/tpi_ridge_valley.py) implements ridge/valley detection:
- Calculates TPI (Topographic Position Index) using weighted convolution
- Classifies terrain based on TPI thresholds (quantile, zscore, or MAD methods)
- Can export results as vector polygons (requires fiona/shapely)

## Development Notes

### Performance Optimization

The visibility interpolation has been heavily optimized through vectorization:

**Optimization Strategy:**
1. **Vectorized distance calculation** - `deg2km_batch()` replaces millions of individual function calls
2. **Batch processing** - Processes grid points in batches (default 10,000) to balance memory and speed
3. **NumPy broadcasting** - Eliminates Python loops entirely
4. **Multiprocessing** - Parallel processing for time series data

**Performance Gains:**
- Original version: ~300 seconds for 50,000 points
- Optimized version: ~0.22 seconds for 50,000 points
- **Speedup: 1,364x**

**For large-scale processing (1M grid points):**
- Single process: ~4.5 seconds
- 7 processes: ~0.64 seconds

See [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md) for detailed benchmarks.

### Memory Management
- For large files, use `chunk_size` parameter in merge functions
- Example: `chunk_size={'lat': 1000, 'lon': 1000}`
- Adjust batch_size in `anisotropic_idw_interpolation()` (line ~185) if memory constrained
- Default batch_size=10000 uses ~1MB per batch

### Multiprocessing Configuration

```python
# src/vis_dem_dis.py supports parallel time series processing
num_processes = max(1, cpu_count() - 1)  # Auto-configured based on CPU cores
```

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

5. **Visibility estimation workflow:** The complete pipeline requires three steps:
   - First, run `get_vis_estimated_by_rh.py` to estimate visibility for regional stations
   - Second, run `vis_dem_dis.py` to perform terrain-aware interpolation
   - Finally, use `evaluate_visibility_model.py` to validate results

6. **Batch processing for performance:** The visibility interpolation uses batch processing (default 10,000 points) combined with full vectorization. This achieves 1,000x+ speedups while keeping memory usage under control. For time series processing, multiprocessing is automatically enabled.

7. **Data source access:** Scripts expect access to Windows network shares (`\\10.148.44.81\surf\`) and internal THREDDS server (`http://10.148.8.71:7080/thredds/`). Modify paths in source files if working in different environment.
