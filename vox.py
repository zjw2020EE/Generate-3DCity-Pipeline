"""
Script to generate VOX data for specified cities.

## Usage:
```bash python vox.py

## Output:
- VOX data files in the 'tmp/outputs' directory.

## Dependencies:
- voxcity.generator

## Note:
- The script expects a configuration file 'city_roi.json' in the 'tmp' directory.
- The script expects a GeoJSON file 'tmp.geojson' in the 'tmp' directory.
- The source code of Voxcity should be replaced by the modified version:
    geoprocessor/heights.py
    geoprocessor/GEOJSON
    geoprocessor/city_sample.py
    geoprocessor/__init__.py
    geoprocessor/draw.py
    geoprocessor/raster/buildings.py
    generator/grids.py
    downloader/osm.py

## Author: Garvin Z
## Date: 2025-12
"""

import ee
import numpy as np
import os
import sys
import json
import geopandas as gpd
from voxcity_custom.generator import (
    get_building_height_grid, 
    get_land_cover_grid, 
    get_canopy_height_grid, 
    get_dem_grid
)

# ---- Set Earth Engine Project ID ----
EE_PROJECT_ID = "your-project-id"

# ---- 1. Initialize Earth Engine ----
# Try to initialize, if it fails, authenticate.
try:
    ee.Initialize(project=EE_PROJECT_ID)
    print("[INFO] Earth Engine initialized successfully.")
except Exception as e:
    print(f"[WARN] Initialization failed: {e}")
    print("[INFO] Attempting authentication...")
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT_ID)

# ---- 2. Setup Paths ----
current_dir = os.path.dirname(os.path.abspath(__file__))
tmp_dir = os.path.join(current_dir, "tmp")
json_path = os.path.join(tmp_dir, "city_roi.json")
geojson_path = os.path.join(tmp_dir, "tmp.geojson")
output_dir = os.path.join(tmp_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

# Check if files exist
if not os.path.exists(json_path):
    raise FileNotFoundError(f"Config file not found: {json_path}")
if not os.path.exists(geojson_path):
    raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

# ---- 3. Read Configuration ----
print(f"[INFO] Reading configuration from: {json_path}")
with open(json_path, 'r', encoding='utf-8') as f:
    config_data = json.load(f)
    # voxcity expects vertices (list of lists)
    rectangle_vertices = config_data["rectangle_vertices"]
    city_name = config_data.get("city_name", "Unknown")

print(f"[INFO] Processing city: {city_name}")

# ---- 4. Load Buildings GeoDataFrame ----
print(f"[INFO] Loading buildings from: {geojson_path}")
buildings_gdf = gpd.read_file(geojson_path)

# ---- 5. Define Voxcity Parameters ----
# Set data sources and meshsize (m)
meshsize = 2
# building_source = 'OpenStreetMap' #@param ['OpenStreetMap', 'Global Building Atlas', 'Overture', 'EUBUCCO v0.1', 'Open Building 2.5D Temporal', 'Microsoft Building Footprints', 'Local file']
# building_complementary_source = "Microsoft Building Footprints" #@param ['None', 'Global Building Atlas', 'Open Building 2.5D Temporal', 'Microsoft Building Footprints', 'England 1m DSM - DTM', 'Netherlands 0.5m DSM - DTM', 'OpenMapTiles', 'Local file', 'OpenStreetMap', 'Overture', 'EUBUCCO v0.1']
# land_cover_source = 'OpenStreetMap' #@param ['OpenStreetMap', 'Urbanwatch', 'OpenEarthMapJapan', 'ESA WorldCover', 'ESRI 10m Annual Land Cover', 'Dynamic World V1']
# canopy_height_source = 'High Resolution 1m Global Canopy Height Maps' #@param ['High Resolution 1m Global Canopy Height Maps', 'ETH Global Sentinel-2 10m Canopy Height (2020)', 'Static']
# dem_source = 'USGS 3DEP 1m' #@param ['DeltaDTM', 'FABDEM', 'England 1m DTM', 'DEM France 1m', 'Netherlands 0.5m DTM', 'AUSTRALIA 5M DEM', 'USGS 3DEP 1m', 'NASA', 'COPERNICUS', 'Flat']
kwargs = {
    # "building_complement_height": 10,
    # "building_complementary_source": "OpenStreetMap",
    # "complement_building_footprints": True,
    "gridvis": False,
    "mapvis": False,
    "voxelvis": False,
    "voxelvis_img_save_path": None,
    "trunk_height_ratio": None,
    "min_canopy_height": None,
    "dem_interpolation": True,
}

# ---- 6. Generate Grids ----

# A. Building Height Grid (Source: Local GeoDataFrame)
print("[INFO] Generating Building Height Grid...")
building_height_grid = get_building_height_grid(
    rectangle_vertices=rectangle_vertices,
    meshsize=meshsize,
    source="GeoDataFrame",  
    building_gdf=buildings_gdf,
    output_dir=output_dir,
    **kwargs
)
np.save(os.path.join(output_dir, "building_height.npy"), building_height_grid[0])
gdf = building_height_grid[3]
gdf.to_file(os.path.join(output_dir, "building.geojson"), driver='GeoJSON')

# B. Land Cover Grid (Source: ESA WorldCover)
print("[INFO] Generating Land Cover Grid (ESA)...")
land_cover_source = 'ESA WorldCover'
land_cover_grid = get_land_cover_grid(
    rectangle_vertices=rectangle_vertices,
    meshsize=meshsize,
    source=land_cover_source,
    output_dir=output_dir,
    **kwargs
)
np.save(os.path.join(output_dir, "land_cover_esa.npy"), land_cover_grid.astype(np.uint8))

# C. Canopy Height Grid (Source: High Resolution 1m)
print("[INFO] Generating Canopy Height Grid...")
canopy_height_source = 'High Resolution 1m Global Canopy Height Maps'
canopy_height_grid = get_canopy_height_grid(
    rectangle_vertices=rectangle_vertices,
    meshsize=meshsize,
    source=canopy_height_source,
    output_dir=output_dir,
    **kwargs
)
np.save(os.path.join(output_dir, "canopy_height_top.npy"), canopy_height_grid[0])
np.save(os.path.join(output_dir, "canopy_height_bottom.npy"), canopy_height_grid[1])

# D. DEM Grid (Source: FABDEM)
print("[INFO] Generating DEM Grid...")
dem_source = 'FABDEM'
dem_grid = get_dem_grid(
    rectangle_vertices=rectangle_vertices,
    meshsize=meshsize,
    source=dem_source,
    output_dir=output_dir,
    **kwargs
)
crop_dem_grid = dem_grid - dem_grid.min()
np.save(os.path.join(output_dir, "dem.npy"), dem_grid)
np.save(os.path.join(output_dir, "crop_dem.npy"), crop_dem_grid)

# E. Land Use Grid (Source: OpenStreetMap)
# Note: Re-using get_land_cover_grid with a different source as per your request
print("[INFO] Generating Land Use Grid (OSM)...")
land_use_source = 'OpenStreetMap'
land_use_grid = get_land_cover_grid(
    rectangle_vertices=rectangle_vertices,
    meshsize=meshsize,
    source=land_use_source,
    output_dir=output_dir,
    **kwargs
)
land_use_grid[land_use_grid == 12] = 10
land_use_grid[building_height_grid[0] > 0.0] = 12
np.save(os.path.join(output_dir, "land_use_osm.npy"), land_use_grid.astype(np.uint8))

print("------------------------------------------------")
print(f"[SUCCESS] All grids generated and saved to: {tmp_dir}")