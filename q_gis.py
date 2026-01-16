""" 
Script to extract building footprints directly to GeoJSON using QGIS API and GeoPandas.
(No intermediate Shapefile created)

## Usage:
```bash setlocal EnableExtensions EnableDelayedExpansion
```bash set "QGIS_ROOT={your path to QGIS 3.40.11}" (set "QGIS_ROOT=D:\Programs\QGIS 3.40.11")
```bash set "PYQGIS_BAT=%QGIS_ROOT%\bin\python-qgis-ltr.bat"
```bash call "%PYQGIS_BAT%" q_gis.py

## Output:
- A GeoJSON file in the 'tmp/outputs' directory.

## Dependencies:
- QGIS Python API
- Geopandas, Pandas, Shapely

## Author: Garvin Z (Modified)
## Date: 2025-12
"""
# -*- coding: utf-8 -*-
import pathlib
import os
import sys
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt 

from qgis.core import (
    QgsApplication, QgsCoordinateReferenceSystem, QgsCoordinateTransform,
    QgsProject, QgsVectorLayer, QgsPointXY, QgsGeometry,
    QgsFeatureRequest
)

# ---- Initialize QGIS (Headless) ----
QGIS_ROOT = os.environ.get("QGIS_ROOT", r"D:\Programs\QGIS 3.40.11")
QgsApplication.setPrefixPath(QGIS_ROOT, True)
qgs = QgsApplication([], False)
qgs.initQgis()

city2p = {
    "Shenzhen": "g",
    "Chongqing": "c",
    "Chengdu": "s",
    "Suzhou": "j",
    "Hangzhou": "z",
    "Hefei": "a"
}

def export_features_to_geojson(layer, features, output_path):
    """
    Directly converts QGIS features to a GeoPandas GeoDataFrame in memory,
    cleans the data (rename Height, reset ID), and exports to GeoJSON.
    """
    print(f"[INFO] Processing {len(features)} features for GeoJSON export...")

    if not features:
        print("[WARN] No features to export.")
        return

    # 1. Extract Attributes and Geometry from QGIS features
    # Get field names
    field_names = [field.name() for field in layer.fields()]
    
    data_list = []
    geometry_list = []

    for f in features:
        # Extract Attributes
        attrs = f.attributes()
        attr_dict = dict(zip(field_names, attrs))
        data_list.append(attr_dict)

        # Extract Geometry (Convert QGIS Geometry -> WKT String -> Shapely Object)
        # This is the bridge between QGIS and GeoPandas
        geom_wkt = f.geometry().asWkt()
        geometry_list.append(wkt.loads(geom_wkt))

    # 2. Create GeoDataFrame
    df = pd.DataFrame(data_list)
    
    # Get Layer CRS
    crs_wkt = layer.crs().toWkt()
    
    gdf = gpd.GeoDataFrame(df, geometry=geometry_list, crs=crs_wkt)

    # 3. Data Cleaning
    # Rename 'Height' to 'height' if exists
    if 'Height' in gdf.columns:
        gdf.rename(columns={'Height': 'height'}, inplace=True)
    
    # Ensure 'height' exists, fill NaN
    if 'height' not in gdf.columns:
        gdf['height'] = 10.0 # Default value if missing
    
    # Reset ID field (Sequential from 1)
    gdf['id'] = np.arange(1, len(gdf) + 1)

    # 4. CRS Transformation (GeoJSON standard is EPSG:4326)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        print("[INFO] Transforming coordinates to WGS84 (EPSG:4326)...")
        gdf = gdf.to_crs(epsg=4326)

    # 5. Export
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    gdf.to_file(output_path, driver='GeoJSON', encoding='utf-8')
    print(f"[OK] GeoJSON saved successfully: {output_path}")


def main(argv):
    current_dir = pathlib.Path(__file__).parent.resolve()
    
    # 1. Read JSON configuration
    # Assuming json is in tmp folder
    json_path = os.path.join(current_dir, "tmp", "city_roi.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Configuration file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        roi_data = json.load(f)
    
    city_name = roi_data["city_name"]
    vertices_list = roi_data["rectangle_vertices"] 

    print(f"[INFO] Target City: {city_name}")

    # 2. Determine source file path
    p = city2p.get(city_name)
    if not p:
        raise ValueError(f"City not found in city2p dictionary: {city_name}")
    
    filename = f"{p}" 
    src_shp = os.path.join(
        current_dir, "local_shp", filename, f"{filename}.shp"
    )
    
    # Define output path (geojson only)
    out_dir = os.path.join(current_dir, "tmp")
    out_geojson = os.path.join(out_dir, "tmp.geojson")

    if not os.path.exists(src_shp):
        raise FileNotFoundError(f"Source shapefile not found: {src_shp}")

    # 3. Load source data layer
    vl = QgsVectorLayer(src_shp, "buildings", "ogr")
    if not vl.isValid():
        raise RuntimeError(f"Unable to load shapefile: {src_shp}")

    # 4. Build filtering geometry (Polygon)
    crs_wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
    poly_points = [QgsPointXY(v[0], v[1]) for v in vertices_list]
    roi_geometry_wgs84 = QgsGeometry.fromPolygonXY([poly_points])

    # 5. Coordinate Transformation: Transform AOI from WGS84 to Layer CRS
    lyr_crs = vl.crs()
    if not lyr_crs.isValid():
        raise RuntimeError("Source layer CRS is invalid.")

    if lyr_crs != crs_wgs84:
        xform = QgsCoordinateTransform(crs_wgs84, lyr_crs, QgsProject.instance())
        roi_geometry_layer = QgsGeometry(roi_geometry_wgs84)
        roi_geometry_layer.transform(xform)
    else:
        roi_geometry_layer = roi_geometry_wgs84

    # 6. Spatial Filtering
    print("[INFO] Starting spatial query...")
    req = QgsFeatureRequest().setFilterRect(roi_geometry_layer.boundingBox())
    
    selected = [f for f in vl.getFeatures(req) if f.geometry() and f.geometry().intersects(roi_geometry_layer)]
    print(f"[INFO] Hit features: {len(selected)}")

    if len(selected) == 0:
        print("[WARN] No buildings selected. Please check coordinate range.")
        # Create empty GeoJSON just in case downstream apps need it
        os.makedirs(os.path.dirname(out_geojson), exist_ok=True)
        with open(out_geojson, 'w') as f:
            f.write('{"type": "FeatureCollection", "features": []}')
        return 0

    # 7. Export directly to GeoJSON (New Function)
    export_features_to_geojson(vl, selected, out_geojson)

    return 0

if __name__ == "__main__":
    try:
        code = main(sys.argv[1:])
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        qgs.exitQgis()
    sys.exit(code)