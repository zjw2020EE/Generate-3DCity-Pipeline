import os
import glob
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box, Polygon, MultiPolygon
from shapely.prepared import prep
from shapely.strtree import STRtree

from pathlib import Path
GEOJSON_DIR = os.path.join(Path(__file__).parent, "GEOJSON")
ALLOW_TOUCH_BOUNDARY = False

def _series_union_all(series: gpd.GeoSeries):
    """
    优先使用 GeoSeries.union_all()，若不可用则回退 shapely.ops.unary_union。
    """
    if hasattr(series, "union_all"):
        return series.union_all()
    else:
        from shapely.ops import unary_union
        return unary_union(series)


def _load_city_polygons_wgs84(geojson_path: str):
    """读取城市 GeoJSON 到 WGS84，合并后拆成纯 Polygon 列表（支持多要素/多面）。"""
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    union_geom = _series_union_all(gdf.geometry)   # 替代 unary_union
    if isinstance(union_geom, Polygon):
        polys = [union_geom]
    elif isinstance(union_geom, MultiPolygon):
        polys = list(union_geom.geoms)
    else:
        # 极端情况回退到逐要素展开
        polys = []
        for geom in gdf.geometry:
            if isinstance(geom, Polygon):
                polys.append(geom)
            elif isinstance(geom, MultiPolygon):
                polys.extend(list(geom.geoms))
    return polys, gdf.crs


def _normalize_query_to_indices(tree, candidates, geom_to_index):
    """
    兼容 Shapely 1.x/2.x 的 STRtree.query 返回：
    - 若返回索引(np.ndarray of int)，直接用；
    - 若返回几何对象列表/数组，则用 id(geom) 在映射里找索引。
    """
    if candidates is None:
        return []
    if isinstance(candidates, (list, tuple, np.ndarray)) and len(candidates) > 0:
        first = candidates[0]
        if isinstance(first, (int, np.integer)):
            return list(candidates)
        else:
            idxs = []
            for g in candidates:
                idx = geom_to_index.get(id(g))
                if idx is not None:
                    idxs.append(idx)
            return idxs
    return []

def sample_from_cityname(
    cityname: str,
    side_m: float,
    n_points: int = 1,
    allow_touch_boundary: bool = False,
    max_tries: int = 10000,
) -> tuple:
    """
    在城市边界内随机生成中心点，使以该点为中心的 side_m×side_m 正方形
    完全位于“任意一个”多边形内部（或贴边内部）。
    返回 : lan, lon（WGS84）
    """
    random.seed()

    # 1) 读取并拆成 polygon 列表（WGS84）
    geojson_path = os.path.join(GEOJSON_DIR, f"{cityname}.geojson")
    if not os.path.isfile(geojson_path):
        raise FileNotFoundError(f"找不到城市边界文件：{geojson_path}")
    polys_wgs84, crs_wgs84 = _load_city_polygons_wgs84(geojson_path)
    if not polys_wgs84:
        return pd.DataFrame(columns=["lon", "lat"])

    # 2) 投影到 UTM（米制）
    gseries = gpd.GeoSeries(polys_wgs84, crs=crs_wgs84)
    utm_crs = gseries.estimate_utm_crs()
    gseries_utm = gseries.to_crs(utm_crs)
    polys_utm = list(gseries_utm.values)  # list of Polygon
    prepared = [prep(p) for p in polys_utm]

    # 空间索引
    tree = STRtree(polys_utm)
    # 为“几何返回”的情况准备 id->index 映射（Shapely 2.x 提供 tree.geometries）
    geoms_in_tree = getattr(tree, "geometries", polys_utm)
    geom_to_index = {id(geom): i for i, geom in enumerate(geoms_in_tree)}

    # 统一采样范围（所有多边形的联合外包框）——使用 union_all() 获取联合
    union_geom = _series_union_all(gpd.GeoSeries(polys_utm, crs=utm_crs))
    minx, miny, maxx, maxy = union_geom.bounds
    half = side_m / 2.0

    centers = []
    tries = 0
    while len(centers) < n_points and tries < max_tries:
        tries += 1
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        p = Point(x, y)

        # —— 点候选：用树粗筛，然后逐候选多边形精判
        cand_point_raw = tree.query(p)
        cand_point_idxs = _normalize_query_to_indices(tree, cand_point_raw, geom_to_index)
        if not cand_point_idxs:
            continue

        if allow_touch_boundary:
            ok_point = any(prepared[i].covers(p) for i in cand_point_idxs)
        else:
            ok_point = any(prepared[i].contains(p) for i in cand_point_idxs)
        if not ok_point:
            continue

        # —— 正方形候选：同样用树粗筛，然后逐候选精判
        sq = box(x - half, y - half, x + half, y + half)
        cand_sq_raw = tree.query(sq)
        cand_sq_idxs = _normalize_query_to_indices(tree, cand_sq_raw, geom_to_index)
        if not cand_sq_idxs:
            continue

        if allow_touch_boundary:
            ok_square = any(polys_utm[i].covers(sq) for i in cand_sq_idxs)
        else:
            ok_square = any(polys_utm[i].contains(sq) for i in cand_sq_idxs)

        if ok_square:
            centers.append(p)
    if len(centers) < n_points:
        print(f"[警告] {os.path.basename(geojson_path)} 仅生成 {len(centers)}/{n_points} 个中心点；"
            f"可增大 MAX_TRIES_PER_CITY，设 ALLOW_TOUCH_BOUNDARY=True，或减小 SIDE_M。")

    # 投回 WGS84
    centers_gdf_utm = gpd.GeoDataFrame(geometry=centers, crs=utm_crs)
    centers_wgs84 = centers_gdf_utm.to_crs(epsg=4326)
    lon = centers_wgs84.geometry.x.iloc[0]
    lat = centers_wgs84.geometry.y.iloc[0]
    return (round(lat, 8), round(lon, 8))
    