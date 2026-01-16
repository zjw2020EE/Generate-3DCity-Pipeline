"""
Height extraction and complement utilities for building footprints.
"""

from typing import List, Dict

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.errors import GEOSException
from shapely.geometry import shape
from rtree import index
import rasterio
from pyproj import Transformer, CRS


def extract_building_heights_from_gdf(gdf_0: gpd.GeoDataFrame, gdf_1: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Extract building heights from one GeoDataFrame and apply them to another based on spatial overlap.
    """
    gdf_primary = gdf_0.copy()
    gdf_ref = gdf_1.copy()

    if 'height' not in gdf_primary.columns:
        gdf_primary['height'] = 0.0
    if 'height' not in gdf_ref.columns:
        gdf_ref['height'] = 0.0

    count_0 = 0
    count_1 = 0
    count_2 = 0

    spatial_index = index.Index()
    for i, geom in enumerate(gdf_ref.geometry):
        if geom.is_valid:
            spatial_index.insert(i, geom.bounds)

    for idx_primary, row in gdf_primary.iterrows():
        if row['height'] <= 0 or pd.isna(row['height']):
            count_0 += 1
            geom = row.geometry

            overlapping_height_area = 0
            overlapping_area = 0

            potential_matches = list(spatial_index.intersection(geom.bounds))

            for ref_idx in potential_matches:
                if ref_idx >= len(gdf_ref):
                    continue

                ref_row = gdf_ref.iloc[ref_idx]
                try:
                    if geom.intersects(ref_row.geometry):
                        overlap_area = geom.intersection(ref_row.geometry).area
                        overlapping_height_area += ref_row['height'] * overlap_area
                        overlapping_area += overlap_area
                except GEOSException:
                    try:
                        fixed_ref_geom = ref_row.geometry.buffer(0)
                        if geom.intersects(fixed_ref_geom):
                            overlap_area = geom.intersection(fixed_ref_geom).area
                            overlapping_height_area += ref_row['height'] * overlap_area
                            overlapping_area += overlap_area
                    except Exception:
                        print(f"Failed to fix polygon")
                    continue

            if overlapping_height_area > 0:
                count_1 += 1
                new_height = overlapping_height_area / overlapping_area
                gdf_primary.at[idx_primary, 'height'] = new_height
            else:
                count_2 += 1
                gdf_primary.at[idx_primary, 'height'] = np.nan

    if count_0 > 0:
        print(f"For {count_1} of these building footprints without height, values from the complementary source were assigned.")
        print(f"For {count_2} of these building footprints without height, no data exist in complementary data.")

    return gdf_primary


def complement_building_heights_from_gdf(gdf_0, gdf_1, primary_id='id', ref_id='id'):
    """
    Vectorized approach with GeoPandas to compute weighted heights and add non-intersecting buildings.
    Returns a single combined GeoDataFrame.
    """
    gdf_primary = gdf_0.copy()
    gdf_ref = gdf_1.copy()

    if 'height' not in gdf_primary.columns:
        gdf_primary['height'] = 0.0
    if 'height' not in gdf_ref.columns:
        gdf_ref['height'] = 0.0

    gdf_primary = gdf_primary.rename(columns={'height': 'height_primary'})
    gdf_ref = gdf_ref.rename(columns={'height': 'height_ref'})

    intersect_gdf = gpd.overlay(gdf_primary, gdf_ref, how='intersection')
    # intersect_gdf['intersect_area'] = intersect_gdf.area
    if not intersect_gdf.empty:
        if intersect_gdf.crs.is_geographic:
            crs_projected = intersect_gdf.estimate_utm_crs()
            intersect_gdf['intersect_area'] = intersect_gdf.to_crs(crs_projected).area
        else:
            intersect_gdf['intersect_area'] = intersect_gdf.area
    else:
        intersect_gdf['intersect_area'] = 0.0
    
    intersect_gdf['height_area'] = intersect_gdf['height_ref'] * intersect_gdf['intersect_area']

    group_cols = {
        'height_area': 'sum',
        'intersect_area': 'sum'
    }
    grouped = intersect_gdf.groupby(f'{primary_id}_1').agg(group_cols)
    grouped['weighted_height'] = grouped['height_area'] / grouped['intersect_area']

    gdf_primary = gdf_primary.merge(grouped['weighted_height'],
                                    left_on=primary_id,
                                    right_index=True,
                                    how='left')

    zero_or_nan_mask = (gdf_primary['height_primary'] == 0) | (gdf_primary['height_primary'].isna())
    valid_weighted_height_mask = zero_or_nan_mask & gdf_primary['weighted_height'].notna()
    gdf_primary.loc[valid_weighted_height_mask, 'height_primary'] = gdf_primary.loc[valid_weighted_height_mask, 'weighted_height']
    gdf_primary['height_primary'] = gdf_primary['height_primary'].fillna(np.nan)

    sjoin_gdf = gpd.sjoin(gdf_ref, gdf_primary, how='left', predicate='intersects')
    non_intersect_mask = sjoin_gdf[f'{primary_id}_right'].isna()
    non_intersect_ids = sjoin_gdf[non_intersect_mask][f'{ref_id}_left'].unique()
    gdf_ref_non_intersect = gdf_ref[gdf_ref[ref_id].isin(non_intersect_ids)]
    gdf_ref_non_intersect = gdf_ref_non_intersect.rename(columns={'height_ref': 'height'})

    gdf_primary = gdf_primary.rename(columns={'height_primary': 'height'})
    if 'weighted_height' in gdf_primary.columns:
        gdf_primary.drop(columns='weighted_height', inplace=True)

    final_gdf = pd.concat([gdf_primary, gdf_ref_non_intersect], ignore_index=True)

    count_total = len(gdf_primary)
    count_0 = len(gdf_primary[zero_or_nan_mask])
    count_1 = len(gdf_primary[valid_weighted_height_mask])
    count_2 = count_0 - count_1
    count_3 = len(gdf_ref_non_intersect)
    count_4 = count_3
    height_mask = gdf_ref_non_intersect['height'].notna() & (gdf_ref_non_intersect['height'] > 0)
    count_5 = len(gdf_ref_non_intersect[height_mask])
    count_6 = count_4 - count_5
    final_height_mask = final_gdf['height'].notna() & (final_gdf['height'] > 0)
    count_7 = len(final_gdf[final_height_mask])
    count_8 = len(final_gdf)

    if count_0 > 0:
        print(f"{count_0} of the total {count_total} building footprints from base data source did not have height data.")
        print(f"For {count_1} of these building footprints without height, values from complementary data were assigned.")
        print(f"For the rest {count_2}, no data exists in complementary data.")
        print(f"Footprints of {count_3} buildings were added from the complementary source.")
        print(f"Of these {count_4} additional building footprints, {count_5} had height data while {count_6} had no height data.")
        print(f"In total, {count_7} buildings had height data out of {count_8} total building footprints.")

    return final_gdf


def extract_building_heights_from_geotiff(geotiff_path, gdf):
    """
    Extract building heights from a GeoTIFF raster for building footprints in a GeoDataFrame.
    """
    gdf = gdf.copy()

    count_0 = 0
    count_1 = 0
    count_2 = 0

    with rasterio.open(geotiff_path) as src:
        transformer = Transformer.from_crs(CRS.from_epsg(4326), src.crs, always_xy=True)

        mask_condition = (gdf.geometry.geom_type == 'Polygon') & ((gdf.get('height', 0) <= 0) | gdf.get('height').isna())
        buildings_to_process = gdf[mask_condition]
        count_0 = len(buildings_to_process)

        for idx, row in buildings_to_process.iterrows():
            coords = list(row.geometry.exterior.coords)
            transformed_coords = [transformer.transform(lon, lat) for lon, lat in coords]
            polygon = shape({"type": "Polygon", "coordinates": [transformed_coords]})

            try:
                masked_data, _ = rasterio.mask.mask(src, [polygon], crop=True, all_touched=True)
                heights = masked_data[0][masked_data[0] != src.nodata]
                if len(heights) > 0:
                    count_1 += 1
                    gdf.at[idx, 'height'] = float(np.mean(heights))
                else:
                    count_2 += 1
                    gdf.at[idx, 'height'] = np.nan
            except ValueError as e:
                print(f"Error processing building at index {idx}. Error: {str(e)}")
                gdf.at[idx, 'height'] = None

    if count_0 > 0:
        print(f"{count_0} of the total {len(gdf)} building footprint from OSM did not have height data.")
        print(f"For {count_1} of these building footprints without height, values from complementary data were assigned.")
        print(f"For {count_2} of these building footprints without height, no data exist in complementary data.")

    return gdf

import geopandas as gpd
from rtree import index
import numpy as np
import pandas as pd


def fuse_buildings_by_overlap(
    gdf_0: gpd.GeoDataFrame,
    gdf_1: gpd.GeoDataFrame,
    iou_high_threshold: float = 0.8,
    area_similarity_ratio: float = 1.3,
    overlap_area_ratio_threshold: float = 0.7,
) -> gpd.GeoDataFrame:
    """
    按照建筑重合关系融合两个 GeoDataFrame（gdf_0 为基础，gdf_1 为补充）。

    规则（只针对有重叠的建筑对）：
    1. 以 gdf_0 为基础。
    2. 对于每个 gdf_1 的建筑，找到与其 IoU（交并比）最大的 gdf_0 建筑（如果有相交）。
       - IoU = intersection_area / union_area
    3. 对于这个最佳匹配对 (b0, b1)：
       - 若 IoU >= iou_high_threshold：
            认为是同一栋建筑，用 gdf_1 的高度替换 gdf_0 的高度（只改 height，不改 geometry）。
       - 若 0 < IoU < iou_high_threshold（交并比较低，但有相交）：
            * 计算 overlap_ratio_g1 = intersection_area / area1（gdf_1 建筑面积）
            * 若 overlap_ratio_g1 >= overlap_area_ratio_threshold（相交面积占 gdf_1 的面积比较高）：
                  保留 gdf_0，同时“直接添加”这栋 gdf_1 建筑（两栋都存在）。
            * 若 overlap_ratio_g1 < overlap_area_ratio_threshold（相交面积占 gdf_1 面积比较低）：
                  - 计算 area_ratio = max(area0, area1) / min(area0, area1)
                  - 若 area_ratio <= area_similarity_ratio（两者面积相近）：
                        用 gdf_1 的高度替换 gdf_0 的高度（只改 height，不改 geometry）
                  - 若 area_ratio > area_similarity_ratio（面积差异大）：
                        **丢弃这栋 gdf_1 建筑，不做任何操作**。
    4. 对于 gdf_1 中完全不与 gdf_0 相交的建筑，直接添加到结果中。
    """

    # 拷贝并重新编号索引，方便用整数索引定位
    g0 = gdf_0.copy().reset_index(drop=True)
    g1 = gdf_1.copy().reset_index(drop=True)

    # 确保 height 列存在
    if 'height' not in g0.columns:
        g0['height'] = 0.0
    if 'height' not in g1.columns:
        g1['height'] = 0.0

    # 构建 g0 的空间索引（基础数据）
    spatial_index = index.Index()
    for i, geom in enumerate(g0.geometry):
        if geom is not None and not geom.is_empty:
            spatial_index.insert(i, geom.bounds)

    # 记录需要删除的 g0 行 & 需要追加的 g1 行
    indices_to_drop_from_g0 = set()
    rows_to_append_from_g1 = []
    appended_g1_indices = set()

    # 统计信息（可选）
    cnt_height_replaced_high_iou = 0
    cnt_height_replaced_low_iou = 0
    cnt_added_overlap_high_ratio = 0
    cnt_discarded_large_area_diff = 0
    cnt_added_no_overlap = 0

    for idx1, row1 in g1.iterrows():
        geom1 = row1.geometry
        if geom1 is None or geom1.is_empty:
            continue

        # 找与 geom1 的包络框相交的 g0 候选
        candidate_indices = list(spatial_index.intersection(geom1.bounds))

        best_iou = 0.0
        best_idx0 = None

        # 在候选中寻找交并比最大的 g0 建筑
        for idx0 in candidate_indices:
            geom0 = g0.geometry.iloc[idx0]
            if geom0 is None or geom0.is_empty:
                continue
            if not geom0.intersects(geom1):
                continue

            inter_geom = geom0.intersection(geom1)
            inter_area = inter_geom.area
            if inter_area <= 0:
                continue

            area0 = geom0.area
            area1 = geom1.area
            union_area = area0 + area1 - inter_area
            if union_area <= 0:
                continue

            iou = inter_area / union_area
            if iou > best_iou:
                best_iou = iou
                best_idx0 = idx0

        # 没有任何相交的 g0 建筑：直接把 g1 这栋建筑加进结果
        if best_idx0 is None:
            if idx1 not in appended_g1_indices:
                rows_to_append_from_g1.append(row1)
                appended_g1_indices.add(idx1)
                cnt_added_no_overlap += 1
            continue

        # 有相交建筑：根据 IoU 和相交面积比例以及面积相似度来处理
        geom0 = g0.geometry.iloc[best_idx0]
        area0 = geom0.area
        area1 = geom1.area
        area_min = min(area0, area1)

        # 为了计算 overlap_ratio_g1，需要重新算交集面积
        inter_geom = geom0.intersection(geom1)
        inter_area = inter_geom.area if not inter_geom.is_empty else 0.0

        if area1 > 0:
            overlap_ratio_g1 = inter_area / area1
        else:
            overlap_ratio_g1 = 0.0
        
        if area0 > 0:
            overlap_ratio_g0 = inter_area / area0
        else:
            overlap_ratio_g0 = 0.0

        if area_min > 0:
            area_ratio = max(area0, area1) / area_min
        else:
            area_ratio = np.inf

        # 情况 1：IoU 非常高，用 gdf_1 的高度替换 gdf_0 的高度
        if best_iou >= iou_high_threshold:
            if best_idx0 not in indices_to_drop_from_g0:
                g0.at[best_idx0, 'height'] = row1['height']
                cnt_height_replaced_high_iou += 1
            # 不追加 g1（认为是同一栋）
            continue

        # 情况 2：IoU 较低但有重叠
        if best_iou > 0:
            # 2.1 相交面积 / gdf_1 面积 比较高：直接添加 gdf_1 建筑
            if overlap_ratio_g1 >= overlap_area_ratio_threshold:
                if idx1 not in appended_g1_indices:
                    rows_to_append_from_g1.append(row1)
                    appended_g1_indices.add(idx1)
                    cnt_added_overlap_high_ratio += 1
            elif overlap_ratio_g0 >= overlap_area_ratio_threshold:
                # 如果gdf_0对应建筑高度高于gdf_1，则添加gdf_1建筑
                if g0.at[best_idx0, 'height'] > row1['height']:
                    if idx1 not in appended_g1_indices:
                        rows_to_append_from_g1.append(row1)
                        appended_g1_indices.add(idx1)
                        cnt_added_overlap_high_ratio += 1
                else:
                    # 否则用 gdf_1 的高度替换 gdf_0 的高度
                    if best_idx0 not in indices_to_drop_from_g0:
                        g0.at[best_idx0, 'height'] = row1['height']
                        cnt_height_replaced_low_iou += 1
            else:
                # 2.2 相交面积 / gdf_1 面积 比较低：判断面积是否相近
                if area_ratio <= area_similarity_ratio:
                    # 面积相似：只用 g1 的高度替换 g0 的高度
                    if best_idx0 not in indices_to_drop_from_g0:
                        g0.at[best_idx0, 'height'] = row1['height']
                        cnt_height_replaced_low_iou += 1
                else:
                    cnt_discarded_large_area_diff += 1
                    # 不追加，直接跳过
                    continue

        # best_iou == 0 的情况已经在 best_idx0 is None 分支里处理完了

    # 从 g0 删除被替换掉的建筑（当前逻辑没有几何替换，所以集合一般是空的）
    g0_kept = g0.drop(index=list(indices_to_drop_from_g0)) if indices_to_drop_from_g0 else g0

    # 把所有需要追加的 g1 行组成一个 GeoDataFrame
    if rows_to_append_from_g1:
        g1_to_add = gpd.GeoDataFrame(rows_to_append_from_g1, crs=g1.crs)
        result = gpd.GeoDataFrame(
            pd.concat([g0_kept, g1_to_add], ignore_index=True),
            crs=g0.crs
        )
    else:
        result = g0_kept


    return result
