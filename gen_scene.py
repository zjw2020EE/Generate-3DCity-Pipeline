"""
Script to generate a 3D scene (GLB) with Multi-Material Terrain, Buildings, and Trees.

## Updates in this version:
1.  **Multi-Material Terrain:** Terrain is now split into 4 sub-meshes (Water, Wet, Medium Dry, Very Dry) based on OSM Land Use.
2.  **Material Mapping:** Land Use IDs are mapped to specific ground physical properties.
3.  **Visual Distinction:** Water is blue/reflective, Dry ground is grey/matte, etc.

## Author: Garvin Z
## Date: 2025-12
"""

import os
import json
import random
import numpy as np
import geopandas as gpd
import trimesh
import pyproj
import pandas as pd
from shapely.geometry import Polygon, Point
from shapely.ops import transform
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals
from scipy.interpolate import RegularGridInterpolator
import warnings

warnings.filterwarnings("ignore", message=".*Several features with id.*")

# ================= CONFIGURATION =================

# --- File Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
tmp_dir = os.path.join(current_dir, "tmp")
tmp_output_dir = os.path.join(tmp_dir, "outputs")

GEOJSON_PATH = os.path.join(tmp_output_dir, "building.geojson")
DEM_PATH = os.path.join(tmp_output_dir, "crop_dem.npy")
VERTICES_PATH = os.path.join(tmp_dir, "city_roi.json")
CANOPY_TOP_PATH = os.path.join(tmp_output_dir, "canopy_height_top.npy")
CANOPY_BOTTOM_PATH = os.path.join(tmp_output_dir, "canopy_height_bottom.npy")
LAND_USE_PATH = os.path.join(tmp_output_dir, "land_use_osm.npy")

OUTPUT_PATH = os.path.join(tmp_dir, "scene.glb")

# --- Global Parameters ---
MESHSIZE = 2.0  
CONVERT_TO_Y_UP = True 

# --- Tree Generation Settings ---
TREE_MIN_HEIGHT = 1.5  
TREE_JITTER = 1.2      

# --- Material Definitions ---

# 1. Buildings & Trees
MAT_BUILDING = PBRMaterial(name="Building", baseColorFactor=[0.35, 0.55, 0.85, 1.0], metallicFactor=0.2, roughnessFactor=0.25)
MAT_TRUNK = PBRMaterial(name="Trunk", baseColorFactor=[0.25, 0.15, 0.1, 1.0], roughnessFactor=0.9)
MAT_VEGETATION = PBRMaterial(name="Vegetation", baseColorFactor=[0.2, 0.5, 0.2, 1.0], roughnessFactor=0.8)

# 2. [NEW] Terrain Materials (Sionna-Ready Categories)
# Concrete -> Medium Roughness, Dark Grey
MAT_CONCRETE = PBRMaterial(name="Terrain_Concrete", baseColorFactor=[0.4, 0.4, 0.4, 1.0], roughnessFactor=0.5, metallicFactor=0.0)
# Road -> Low Roughness, Light Grey
MAT_ROAD = PBRMaterial(name="Terrain_Road", baseColorFactor=[0.7, 0.7, 0.65, 1.0], roughnessFactor=0.3, metallicFactor=0.0)

# Very Dry (Sand, Rock) -> High Roughness, Light Grey
MAT_VERY_DRY = PBRMaterial(name="Terrain_VeryDry", baseColorFactor=[0.7, 0.7, 0.65, 1.0], roughnessFactor=0.9, metallicFactor=0.0)

# Medium Dry (Grass, Forest Soil, Agri) -> Medium Roughness, Greenish-Brown
MAT_MEDIUM_DRY = PBRMaterial(name="Terrain_MediumDry", baseColorFactor=[0.4, 0.45, 0.3, 1.0], roughnessFactor=0.8, metallicFactor=0.0)

# Wet (Wetland, Mud) -> Darker, Lower Roughness
MAT_WET = PBRMaterial(name="Terrain_Wet", baseColorFactor=[0.25, 0.2, 0.15, 1.0], roughnessFactor=0.4, metallicFactor=0.0)

# Water (Lake, River) -> Blue, Very Low Roughness, High Reflectivity
MAT_WATER = PBRMaterial(name="Terrain_Water", baseColorFactor=[0.1, 0.3, 0.6, 1.0], roughnessFactor=0.1, metallicFactor=0.8)

# ===========================================

def load_vertices(path):
    if not os.path.exists(path): raise FileNotFoundError(f"Vertices file not found: {path}")
    with open(path, 'r') as f: return json.load(f)["rectangle_vertices"]

def estimate_utm_crs(lat, lon):
    zone = int((lon + 180) / 6) + 1
    base = 32600 if lat >= 0 else 32700
    return f"EPSG:{base + zone}"

def get_terrain_category(lu_code):
    """
    Maps Land Use ID to Terrain Material Category.
    """
    # 8: Water
    if lu_code == 8: 
        return 'water'
    
    # 6: Wetland, 7: Mangrove
    elif lu_code in [6, 7]: 
        return 'wet'
    
    # 0: Bareland,  9: Ice
    elif lu_code in [0, 9, 13]: 
        return 'very_dry'
    
    # 10: Developed/Paved, 12: Building Base,
    elif lu_code in [10, 12]: 
        return 'concrete'
    
    # 11: Road
    elif lu_code in [11]: 
        return 'road'
    
    # 1: Grass, 2: Shrub, 3: Agri, 4: Forest, 5: Moss
    else: 
        return 'medium_dry'

def create_multimaterial_terrain(dem_array, land_use_array, meshsize, canopy_height_top_array=None, canopy_height_bottom_array=None):
    """
    Generates 4 separate terrain meshes based on land use categories.
    """
    print(">>> Generating Multi-Material Terrain...")
    rows, cols = dem_array.shape
    
    # Check shape consistency
    if land_use_array.shape != dem_array.shape:
        print("[WARN] Land Use shape mismatch. Fallback to single material.")
        # Create a dummy array of 'medium_dry' (code 1)
        land_use_array = np.ones_like(dem_array)

    if canopy_height_top_array is not None and canopy_height_bottom_array is not None:
        tree_threshold = 3.0
        vegetation_min_height = 0.5
        print("    Canopy data provided. Adjusting tree generation thresholds.")
        mask_shrub = (canopy_height_top_array > vegetation_min_height) & (canopy_height_top_array <= tree_threshold)
        land_use_array[mask_shrub] = 2  # Shrub
        mask_tree = canopy_height_top_array > tree_threshold
        land_use_array[mask_tree] = 4  # Forest

    width_m, height_m = cols * meshsize, rows * meshsize
    x_grid = np.linspace(-width_m / 2, width_m / 2, cols)
    y_grid = np.linspace(-height_m / 2, height_m / 2, rows)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing='xy')
    
    # Master Vertices Array
    vertices = np.column_stack((xx.flatten(), yy.flatten(), dem_array.flatten()))
    
    # Lists to store faces (triangles) for each category
    faces_dict = {
        'very_dry': [],
        'medium_dry': [],
        'wet': [],
        'water': [],
        'concrete': [],
        'road': [],
    }
    
    # Iterate grid to form triangles and classify them
    for r in range(rows - 1):
        for c in range(cols - 1):
            # Vertex indices
            v_sw = r * cols + c
            v_se = r * cols + (c + 1)
            v_nw = (r + 1) * cols + c
            v_ne = (r + 1) * cols + (c + 1)
            
            # Determine Material Category
            # We sample the Land Use at the South-West corner of the grid cell
            lu_code = land_use_array[r, c]
            category = get_terrain_category(lu_code)
            
            # Add two triangles for this grid cell to the specific category list
            faces_dict[category].append([v_sw, v_se, v_nw])
            faces_dict[category].append([v_se, v_ne, v_nw])

    meshes = []
    
    # Create Trimesh objects for each populated category
    for cat, faces in faces_dict.items():
        if not faces: continue
        
        print(f"    Creating sub-mesh for {cat}: {len(faces)} faces")
        
        # Create mesh referencing the master vertices
        # process=True will remove unreferenced vertices (optimizing file size)
        sub_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        
        # Assign Material
        if cat == 'water': sub_mesh.visual = TextureVisuals(material=MAT_WATER)
        elif cat == 'wet': sub_mesh.visual = TextureVisuals(material=MAT_WET)
        elif cat == 'medium_dry': sub_mesh.visual = TextureVisuals(material=MAT_MEDIUM_DRY)
        elif cat == 'concrete': sub_mesh.visual = TextureVisuals(material=MAT_CONCRETE)
        elif cat == 'road': sub_mesh.visual = TextureVisuals(material=MAT_ROAD)
        else: sub_mesh.visual = TextureVisuals(material=MAT_VERY_DRY)
        
        meshes.append(sub_mesh)
        
    return meshes, x_grid, y_grid, width_m, height_m

def get_elevation_interpolator(dem_array, x_coords, y_coords):
    return RegularGridInterpolator((y_coords, x_coords), dem_array, bounds_error=False, fill_value=None)

def project_geometry_to_local(geom, center_x_utm, center_y_utm, transformer):
    def transform_coords(x, y, z=None):
        x_utm, y_utm = transformer.transform(x, y)
        return x_utm - center_x_utm, y_utm - center_y_utm
    return transform(transform_coords, geom)

def clean_polygon(poly):
    if poly.is_empty or not poly.is_valid: return poly.buffer(0)
    return poly

# --- Tree Generation Functions ---

def create_lowpoly_crown(style, radius, height):
    if style == 'round': 
        crown = trimesh.creation.icosahedron(subdivisions=1) 
        crown.apply_scale([radius * 1.1, radius * 1.1, height/1.8]) 
        return crown
    elif style == 'pine':
        crown = trimesh.creation.cone(radius=radius, height=height, sections=6) 
        return crown
    elif style == 'layered':
        h1, h2 = height * 0.65, height * 0.55
        c1 = trimesh.creation.cone(radius=radius * 1.3, height=h1, sections=7)
        c2 = trimesh.creation.cone(radius=radius * 0.9, height=h2, sections=7)
        c1.apply_translation([0, 0, h1/2 - h1/2]) 
        c2.apply_translation([0, 0, h1 * 0.55])      
        return trimesh.util.concatenate([c1, c2])
    return trimesh.creation.box([radius, radius, height])

def generate_trees_stylized(dem_data, top_data, bottom_data, land_use_data, x_grid, y_grid):
    print(f">>> Generating Trees (Unified Vegetation Material)...")

    valid_mask = (top_data > TREE_MIN_HEIGHT)
    r_idx_pot, c_idx_pot = np.where(valid_mask)
    print(f"    Found {len(r_idx_pot)} potential tree locations.")

    trunk_meshes = []
    all_crown_meshes = []
    style_counts = {'round': 0, 'pine': 0, 'layered': 0}
    
    count = 0
    indices = zip(r_idx_pot, c_idx_pot)

    for r, c in indices:
        lu_code = land_use_data[r, c]
        
        # Land Use Logic (Density & Style)
        density_prob = 0.0
        preferred_style = 'round'
        scale_factor = 1.0

        if lu_code == 4: # Forest
            density_prob = 0.30
            preferred_style = 'forest_mix'
            scale_factor = 1.0 
        elif lu_code == 2: # Shrub
            density_prob = 0.15
            preferred_style = 'round'
            scale_factor = 0.6
        elif lu_code == 1: # Park
            density_prob = 0.25
            preferred_style = 'park_mix'
            scale_factor = 0.8
        elif lu_code == 11: # Road
            density_prob = 0.15
            preferred_style = 'round'
            scale_factor = 0.8
        elif lu_code in [10, 3]: # Developed/Agri
            density_prob = 0.15
            preferred_style = 'round'
            scale_factor = 0.7
        elif lu_code in [6, 7]: # Wetland
            density_prob = 0.2
            preferred_style = 'round'
            scale_factor = 0.8
        else:
            if top_data[r, c] > 6.0: density_prob = 0.05
            else: continue

        if random.random() > density_prob: continue

        # Data Extraction
        h_top = top_data[r, c]
        h_bottom = bottom_data[r, c]
        z_ground = dem_data[r, c]

        if np.isnan(h_bottom) or h_bottom < 0: h_bottom = 0.0
        if h_bottom >= h_top - 1.0: h_bottom = max(0.5, h_top * 0.3)
        
        h_trunk = h_bottom
        h_crown = h_top - h_bottom
        
        if lu_code == 2: 
            h_trunk = min(h_trunk, 0.3) 
            h_crown = h_top - h_trunk

        if h_crown < 1.0: continue

        # Style Selection
        style = 'round'
        rand_val = random.random()
        if preferred_style == 'forest_mix':
            style = 'pine' if rand_val < 0.6 else 'layered'
        elif preferred_style == 'park_mix':
            style = 'round' if rand_val < 0.7 else 'layered'
        
        style_counts[style] += 1

        # Geometry Generation
        pos_x = x_grid[c] + random.uniform(-TREE_JITTER, TREE_JITTER)
        pos_y = y_grid[r] + random.uniform(-TREE_JITTER, TREE_JITTER)

        radius_crown = h_crown * 0.9 * scale_factor
        radius_crown = min(radius_crown, MESHSIZE * 3.0) 
        radius_crown = max(radius_crown, 1.2)

        # Trunk
        actual_trunk_h = max(h_trunk, 0.5)
        radius_trunk = max(radius_crown * 0.15, 0.3) 
        
        trunk = trimesh.creation.cylinder(radius=radius_trunk, height=actual_trunk_h, sections=6)
        t_trunk = np.eye(4)
        trunk_top_z = z_ground + actual_trunk_h
        t_trunk[:3, 3] = [pos_x, pos_y, z_ground + actual_trunk_h/2.0]
        trunk.apply_transform(t_trunk)
        trunk_meshes.append(trunk)

        # Crown
        crown = create_lowpoly_crown(style, radius_crown, h_crown)
        crown.apply_transform(trimesh.transformations.rotation_matrix(random.uniform(0, 6.28), [0, 0, 1]))
        
        base_sink = random.uniform(0.3, 0.6)
        if lu_code == 2 or style == 'round': base_sink = random.uniform(0.5, 0.8)
        
        overlap_height = h_crown * base_sink
        crown_bottom_target_z = trunk_top_z - overlap_height
        crown_center_z = crown_bottom_target_z + (h_crown / 2.0)
        crown_center_z = max(crown_center_z, z_ground + h_crown/2.0 + 0.1)

        t_crown = np.eye(4)
        t_crown[:3, 3] = [pos_x, pos_y, crown_center_z]
        crown.apply_transform(t_crown)
        
        all_crown_meshes.append(crown)
        count += 1

    print(f"    Total Generated: {count}")
    print(f"    Tree Type Stats: {json.dumps(style_counts, indent=4)}")
    
    scene_nodes = []
    if trunk_meshes:
        print(f"    Merging {len(trunk_meshes)} trunks...")
        merged_trunks = trimesh.util.concatenate(trunk_meshes)
        merged_trunks.visual = TextureVisuals(material=MAT_TRUNK)
        scene_nodes.append(merged_trunks)
    
    if all_crown_meshes:
        print(f"    Merging {len(all_crown_meshes)} vegetation crowns...")
        merged_crowns = trimesh.util.concatenate(all_crown_meshes)
        merged_crowns.visual = TextureVisuals(material=MAT_VEGETATION)
        scene_nodes.append(merged_crowns)
            
    return scene_nodes

# ================= MAIN Execution =================

def main():
    print(">>> Starting Scene Generation...")

    if not os.path.exists(GEOJSON_PATH): raise FileNotFoundError(f"{GEOJSON_PATH} missing")
    if not os.path.exists(DEM_PATH): raise FileNotFoundError(f"{DEM_PATH} missing")

    vertices = load_vertices(VERTICES_PATH)
    # 1. Load Original Data
    original_dem = np.load(DEM_PATH)
    
    # Check for Land Use data
    lu_data = None
    if os.path.exists(LAND_USE_PATH):
        lu_data = np.load(LAND_USE_PATH)
    else:
        print("[WARN] Land Use data not found! Terrain will be single material.")
        lu_data = np.zeros_like(original_dem)

    # 2. Setup Coordinates & Grid (Moved up for pre-processing)
    lons, lats = zip(*vertices)
    cx, cy = np.mean(lons), np.mean(lats)
    target_crs = estimate_utm_crs(cy, cx)
    print(f"CRS: {target_crs}")
    
    transformer = pyproj.Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    min_x, min_y = transformer.transform(min(lons), min(lats))
    max_x, max_y = transformer.transform(max(lons), max(lats))
    center_x, center_y = (min_x + max_x)/2, (min_y + max_y)/2

    # Define Grid arrays manually here (so we can use them to flatten terrain)
    rows, cols = original_dem.shape
    width_m, height_m = cols * MESHSIZE, rows * MESHSIZE
    x_grid = np.linspace(-width_m / 2, width_m / 2, cols)
    y_grid = np.linspace(-height_m / 2, height_m / 2, rows)
    
    # Interpolator for finding building base height from ORIGINAL terrain
    get_z_interpolator = get_elevation_interpolator(original_dem, x_grid, y_grid)

    # 3. Process Buildings & Flatten Terrain
    print(">>> Processing Buildings & Flattening Terrain...")
    
    # Create a copy of DEM to modify
    modified_dem = original_dem.copy()
    
    gdf = gpd.read_file(GEOJSON_PATH)
    gdf['height'] = pd.to_numeric(gdf['height'], errors='coerce').fillna(10.0)
    gdf['min_height'] = pd.to_numeric(gdf.get('min_height', 0), errors='coerce').fillna(0.0)
    
    building_meshes = []
    
    # Define bounds for grid lookup optimization
    x_min_grid, x_max_grid = x_grid[0], x_grid[-1]
    y_min_grid, y_max_grid = y_grid[0], y_grid[-1]

    for idx, row in gdf.iterrows():
        if row.geometry is None: continue
        polys = [row.geometry] if row.geometry.geom_type == 'Polygon' else list(row.geometry.geoms)
        
        for poly in polys:
            # A. Prepare Building Geometry
            local_poly = project_geometry_to_local(clean_polygon(poly), center_x, center_y, transformer)
            
            # Calculate Base Z using centroid on ORIGINAL terrain
            cx_pt = max(x_min_grid, min(local_poly.centroid.x, x_max_grid))
            cy_pt = max(y_min_grid, min(local_poly.centroid.y, y_max_grid))
            base_z = get_z_interpolator([cy_pt, cx_pt])[0] + row['min_height']
            
            # B. Create Mesh
            mesh = trimesh.creation.extrude_polygon(local_poly, height=max(row['height'] - row['min_height'], 1.0))
            mesh.visual = TextureVisuals(material=MAT_BUILDING)
            mesh.apply_translation([0, 0, base_z])
            building_meshes.append(mesh)

            # C. Flatten Terrain Underneath
            # Find the bounding box of the building on the grid
            min_px, min_py, max_px, max_py = local_poly.bounds
            
            # Convert metric bounds to grid indices
            # usage: np.searchsorted finds the index where values should be inserted to maintain order
            c_start = np.searchsorted(x_grid, min_px)
            c_end = np.searchsorted(x_grid, max_px)
            r_start = np.searchsorted(y_grid, min_py)
            r_end = np.searchsorted(y_grid, max_py)
            
            # Clamp indices
            c_start, c_end = max(0, c_start), min(cols, c_end + 1)
            r_start, r_end = max(0, r_start), min(rows, r_end + 1)
            
            # Iterate only the subset of pixels in the bounding box
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    # Check if the grid point center is inside the polygon
                    # Note: x_grid[c] is the x coord, y_grid[r] is the y coord
                    pt = Point(x_grid[c], y_grid[r])
                    if local_poly.contains(pt):
                        modified_dem[r, c] = base_z

    # 4. Save the Modified DEM
    terrain_npy_path = os.path.join(tmp_output_dir, "terrain.npy")
    np.save(terrain_npy_path, modified_dem)
    print(f">>> Modified DEM saved to: {terrain_npy_path}")

    # 5. Generate Terrain Mesh (Using MODIFIED DEM)
    # We pass the modified_dem here so the visual mesh matches the flattened data
    ch_top = np.load(CANOPY_TOP_PATH)
    ch_bottom = np.load(CANOPY_BOTTOM_PATH)
    terrain_meshes, _, _, _, _ = create_multimaterial_terrain(modified_dem, lu_data, MESHSIZE, canopy_height_top_array=ch_top, canopy_height_bottom_array=ch_bottom)

    # 6. Generate Trees (Using MODIFIED DEM ensures trees sit on the flat ground near buildings)
    tree_nodes = []
    if os.path.exists(CANOPY_TOP_PATH) and lu_data is not None:
        print(">>> Processing Trees...")
        ch_top = np.load(CANOPY_TOP_PATH)
        ch_bottom = np.load(CANOPY_BOTTOM_PATH)
        if ch_top.shape == lu_data.shape:
            # Use modified_dem here
            tree_nodes = generate_trees_stylized(modified_dem, ch_top, ch_bottom, lu_data, x_grid, y_grid)
    
    # 7. Assemble Scene
    scene = trimesh.Scene()
    
    for tm in terrain_meshes:
        scene.add_geometry(tm)
        
    if building_meshes: 
        scene.add_geometry(trimesh.util.concatenate(building_meshes))
        
    for node in tree_nodes: 
        scene.add_geometry(node)

    if CONVERT_TO_Y_UP:
        scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))

    scene.export(OUTPUT_PATH)
    print(f"Done! Scene saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()