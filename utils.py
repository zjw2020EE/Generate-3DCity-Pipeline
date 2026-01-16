import os
import xml.etree.ElementTree as ET
import sionna
from sionna.rt import load_scene, SceneObject, RadioMaterial, PlanarArray, Transmitter
import numpy as np
import drjit as dr

# ==============================================================================
# 1. ITU-R P.2040 / Standard Material Coefficients Configuration
#    Formula: 
#      epsilon_r = a * f_ghz^b
#      sigma     = c * f_ghz^d
# ==============================================================================
ITU_COEFFICIENTS = {
    # Name from table           a       b       c        d
    "vacuum":               (1.0,    0.0,    0.0,     0.0),
    "concrete":             (5.24,   0.0,    0.0462,  0.7822),
    "brick":                (3.91,   0.0,    0.0238,  0.16),
    "plasterboard":         (2.73,   0.0,    0.0085,  0.9395),
    "wood":                 (1.99,   0.0,    0.0047,  1.0718),
    "glass":                (6.31,   0.0,    0.0036,  1.3394),
    "ceiling_board":        (1.48,   0.0,    0.0011,  1.0750),
    "chipboard":            (2.58,   0.0,    0.0217,  0.7800),
    "plywood":              (2.71,   0.0,    0.33,    0.0),
    "marble":               (7.074,  0.0,    0.0055,  0.9262),
    "floorboard":           (3.66,   0.0,    0.0044,  1.3515),
    "very_dry_ground":      (3.0,    0.0,    0.00015, 2.52),
    "medium_dry_ground":    (15.0,  -0.1,    0.035,   1.63),
    "wet_ground":           (30.0,  -0.4,    0.15,    1.30),
}

def get_freq_callback(mat_type):
    """
    Returns a callback function for Sionna RadioMaterial based on material type.
    """
    if mat_type not in ITU_COEFFICIENTS:
        return None
    
    # Capture coefficients in closure
    a, b, c, d = ITU_COEFFICIENTS[mat_type]
    
    def callback(f_hz):
        # Convert Hz to GHz
        f_ghz = f_hz / 1e9
        
        # Calculate properties based on provided formulas
        # Note: Operators are overloaded by Sionna/DrJit to handle tensors
        rel_perm = a * (f_ghz ** b)
        conductivity = c * (f_ghz ** d)
        
        return rel_perm, conductivity
        
    return callback

def parse_mitsuba_xml(xml_path):
    """
    Parses the Mitsuba XML file to extract shape information.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    obj_ids_list = []
    id_to_filename = {}
    id_to_matname = {}
    
    for shape in root.findall('shape'):
        obj_id = shape.get('id')
        if not obj_id: continue
            
        obj_ids_list.append(obj_id)
        
        filename_node = shape.find("string[@name='filename']")
        if filename_node is not None:
            base_path = os.path.dirname(xml_path) + "/"
            id_to_filename[obj_id] = base_path + filename_node.get('value')
            
        ref_node = shape.find("ref[@name='bsdf']")
        if ref_node is not None:
            id_to_matname[obj_id] = ref_node.get('id')

    unique_materials = set(id_to_matname.values())

    return obj_ids_list, id_to_filename, id_to_matname, unique_materials

def build_sionna_scene(obj_ids_list, id_to_filename, id_to_matname, unique_materials):
    """
    Constructs a Sionna Scene with frequency-dependent RadioMaterials.
    """
    # 1. Initialize empty Scene
    scene = load_scene()
    
    # -------------------------------------------------------------------------
    # Step A: Define Mapping from XML Material Names to ITU Types
    # -------------------------------------------------------------------------
    # Map your XML names to the keys in ITU_COEFFICIENTS or define Static fallbacks
    mat_config_map = {
        "mat-Terrain_VeryDry":   {"type": "very_dry_ground", "scattering": 0.4, "thickness": 30.0},
        "mat-Terrain_MediumDry": {"type": "medium_dry_ground", "scattering": 0.4, "thickness": 30.0},
        "mat-Terrain_Wet":       {"type": "wet_ground", "scattering": 0.3, "thickness": 30.0},
        "mat-Terrain_Concrete":  {"type": "concrete", "scattering": 0.2, "thickness": 30.0},
        "mat-Building":          {"type": "concrete", "scattering": 0.1, "thickness": 1.0}, # Assume concrete for buildings
        "mat-Trunk":             {"type": "wood", "scattering": 0.2, "thickness": 0.4},
        "mat-Terrain_Road":      {"type": "concrete", "scattering": 0.4, "thickness": 30.0},
        # Materials NOT in the coefficient table (Keep Static):
        "mat-Terrain_Water":     {"static": {"rel_perm": 78.0, "conductivity": 2.0}, "scattering": 0.0, "thickness": 10.0},
        "mat-Vegetation":        {"static": {"rel_perm": 1.02,  "conductivity": 0.002}, "scattering": 0.3, "thickness": 2.5},
    }

    # -------------------------------------------------------------------------
    # Step B: Instantiate RadioMaterials
    # -------------------------------------------------------------------------
    # print(f">>> Instantiating {len(unique_materials)} RadioMaterials...")
    material_objects = {}

    for mat_name in unique_materials:
        config = mat_config_map.get(mat_name)
        
        if config and "type" in config:
            # Case 1: Frequency Dependent Material
            itu_type = config["type"]
            callback = get_freq_callback(itu_type)
            scat_coeff = config.get("scattering", 0.0)
            thickness = config.get("thickness", 0.1)
            
            if callback:
                # print(f"   Creating [Frequency Dependent] material: {mat_name} ({itu_type})")
                rm = RadioMaterial(
                    name=mat_name,
                    frequency_update_callback=callback,
                    scattering_coefficient=scat_coeff,
                    thickness=thickness,
                )
            else:
                # Fallback if typo in code
                print(f"[WARN] Unknown ITU type '{itu_type}' for {mat_name}. Using default.")
                rm = RadioMaterial(name=mat_name)
                
        elif config and "static" in config:
            # Case 2: Static Material (Water, Vegetation)
            vals = config["static"]
            scat_coeff = config.get("scattering", 0.0)
            thickness = config.get("thickness", 0.1)
            
            # print(f"   Creating [Static] material: {mat_name}")
            rm = RadioMaterial(
                name=mat_name,
                relative_permittivity=vals["rel_perm"],
                conductivity=vals["conductivity"],
                scattering_coefficient=scat_coeff,
                thickness=thickness,
            )
        else:
            # Case 3: Unknown/Default
            print(f"[WARN] No config for {mat_name}. Using Vacuum default.")
            rm = RadioMaterial(name=mat_name)

        scene.add(rm)
        material_objects[mat_name] = rm

    # -------------------------------------------------------------------------
    # Step C: Create SceneObjects
    # -------------------------------------------------------------------------
    # print(f">>> Instantiating {len(obj_ids_list)} SceneObjects...")
    objects_to_add = []

    for obj_id in obj_ids_list:
        filename = id_to_filename.get(obj_id)
        mat_name = id_to_matname.get(obj_id)
        radio_mat_obj = material_objects.get(mat_name)
        
        if filename and radio_mat_obj:
            obj = SceneObject(
                name=obj_id,
                fname=filename,
                radio_material=radio_mat_obj
            )
            objects_to_add.append(obj)
        else:
            print(f"[Warning] Skipping {obj_id}: Missing filename or material.")

    # -------------------------------------------------------------------------
    # Step D: Batch Add
    # -------------------------------------------------------------------------
    if objects_to_add:
        scene.edit(add=objects_to_add)
        # print(f"Successfully added {len(objects_to_add)} objects.")
    
    return scene
