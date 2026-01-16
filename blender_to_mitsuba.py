"""
Script to convert a GLB file to a Mitsuba 3 XML scene using Blender.

## Usage:
    Run this script using Blender's background mode:
    
    [Windows]
    "C:/Path/To/blender.exe" --background --python blender_to_mitsuba.py
    ( "D:\Blender Foundation\Blender 3.6\blender.exe" --background --python blender_to_mitsuba.py )
    [Mac/Linux]
    blender --background --python blender_to_mitsuba.py

## Workflow:
1. Locates 'scene.glb' in the 'tmp' directory relative to this script.
2. Clears the default Blender scene (Cube, Camera, Light).
3. Imports the GLB file.
4. Exports the scene to Mitsuba XML format in 'tmp/mitsuba_export'.

## Dependencies:
- Blender (Version 3.x or 4.x recommended)
- Mitsuba Blender Add-on (Must be installed and enabled in Blender)https://github.com/mitsuba-renderer/mitsuba-blender

## Author: Garvin Z
## Date: 2025-12
"""

import bpy
import os
import sys

# ================= CONFIGURATION =================

# Get the absolute path of the directory containing this script
# This ensures paths work correctly regardless of where you run the command from
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define input/output paths based on the 'tmp' folder structure
TMP_DIR = os.path.join(CURRENT_DIR, "tmp")
GLB_INPUT_PATH = os.path.join(TMP_DIR, "scene.glb")

# Output directory for Mitsuba XML
XML_OUTPUT_DIR = os.path.join(TMP_DIR, "mitsuba_export")
XML_OUTPUT_PATH = os.path.join(XML_OUTPUT_DIR, "scene.xml")

# ===============================================

def clear_scene():
    """
    Clears all objects and meshes from the current Blender scene.
    This prevents merging the GLB with the default Cube/Camera/Light.
    """
    # Deselect all first to avoid context errors
    bpy.ops.object.select_all(action='DESELECT')
    
    # Iterate over all objects and remove them
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
        
    # Also remove orphaned meshes to keep the file clean
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh, do_unlink=True)
    
    print("[Info] Scene cleared.")

def import_glb(glb_path):
    """
    Imports a GLB file into the scene.
    """
    if not os.path.exists(glb_path):
        print(f"[Error] GLB file not found: {glb_path}")
        return False
    
    print(f"[Info] Importing GLB: {glb_path}")
    try:
        # Import the GLTF/GLB file
        bpy.ops.import_scene.gltf(filepath=glb_path)
        return True
    except Exception as e:
        print(f"[Error] Import failed: {e}")
        return False

def export_mitsuba(xml_path):
    """
    Exports the current scene to Mitsuba XML format.
    """
    print(f"[Info] Attempting export to: {xml_path}")
    
    # Ensure the output directory exists
    if not os.path.exists(os.path.dirname(xml_path)):
        os.makedirs(os.path.dirname(xml_path), exist_ok=True)
    
    try:
        # Check if the Mitsuba exporter operator exists
        if not hasattr(bpy.ops.export_scene, "mitsuba"):
            print("[Error] Mitsuba add-on not found! Please install and enable 'mitsuba-blender' in Blender.")
            return

        # Execute export
        # Note: Coordinate axis settings ensure the model stands upright in Mitsuba
        bpy.ops.export_scene.mitsuba(
            filepath=xml_path,
            export_ids=True,        # Useful for identifying objects in Sionna
            ignore_background=True, # Use default Mitsuba environment instead of Blender's grey world
            axis_forward='Y',
            axis_up='Z'
        )
        print("[Success] Export succeeded.")
    except Exception as e:
        print(f"[Error] Export failed: {e}")

def main():
    print(">>> Starting Blender to Mitsuba Conversion...")

    # 1. Clean up the default scene
    clear_scene()

    # 2. Import the generated GLB
    if not import_glb(GLB_INPUT_PATH):
        print("Aborting due to import failure.")
        return

    # 3. Export to Mitsuba XML
    export_mitsuba(XML_OUTPUT_PATH)
    
    print(">>> Conversion Process Finished.")

if __name__ == "__main__":
    main()