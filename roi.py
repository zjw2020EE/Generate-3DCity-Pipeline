"""
Script to generate Region of Interest (ROI) for specified cities and save to JSON.

## Usage:
```bash python roi.py <city_index>
## city_index: Index of the city in the predefined city_names list.

## Output: A JSON file in the 'tmp' directory containing the city name and rectangle vertices.

## Dependencies:
- voxcity.geoprocessor.draw.center_location_map_cityname
- os, sys, json

## Note:
- The center_location_map_cityname function is modified to automatically sample a rectangle according to the cityname.

## Author: Garvin Z
## Date: 2025-12
"""
import sys
import os
import json
from voxcity_custom.geoprocessor.draw import center_location_map_cityname

# Parameters
width = 600 #@param {type:"number"}
height = 600 #@param {type:"number"}
city_names = ["Shenzhen", "Chongqing", "Chengdu", "Suzhou", "Hangzhou", "Hefei"]

def main(argv):
    if len(sys.argv) < 2:
        print("Usage: python ROI.py <city_index>")
        sys.exit(1)
    city_index = int(sys.argv[1])
    if city_index < 0 or city_index >= len(city_names):
        print("Invalid city index. Please provide an index between 0 and", len(city_names)-1)
        sys.exit(1)
    m, rectangle_vertices = center_location_map_cityname(
        city_names[city_index], width, height, zoom=15
    )
    data_to_save = {
        "city_name": city_names[city_index],
        "rectangle_vertices": rectangle_vertices
    }
    tmp_output_dir = "tmp"
    os.makedirs(tmp_output_dir, exist_ok=True)
    tmp_output_path = os.path.join(tmp_output_dir, f"city_roi.json") 
    with open(tmp_output_path, "w") as f:
        json.dump(data_to_save, f, indent=4)
    print(f"ROI for {city_names[city_index]} saved to {tmp_output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))