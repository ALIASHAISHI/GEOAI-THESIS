import os
import argparse
import random
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
from PIL import Image

import rasterio
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds_func
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.crs import CRS

import geopandas as gpd
from shapely.geometry import box, Point, Polygon
from shapely.ops import transform
import pyproj

def save_labelme_json(img_name, polygons, save_dir, img_size, class_mapping=None):
    """
    Save polygons in LabelMe JSON format
    
    Args:
        img_name (str): Image filename
        polygons (list): List of tuples (polygon, class_name) or just polygons
        save_dir (str/Path): Directory to save JSON files
        img_size (tuple): (width, height) of the image
        class_mapping (dict): Optional mapping from class names to IDs
    """
    data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imagePath": img_name,
        "imageData": None,
        "imageHeight": img_size[1],
        "imageWidth": img_size[0]
    }
    
    for item in polygons:
        # Handle both (polygon, class) tuples and plain polygons
        if isinstance(item, tuple):
            poly, class_name = item
        else:
            poly = item
            class_name = "GH"  # Default class name
        
        if poly.is_empty or poly.geom_type != 'Polygon':
            continue
            
        # Get exterior coordinates and convert to list
        coords = list(poly.exterior.coords[:-1])  # Remove last duplicate point
        
        data["shapes"].append({
            "label": str(class_name),
            "points": coords,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })
    
    json_path = os.path.join(save_dir, img_name.replace(".jpg", ".json").replace(".png", ".json"))
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    
    return json_path

def find_correct_raster_crs_global(raster_bounds, vector_bounds_latlon):
    """
    Try to find the correct CRS for the raster by testing global projections
    and seeing which one brings the raster bounds close to the vector bounds
    """
    print("🔍 Searching for correct raster CRS globally...")
    
    # Determine approximate location from vector bounds
    center_lon = (vector_bounds_latlon[0] + vector_bounds_latlon[2]) / 2
    center_lat = (vector_bounds_latlon[1] + vector_bounds_latlon[3]) / 2
    
    print(f"Target location: ({center_lon:.4f}, {center_lat:.4f})")
    
    # Generate candidate CRS based on location
    candidate_crs = []
    
    # UTM zones based on longitude
    utm_zone = int((center_lon + 180) / 6) + 1
    utm_zone = max(1, min(60, utm_zone))  # Clamp to valid range
    
    # Northern/Southern hemisphere
    if center_lat >= 0:
        utm_epsg = f"EPSG:{32600 + utm_zone}"  # UTM North
    else:
        utm_epsg = f"EPSG:{32700 + utm_zone}"  # UTM South
    
    candidate_crs.append(utm_epsg)
    
    # Add nearby UTM zones
    if utm_zone > 1:
        if center_lat >= 0:
            candidate_crs.append(f"EPSG:{32600 + utm_zone - 1}")
        else:
            candidate_crs.append(f"EPSG:{32700 + utm_zone - 1}")
    if utm_zone < 60:
        if center_lat >= 0:
            candidate_crs.append(f"EPSG:{32600 + utm_zone + 1}")
        else:
            candidate_crs.append(f"EPSG:{32700 + utm_zone + 1}")
    
    # Common global projections
    candidate_crs.extend([
        "EPSG:3857",   # Web Mercator
        "EPSG:3395",   # World Mercator
        "EPSG:4326",   # WGS84 (if raster is already in geographic coordinates)
        "EPSG:54009",  # Mollweide
        "EPSG:54030",  # Robinson
    ])
    
    # Regional projections based on continent/region
    if -180 <= center_lon <= -30:  # Americas
        candidate_crs.extend([
            "EPSG:5070",   # NAD83 / Conus Albers (North America)
            "EPSG:102033", # North America Albers Equal Area Conic
        ])
    elif -30 <= center_lon <= 60:  # Europe/Africa
        candidate_crs.extend([
            "EPSG:3035",   # ETRS89 / LAEA Europe
            "EPSG:3857",   # Web Mercator
            "EPSG:102024", # Africa Albers Equal Area Conic
        ])
    elif 60 <= center_lon <= 180:  # Asia/Oceania
        candidate_crs.extend([
            "EPSG:3832",   # WGS 84 / UTM zone 32N (Central Asia)
            "EPSG:102025", # Asia South Albers Equal Area Conic
        ])
    
    best_crs = None
    best_distance = float('inf')
    
    print(f"Testing {len(candidate_crs)} candidate CRS...")
    
    for test_crs in candidate_crs:
        try:
            # Transform raster bounds from test_crs to EPSG:4326
            bounds_4326 = transform_bounds(test_crs, "EPSG:4326", *raster_bounds)
            
            # Calculate center of transformed bounds
            test_center_lon = (bounds_4326[0] + bounds_4326[2]) / 2
            test_center_lat = (bounds_4326[1] + bounds_4326[3]) / 2
            
            # Distance from target location
            distance = ((test_center_lon - center_lon)**2 + (test_center_lat - center_lat)**2)**0.5
            
            print(f"  {test_crs}: center=({test_center_lon:.4f}, {test_center_lat:.4f}), distance={distance:.4f}")
            
            # Check if bounds are reasonable (not too large, not obviously wrong)
            width = bounds_4326[2] - bounds_4326[0]
            height = bounds_4326[3] - bounds_4326[1]
            
            # Reasonable geographic extent (not covering entire globe)
            if 0.001 < width < 180 and 0.001 < height < 90:
                if distance < best_distance:
                    best_distance = distance
                    best_crs = test_crs
                    
        except Exception as e:
            print(f"  {test_crs}: failed - {e}")
    
    if best_crs:
        print(f"✅ Best match: {best_crs} (distance: {best_distance:.4f})")
        return best_crs
    else:
        print("❌ No suitable CRS found, trying EPSG:4326 as fallback")
        return "EPSG:4326"

def test_crs_fix(tif_path, shp_path, correct_crs):
    """Test if the CRS fix works"""
    print(f"\n🧪 Testing CRS fix with {correct_crs}")
    
    # Load vector in EPSG:4326
    gdf = gpd.read_file(shp_path)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    vector_bounds = gdf.total_bounds
    
    with rasterio.open(tif_path) as src:
        original_bounds = src.bounds
        
        # Transform raster bounds to EPSG:4326 using correct CRS
        try:
            raster_bounds_4326 = transform_bounds(correct_crs, "EPSG:4326", *original_bounds)
        except Exception as e:
            print(f"Failed to transform bounds: {e}")
            return False
        
        print(f"Original raster bounds (in {correct_crs}): {original_bounds}")
        print(f"Raster bounds in EPSG:4326 (corrected): {raster_bounds_4326}")
        print(f"Vector bounds in EPSG:4326: {vector_bounds}")
        
        # Check overlap
        raster_box = box(*raster_bounds_4326)
        overlapping = sum(1 for _, row in gdf.iterrows() if row.geometry.intersects(raster_box))
        
        print(f"Overlapping polygons: {overlapping}/{len(gdf)}")
        
        return overlapping > 0

# ---------- Helper functions ----------

def random_point_in_polygon(poly, max_tries=200):
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(max_tries):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        p = Point(x, y)
        if p.within(poly):
            return p
    return poly.representative_point()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def polygon_to_yolo_segmentation(polygon, tile_size, dst_transform):
    """
    Convert a polygon to YOLO segmentation format (normalized coordinates)
    Returns list of normalized coordinates [x1, y1, x2, y2, ..., xn, yn]
    """
    inv_dst = ~dst_transform
    
    if polygon.geom_type == 'Polygon':
        # Use exterior coordinates
        coords = list(polygon.exterior.coords)
    else:
        return []  # Skip non-polygon geometries
    
    yolo_coords = []
    for x, y in coords[:-1]:  # Skip last coordinate (same as first)
        # Transform to pixel coordinates
        col, row = inv_dst * (x, y)
        
        # Normalize to [0, 1] range
        x_norm = max(0.0, min(1.0, col / tile_size))
        y_norm = max(0.0, min(1.0, row / tile_size))
        
        yolo_coords.extend([x_norm, y_norm])
    
    return yolo_coords

def polygon_to_labelme_pixel_coords(polygon, dst_transform):
    """
    Convert a polygon to LabelMe format (pixel coordinates)
    Returns list of [x, y] pixel coordinates
    """
    inv_dst = ~dst_transform
    
    if polygon.geom_type == 'Polygon':
        coords = list(polygon.exterior.coords)
    else:
        return []
    
    pixel_coords = []
    for x, y in coords[:-1]:  # Skip last coordinate (same as first)
        col, row = inv_dst * (x, y)
        pixel_coords.append([float(col), float(row)])
    
    return pixel_coords

def simplify_polygon_for_yolo(polygon, tolerance=1.0):
    """
    Simplify polygon to reduce number of points for YOLO format
    """
    try:
        simplified = polygon.simplify(tolerance, preserve_topology=True)
        
        # Ensure we have a valid polygon
        if simplified.is_empty or simplified.geom_type != 'Polygon':
            return polygon
        
        # Limit number of points (YOLO works better with fewer points)
        coords = list(simplified.exterior.coords)
        if len(coords) > 50:  # If too many points, simplify more aggressively
            simplified = polygon.simplify(tolerance * 2, preserve_topology=True)
            
        return simplified
    except:
        return polygon

# ---------- Main processing with global CRS support ----------

def generate_tiles_with_segmentation_global(tif_path: str,
                                           shp_path: str,
                                           out_dir: str,
                                           correct_raster_crs: str = None,
                                           tile_size: int = 256,
                                           scales = (1.0, 1.5, 0.75),
                                           tiles_per_polygon: int = 5,
                                           class_mapping = {"greenhouse": 0},
                                           seed: int = 42,
                                           simplify_tolerance: float = 1.0,
                                           save_labelme: bool = False):
    
    random.seed(seed)
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    ensure_dir(images_dir)
    ensure_dir(labels_dir)
    
    # Create LabelMe directory if needed
    if save_labelme:
        labelme_dir = out_dir / "labelme_annotations"
        ensure_dir(labelme_dir)

    # Load polygons and ensure they are in EPSG:4326
    print("📍 Loading vector data...")
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("Input shapefile has no features.")
    
    # Convert to EPSG:4326 if not already
    if gdf.crs != "EPSG:4326":
        print(f"Converting vector from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")
    
    if 'class' not in gdf.columns:
        gdf['class'] = 'greenhouse'
    
    print(f"Found {len(gdf)} polygons in {gdf.crs}")
    vector_bounds = gdf.total_bounds
    print(f"Vector bounds: {vector_bounds}")

    # Auto-detect correct raster CRS if not provided
    if correct_raster_crs is None:
        with rasterio.open(tif_path) as src:
            raster_bounds = src.bounds
        correct_raster_crs = find_correct_raster_crs_global(raster_bounds, vector_bounds)
    
    # Test the CRS fix
    if not test_crs_fix(tif_path, shp_path, correct_raster_crs):
        print("❌ CRS fix didn't work. Trying with original CRS...")
        correct_raster_crs = "EPSG:4326"
        if not test_crs_fix(tif_path, shp_path, correct_raster_crs):
            print("❌ Still no overlap. Please check your data.")
            return

    # Open raster and process with correct CRS
    with rasterio.open(tif_path) as src:
        src_width = src.width
        src_height = src.height
        src_transform = src.transform
        
        print(f"\n🖼️ Processing raster as {correct_raster_crs}")
        print(f"Raster dimensions: {src_width} x {src_height}")

        # Reproject polygons to the correct raster CRS
        print(f"📐 Reprojecting polygons from {gdf.crs} to {correct_raster_crs}")
        gdf_reproj = gdf.to_crs(correct_raster_crs)
        
        # Check intersection with corrected CRS
        raster_bounds = src.bounds
        raster_box = box(*raster_bounds)
        
        intersecting_indices = []
        for idx, row in gdf_reproj.iterrows():
            if row.geometry.intersects(raster_box):
                intersecting_indices.append(idx)
        
        print(f"✅ Found {len(intersecting_indices)} intersecting polygons")
        
        if len(intersecting_indices) == 0:
            print("❌ Still no intersecting polygons. Please check your data alignment.")
            return

        # Process tiles - Generate tiles and check ALL polygons for each tile
        tile_count = 0
        successful_tiles = 0
        failed_tiles = 0
        
        # Create a list to store all tile centers to avoid duplicates
        generated_tiles = set()
        
        for idx in tqdm(intersecting_indices, desc="Processing polygons"):
            row = gdf_reproj.iloc[idx]
            geom = row.geometry

            for sample_i in range(tiles_per_polygon):
                try:
                    center_point = random_point_in_polygon(geom)
                except Exception as e:
                    print(f"Failed to sample point in polygon {idx}: {e}")
                    continue

                for s_i, scale in enumerate(scales):
                    try:
                        # Calculate window dimensions
                        win_px_w = int(round(tile_size * scale))
                        win_px_h = int(round(tile_size * scale))

                        # Get center pixel coordinates
                        center_row, center_col = src.index(center_point.x, center_point.y)

                        # Calculate window bounds with safety checks
                        col_off = int(center_col - win_px_w // 2)
                        row_off = int(center_row - win_px_h // 2)

                        # Clamp to raster bounds
                        col_off = max(0, min(col_off, src_width - win_px_w))
                        row_off = max(0, min(row_off, src_height - win_px_h))
                        win_px_w = min(win_px_w, src_width - col_off)
                        win_px_h = min(win_px_h, src_height - row_off)

                        # Skip if window is too small
                        if win_px_w <= 10 or win_px_h <= 10:
                            continue

                        window = Window(col_off, row_off, win_px_w, win_px_h)
                        window_bounds = window_bounds_func(window, src_transform)

                        # Read data with error handling
                        try:
                            data = src.read(
                                window=window,
                                out_shape=(src.count, tile_size, tile_size),
                                resampling=Resampling.bilinear
                            )
                        except Exception as read_error:
                            print(f"Failed to read window: {read_error}")
                            failed_tiles += 1
                            continue

                        # Check if data is valid
                        if data is None or data.size == 0:
                            failed_tiles += 1
                            continue

                        # Process image data
                        img_arr = np.transpose(data, (1, 2, 0))

                        # Handle different band counts
                        if img_arr.shape[2] == 1:
                            img_arr = np.repeat(img_arr, 3, axis=2)
                        elif img_arr.shape[2] == 2:
                            img_arr = np.concatenate([img_arr, np.zeros((tile_size, tile_size, 1), dtype=img_arr.dtype)], axis=2)
                        elif img_arr.shape[2] > 3:
                            img_arr = img_arr[:, :, :3]

                        # Convert to float and normalize
                        img_arr = img_arr.astype(np.float32)
                        img_arr = np.nan_to_num(img_arr, nan=0.0, posinf=255.0, neginf=0.0)
                        
                        # Improved normalization per band
                        for band in range(img_arr.shape[2]):
                            band_data = img_arr[:, :, band]
                            valid_pixels = band_data[np.isfinite(band_data) & (band_data > 0)]
                            
                            if len(valid_pixels) > 100:
                                vmin = np.percentile(valid_pixels, 2)
                                vmax = np.percentile(valid_pixels, 98)
                                
                                if vmax > vmin:
                                    band_data = np.clip(band_data, vmin, vmax)
                                    band_data = (band_data - vmin) / (vmax - vmin) * 255.0
                                else:
                                    band_data = np.ones_like(band_data) * 128.0
                                    
                                img_arr[:, :, band] = band_data
                            else:
                                img_arr[:, :, band] = 128.0

                        img_arr = img_arr.astype(np.uint8)

                        # Create PIL image
                        img = Image.fromarray(img_arr)

                        # Generate file names
                        tile_name = f"tile_poly{idx}_s{s_i}_r{sample_i}_{tile_count}"
                        img_path = images_dir / f"{tile_name}.jpg"
                        label_path = labels_dir / f"{tile_name}.txt"

                        # Save image
                        img.save(img_path, format='JPEG', quality=90)

                        # Create YOLO segmentation annotations - CHECK ALL POLYGONS FOR THIS TILE
                        minx, miny, maxx, maxy = window_bounds
                        if (maxx - minx) <= 0 or (maxy - miny) <= 0:
                            open(label_path, 'w').close()
                            if save_labelme:
                                save_labelme_json(f"{tile_name}.jpg", [], labelme_dir, (tile_size, tile_size))
                            tile_count += 1
                            successful_tiles += 1
                            continue

                        dst_transform = from_bounds(minx, miny, maxx, maxy, tile_size, tile_size)
                        tile_box = box(minx, miny, maxx, maxy)
                        
                        yolo_lines = []
                        labelme_polygons = []  # For LabelMe format
                        
                        # Check ALL polygons in the dataset to see which ones intersect this tile
                        for check_idx, check_row in gdf_reproj.iterrows():
                            check_geom = check_row.geometry
                            check_class_name = str(check_row['class'])
                            check_class_id = class_mapping.get(check_class_name, 0)
                            
                            # Check if this polygon intersects with current tile
                            if check_geom.intersects(tile_box):
                                # Clip polygon to tile bounds
                                clipped = check_geom.intersection(tile_box)
                                
                                if clipped.is_empty:
                                    continue
                                
                                # Process geometry for both YOLO and LabelMe formats
                                geoms = []
                                if clipped.geom_type == 'Polygon':
                                    geoms = [clipped]
                                elif clipped.geom_type == 'MultiPolygon':
                                    geoms = list(clipped.geoms)
                                
                                for g in geoms:
                                    if g.is_empty or g.geom_type != 'Polygon':
                                        continue
                                    
                                    # Simplify polygon to reduce complexity
                                    simplified_poly = simplify_polygon_for_yolo(g, simplify_tolerance)
                                    
                                    # Convert to YOLO segmentation format
                                    seg_coords = polygon_to_yolo_segmentation(simplified_poly, tile_size, dst_transform)
                                    
                                    if len(seg_coords) >= 6:  # At least 3 points (6 coordinates)
                                        # Format: class_id x1 y1 x2 y2 ... xn yn
                                        coord_str = ' '.join([f"{coord:.6f}" for coord in seg_coords])
                                        yolo_lines.append(f"{check_class_id} {coord_str}")
                                        
                                        # For LabelMe format, convert to pixel coordinates
                                        if save_labelme:
                                            # Create a new polygon with pixel coordinates for LabelMe
                                            labelme_poly = Polygon(polygon_to_labelme_pixel_coords(simplified_poly, dst_transform))
                                            labelme_polygons.append((labelme_poly, check_class_name))

                        # Write YOLO label file
                        with open(label_path, 'w') as f:
                            if yolo_lines:
                                f.write("\n".join(yolo_lines))

                        # Save LabelMe annotation if requested
                        if save_labelme:
                            save_labelme_json(f"{tile_name}.jpg", labelme_polygons, labelme_dir, (tile_size, tile_size), class_mapping)

                        tile_count += 1
                        successful_tiles += 1

                    except Exception as tile_error:
                        print(f"Error processing tile: {tile_error}")
                        failed_tiles += 1
                        continue

        print(f"\n🎉 Processing complete!")
        print(f"Successful tiles: {successful_tiles}")
        print(f"Failed tiles: {failed_tiles}")
        print(f"Images saved to: {images_dir}")
        print(f"Labels saved to: {labels_dir}")
        if save_labelme:
            print(f"LabelMe annotations saved to: {labelme_dir}")
        print(f"YOLO label format: class_id x1 y1 x2 y2 ... xn yn")
        if save_labelme:
            print(f"LabelMe format: JSON files with polygon pixel coordinates")


if __name__ == "__main__":
    # First, let's find the correct CRS automatically
    print("=== AUTOMATIC YOLO SEGMENTATION DATASET GENERATION ===")
    
    # Try automatic detection with segmentation labels
    generate_tiles_with_segmentation_global(
        tif_path=r"C:\Users\Admin\Desktop\Cnrs\aadloun\image 3.tif",
        shp_path=r"C:\Users\Admin\Desktop\Cnrs\aadloun\aadloun_poly.shp",
        out_dir=r"C:\Users\Admin\Desktop\Cnrs\aadloun\dataset_output_segmentation",
        correct_raster_crs=None,  # Auto-detect
        tile_size=256,
        scales=(1.0, 0.5,1.5,2,0.25),
        tiles_per_polygon=5,
        class_mapping={"greenhouse": 0},
        seed=42,
        simplify_tolerance=1.0,
        save_labelme=True  # Enable LabelMe format export
    )