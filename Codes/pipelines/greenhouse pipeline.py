import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import rasterio
from rasterio.windows import Window
from rasterio.transform import from_gcps

from shapely.geometry import Polygon
import geopandas as gpd
from pyproj import Transformer

from ultralytics import YOLO


def check_env():
    print("Python:", sys.version)
    try:
        m = YOLO("yolov8n-seg.pt")
        print("YOLOv8 OK:", m.names)
    except Exception as e:
        print("YOLOv8 load failed:", e)
    try:
        print("rasterio:", rasterio.__version__)
        print("geopandas:", gpd.__version__)
    except Exception as e:
        print("GIS stack issue:", e)


def get_affine_and_crs(raster_path: Path):
    with rasterio.open(raster_path) as src:
        crs = src.crs
        transform = src.transform
        gcps, gcps_crs = src.gcps

        is_identity = (
            transform.a == 1 and transform.b == 0 and transform.c == 0
            and transform.d == 0 and transform.e == 1 and transform.f == 0
        )

        if gcps and gcps_crs:
            transform = from_gcps(gcps)
            if crs is None:
                crs = gcps_crs

        if is_identity and not gcps:
            raise RuntimeError(
                f"{raster_path} has no georeferencing (identity transform, no GCPs). "
                "Cannot proceed."
            )

        return crs, transform, src.width, src.height


def tile_raster(raster_path: Path, tiles_dir: Path, index_file: Path, ctx_file: Path,
                 tile_size: int = 640, overlap: int = 64):
    tiles_dir.mkdir(parents=True, exist_ok=True)
    crs, transform, width, height = get_affine_and_crs(raster_path)
    crs_str = str(crs) if crs else "EPSG:3857"

    ctx = {
        "raster_path": str(raster_path),
        "crs": crs_str,
        "transform": [transform.a, transform.b, transform.c,
                      transform.d, transform.e, transform.f],
        "width": width,
        "height": height,
    }
    ctx_file.write_text(json.dumps(ctx, indent=2), encoding="utf-8")

    step = tile_size - overlap
    count = 0
    with rasterio.open(raster_path) as src, open(index_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tile_name", "col_off", "row_off", "width", "height"])
        for row in range(0, height, step):
            for col in range(0, width, step):
                win_w = min(tile_size, width - col)
                win_h = min(tile_size, height - row)
                if win_w <= 0 or win_h <= 0:
                    continue
                window = Window(col_off=col, row_off=row, width=win_w, height=win_h)
                arr = src.read(window=window)
                arr = np.transpose(arr, (1, 2, 0))
                if arr.shape[2] > 3:
                    arr = arr[:, :, :3]
                img = Image.fromarray(arr.astype(np.uint8))
                tile_name = f"tile_{row}_{col}.png"
                img.save(tiles_dir / tile_name)
                writer.writerow([tile_name, col, row, win_w, win_h])
                count += 1

    print(f"{count} tiles saved -> {tiles_dir}")
    return ctx


def run_predict(weights_path: Path, tiles_dir: Path, out_dir: Path):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    images = [p for p in tiles_dir.glob("*") if p.suffix.lower() in exts]
    if not images:
        print("No tiles found in:", tiles_dir)
        return None

    model = YOLO(str(weights_path))
    model.predict(
        source=[str(p) for p in images],
        imgsz=1024,
        device="cpu",
        conf=0.35,
        iou=0.45,
        save=True,
        save_txt=True,
        project=str(out_dir),
        name="yolo_seg",
        exist_ok=True,
    )
    labels_dir = out_dir / "yolo_seg" / "labels"
    print("Predictions saved -> ", out_dir / "yolo_seg")
    return labels_dir


def normpoly_to_tile_pixels(line: str, w: int, h: int) -> np.ndarray:
    vals = [float(v) for v in line.strip().split()]
    pts = np.array(vals[1:], dtype=float).reshape(-1, 2)
    px = np.empty_like(pts)
    px[:, 0] = pts[:, 0] * w
    px[:, 1] = pts[:, 1] * h
    return px


def tilepix_to_globalpix(pts_px: np.ndarray, col_off: int, row_off: int) -> np.ndarray:
    out = pts_px.copy()
    out[:, 0] += col_off
    out[:, 1] += row_off
    return out


def pix_to_map(pts_global_px: np.ndarray, affine6):
    a, b, c, d, e, f = affine6
    col = pts_global_px[:, 0]
    row = pts_global_px[:, 1]
    x = a * col + b * row + c
    y = d * col + e * row + f
    return np.stack([x, y], axis=1)


def load_index(index_csv: Path):
    d = {}
    with open(index_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            d[row["tile_name"]] = {
                "col_off": int(row["col_off"]),
                "row_off": int(row["row_off"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
            }
    return d


def postprocess_to_map(ctx_file: Path, index_csv: Path, labels_dir: Path,
                        out_gpkg: Path, out_shp: Path, layer_name: str = "greenhouses"):
    if not ctx_file.exists():
        raise FileNotFoundError(f"Missing {ctx_file}")
    ctx = json.loads(ctx_file.read_text(encoding="utf-8"))
    if not ctx.get("crs"):
        raise RuntimeError("geo_context.json has no CRS.")

    src_crs = ctx["crs"]
    affine6 = ctx["transform"]
    index = load_index(index_csv)
    to_3857 = Transformer.from_crs(src_crs, "EPSG:3857", always_xy=True)

    geoms, tiles, class_ids = [], [], []
    label_files = sorted([p for p in labels_dir.glob("*.txt") if p.is_file()]) if labels_dir and labels_dir.exists() else []

    if not label_files:
        print("No label files found in:", labels_dir)
        return None

    for lf in label_files:
        tile_name = lf.stem + ".png"
        if tile_name not in index:
            tile_name = lf.stem + ".jpg"
        if tile_name not in index:
            print("Skipping", lf.name, "- no matching tile in index.")
            continue

        meta = index[tile_name]
        w, h = meta["width"], meta["height"]
        col0, row0 = meta["col_off"], meta["row_off"]

        for line in lf.read_text().strip().splitlines():
            vals = [float(v) for v in line.split()]
            if len(vals) < 3:
                continue
            cls = int(vals[0])

            pts_tile_px = normpoly_to_tile_pixels(line, w, h)
            pts_global_px = tilepix_to_globalpix(pts_tile_px, col0, row0)
            pts_xy = pix_to_map(pts_global_px, affine6)

            if len(pts_xy) < 3:
                continue

            poly = Polygon(pts_xy)
            if not poly.is_valid or poly.area <= 0:
                continue

            x3857, y3857 = to_3857.transform(pts_xy[:, 0], pts_xy[:, 1])
            poly3857 = Polygon(np.column_stack([x3857, y3857]))

            if poly3857.is_valid and poly3857.area > 0:
                geoms.append(poly3857)
                tiles.append(tile_name)
                class_ids.append(cls)

    if not geoms:
        print("No polygons collected.")
        return None

    gdf = gpd.GeoDataFrame({"class_id": class_ids, "tile": tiles}, geometry=geoms, crs="EPSG:3857")
    gdf["area_m2"] = gdf.geometry.area

    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    out_shp.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_gpkg, layer=layer_name, driver="GPKG")
    gdf.to_file(out_shp, driver="ESRI Shapefile")

    print(f"Collected {len(gdf)} polygons. Total area_m2: {gdf['area_m2'].sum():.2f}")
    print("Exported:", out_gpkg, "and", out_shp)
    return gdf


def preview_polygons(image_path: Path, label_path: Path, save_path: Path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        coords = list(map(float, parts[1:]))
        pts = list(zip(coords[0::2], coords[1::2]))
        poly = [(x * w, y * h) for x, y in pts]
        draw.polygon(poly, fill=(0, 255, 0, 60), outline=(0, 180, 0, 255))
        draw.line(poly + [poly[0]], fill=(0, 120, 0, 255), width=3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)
    print("Preview saved:", save_path)


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input", default=None, help="Path to input GeoTIFF (defaults to <root>/datasets/demo.tif if not given)")
    # parser.add_argument("--root", default=r"C:\greenhouse-ai", help="Project root folder")
    # parser.add_argument("--weights", default=None, help="Path to trained best.pt (defaults to <root>/models/best.pt)")
    # parser.add_argument("--tile-size", type=int, default=640)
    # parser.add_argument("--overlap", type=int, default=64)
    # parser.add_argument("--check-env", action="store_true")
    # args = parser.parse_args()

    # root = Path(args.root)
    # raster_path = Path(args.input) if args.input else root / "datasets" / "demo.tif"
    # weights_path = Path(args.weights) if args.weights else root / "models" / "best.pt"

    # tiles_dir = root / "tiles" / "images"
    # index_file = root / "tiles" / "tile_index.csv"
    # ctx_file = root / "tiles" / "geo_context.json"
    # preds_dir = root / "results" / "preds"
    # out_gpkg = root / "results" / "greenhouses.gpkg"
    # out_shp = root / "results" / "greenhouses_shp" / "greenhouses.shp"

    # if args.check_env:
    #     check_env()

    # if not raster_path.exists():
    #     raise FileNotFoundError(f"Input raster not found: {raster_path}")

    # tile_raster(raster_path, tiles_dir, index_file, ctx_file,
    #             tile_size=args.tile_size, overlap=args.overlap)

    # if not weights_path.exists():
    #     raise FileNotFoundError(f"Weights not found: {weights_path}")

    weights_path = Path(r"C:\Users\alias\Documents\UNI\Thesis\yolov8n-train\weights\best.pt")

    root = r"C:\Users\alias\AppData\Local\Temp\greenhouse_ns0u562v"

    tiles_dir = Path(os.path.join(root, r"tiles"))
    preds_dir = Path(os.path.join(root, r"preds"))

    labels_dir = run_predict(weights_path, tiles_dir, preds_dir)
    print("Labels dir:", labels_dir)

    # postprocess_to_map(ctx_file, index_file, labels_dir, out_gpkg, out_shp)


if __name__ == "__main__":
    main()
