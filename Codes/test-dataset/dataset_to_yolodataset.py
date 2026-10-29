"""
Convert a labelme dataset (paired image + .json) into a YOLO-ready dataset
with BOTH detect (bbox) and segment (polygon) labels, plus rasterized masks.

    dataset/
    ├── data_detect.yaml
    ├── data_segment.yaml
    ├── images/{train,val[,test]}/
    ├── labels_detect/{train,val[,test]}/    ← YOLO bbox txt
    ├── labels_segment/{train,val[,test]}/   ← YOLO polygon txt
    └── masks/{train,val[,test]}/            ← PNG masks (0 / 255)

Usage
-----
    python labelme2yolo.py --src /path/to/labelme --dst /path/to/yolo_ds \
                           --val 0.2 --seed 42
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# ---------- discovery ------------------------------------------------------- #
def find_pairs(src: Path):
    pairs = []
    json_files = list(src.rglob("*.json"))
    for jp in tqdm(json_files, desc="Scanning labelme files", unit="file"):
        with open(jp) as f:
            data = json.load(f)
        img = jp.parent / data.get("imagePath", "")
        if not img.exists():
            for ext in IMG_EXTS:
                cand = jp.with_suffix(ext)
                if cand.exists():
                    img = cand
                    break
        if img.exists():
            pairs.append((img, jp, data))
        else:
            tqdm.write(f"  skip {jp.name}: no matching image")
    return pairs


def collect_classes(pairs, explicit):
    if explicit:
        return list(explicit)
    labels = set()
    for _, _, d in pairs:
        for s in d.get("shapes", []):
            labels.add(s["label"])
    return sorted(labels)


# ---------- geometry -------------------------------------------------------- #
def rect_to_poly(pts):
    (x1, y1), (x2, y2) = pts
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def clamp01(v):
    return max(0.0, min(1.0, v))


def poly_to_seg_line(cid, pts, W, H):
    coords = []
    for x, y in pts:
        coords.append(clamp01(x / W))
        coords.append(clamp01(y / H))
    return f"{cid} " + " ".join(f"{v:.6f}" for v in coords)


def poly_to_det_line(cid, pts, W, H):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = max(0, min(xs)), min(W, max(xs))
    y1, y2 = max(0, min(ys)), min(H, max(ys))
    cx = clamp01((x1 + x2) / 2 / W)
    cy = clamp01((y1 + y2) / 2 / H)
    bw = clamp01((x2 - x1) / W)
    bh = clamp01((y2 - y1) / H)
    return f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def extract_polygons(shapes):
    """Yield (class_label, polygon_points) for every usable shape."""
    for s in shapes:
        pts = s["points"]
        st = s["shape_type"]
        if st == "rectangle":
            pts = rect_to_poly(pts)
        elif st != "polygon":
            continue
        if len(pts) < 3:
            continue
        yield s["label"], pts


def shapes_to_detect_lines(shapes, class_to_id, W, H):
    return [poly_to_det_line(0, pts, W, H)
            for _label, pts in extract_polygons(shapes)]


def shapes_to_segment_lines(shapes, class_to_id, W, H):
    return [poly_to_seg_line(0, pts, W, H)
            for _label, pts in extract_polygons(shapes)]


def shapes_to_mask(shapes, W, H, binary=True, class_to_id=None):
    """
    Rasterize all polygons into a single-channel mask of size (H, W).
    - binary=True  -> background 0, foreground 255
    - binary=False -> background 0, pixel value = class_id + 1 (multi-class)
    """
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for label, pts in extract_polygons(shapes):
        if binary:
            fill = 255
        else:
            cid = class_to_id.get(label, 0) if class_to_id else 0
            fill = cid + 1
        draw.polygon([(float(x), float(y)) for x, y in pts], fill=fill)
    return mask


# ---------- yaml helpers ---------------------------------------------------- #
def write_yaml(path: Path, dst_root: Path, labels_dir: str,
               class_names, has_test: bool, extra_lines=None):
    lines = [f"path: {dst_root.resolve()}",
             "train: images/train",
             "val: images/val"]
    if has_test:
        lines.append("test: images/test")
    lines.append(f"labels: {labels_dir}")          # non-standard; for reference
    if extra_lines:
        lines.extend(extra_lines)
    lines.append(f"nc: {len(class_names)}")
    lines.append("names: [" + ", ".join(f"'{n}'" for n in class_names) + "]")
    path.write_text("\n".join(lines) + "\n")


# ---------- main pipeline --------------------------------------------------- #
def convert(src, dst, val_ratio, test_ratio, seed, explicit_classes, mask_mode):
    src, dst = Path(src), Path(dst)
    pairs = find_pairs(src)
    if not pairs:
        raise SystemExit(f"No labelme pairs found under {src}")

    class_names = collect_classes(pairs, explicit_classes)
    class_to_id = {n: i for i, n in enumerate(class_names)}

    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    n_train = max(1, n - n_val - n_test) if n > 0 else 0
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val

    splits = {"train": pairs[:n_train],
              "val":   pairs[n_train:n_train + n_val]}
    if n_test:
        splits["test"] = pairs[n_train + n_val:]

    sub_dirs = ["images", "labels_detect", "labels_segment", "masks"]
    for split in splits:
        for sub in sub_dirs:
            (dst / sub / split).mkdir(parents=True, exist_ok=True)

    binary = mask_mode == "binary"

    # outer bar across splits, inner bar across samples
    with tqdm(total=n, desc="Writing dataset", unit="sample") as pbar:
        for split, items in splits.items():
            pbar.set_postfix_str(f"split={split}")
            for img, jp, data in items:
                W = data.get("imageWidth")
                H = data.get("imageHeight")
                if not (W and H):
                    with Image.open(img) as im:
                        W, H = im.size

                shapes = data.get("shapes", [])

                # 1) image
                shutil.copy2(img, dst / "images" / split / img.name)

                # 2) detect label txt
                det_lines = shapes_to_detect_lines(shapes, class_to_id, W, H)
                (dst / "labels_detect" / split / f"{img.stem}.txt").write_text(
                    "\n".join(det_lines) + ("\n" if det_lines else ""))

                # 3) segment label txt
                seg_lines = shapes_to_segment_lines(shapes, class_to_id, W, H)
                (dst / "labels_segment" / split / f"{img.stem}.txt").write_text(
                    "\n".join(seg_lines) + ("\n" if seg_lines else ""))

                # 4) mask PNG
                mask = shapes_to_mask(shapes, W, H,
                                      binary=binary, class_to_id=class_to_id)
                mask.save(dst / "masks" / split / f"{img.stem}.png")

                pbar.update(1)

    has_test = bool(n_test)

    # two separate yamls
    write_yaml(dst / "data_detect.yaml",  dst, "labels_detect",
               class_names, has_test)
    write_yaml(dst / "data_segment.yaml", dst, "labels_segment",
               class_names, has_test,
               extra_lines=["masks_train: masks/train",
                            "masks_val: masks/val"] +
                           (["masks_test: masks/test"] if has_test else []))

    print(f"\nDone. mask_mode={mask_mode}  classes={class_names}")
    print("  " + "  ".join(f"{k}={len(v)}" for k, v in splits.items()))
    print(f"  dataset root: {dst.resolve()}")
    print("  yamls: data_detect.yaml, data_segment.yaml")


# ---------- CLI ------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"C:\Users\alias\Documents\UNI\Thesis\GreenHouse-Dataset_Images+jsonlabels+masks\Greenhousedataset\images", help="folder of labelme pairs")
    ap.add_argument("--dst", default=r"C:\Users\alias\Documents\UNI\Thesis\GreenHouse-Dataset_Images+jsonlabels+masks\clean_yolo_dataset", help="output dataset folder")
    ap.add_argument("--val",  type=float, default=0.2, help="val ratio")
    ap.add_argument("--test", type=float, default=0.0, help="test ratio")
    ap.add_argument("--seed", type=int,   default=42)
    ap.add_argument("--classes", nargs="*", default=["1"],
                    help="fixed class name order; else auto-discover sorted")
    ap.add_argument("--mask-mode", choices=["binary", "class"], default="class",
                    help="binary: 0/255 masks; class: 0=bg, 1..N=class_id+1")
    a = ap.parse_args()
    convert(a.src, a.dst, a.val, a.test, a.seed, a.classes, a.mask_mode)