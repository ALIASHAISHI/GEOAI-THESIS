"""
Convert a labelme dataset (paired image + .json) into a YOLO-ready dataset.

    dataset/
    ├── data.yaml
    ├── images/{train,val[,test]}/
    └── labels/{train,val[,test]}/

Usage
-----
    python labelme2yolo.py --src /path/to/labelme --dst /path/to/yolo_ds \
                           --task segment --val 0.2 --seed 42

Then train with Ultralytics:
    yolo segment train data=/path/to/yolo_ds/data.yaml model=yolo11n-seg.pt
    yolo detect  train data=/path/to/yolo_ds/data.yaml model=yolo11n.pt
"""
import argparse
import json
import random
import shutil
from pathlib import Path

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# ---------- discovery ------------------------------------------------------- #
def find_pairs(src: Path):
    """Return list of (image_path, json_path, json_data) for every .json that
    has a sibling image."""
    pairs = []
    for jp in src.rglob("*.json"):
        with open(jp) as f:
            data = json.load(f)
        img = jp.parent / data.get("imagePath", "")
        if not img.exists():
            for ext in IMG_EXTS:                    # fall back to same-stem match
                cand = jp.with_suffix(ext)
                if cand.exists():
                    img = cand
                    break
        if img.exists():
            pairs.append((img, jp, data))
        else:
            print(f"  skip {jp.name}: no matching image")
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


def shapes_to_yolo_lines(shapes, class_to_id, W, H, task):
    lines = []
    for s in shapes:
        # if s["label"] not in class_to_id:
        #     continue
        pts = s["points"]
        st  = s["shape_type"]
        if st == "rectangle":
            pts = rect_to_poly(pts)
        elif st != "polygon":
            continue                                 # skip point / line / circle
        if len(pts) < 3:
            continue
        cid = 0 # s["label"] # class_to_id[s["label"]]
        lines.append(
            poly_to_seg_line(cid, pts, W, H) if task == "segment"
            else poly_to_det_line(cid, pts, W, H)
        )
    return lines


# ---------- main pipeline --------------------------------------------------- #
def convert(src, dst, task, val_ratio, test_ratio, seed, explicit_classes):
    src, dst = Path(src), Path(dst)
    pairs = find_pairs(src)
    if not pairs:
        raise SystemExit(f"No labelme pairs found under {src}")

    class_names = collect_classes(pairs, explicit_classes)
    class_to_id = {n: i for i, n in enumerate(class_names)}

    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_val  = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    n_train = max(1, n - n_val - n_test) if n > 0 else 0
    # clip in case rounding overshoots
    n_val  = min(n_val,  n - n_train)
    n_test = n - n_train - n_val

    splits = {"train": pairs[:n_train],
              "val":   pairs[n_train:n_train + n_val]}
    if n_test:
        splits["test"] = pairs[n_train + n_val:]

    for split in splits:
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, items in splits.items():
        for img, jp, data in items:
            W = data.get("imageWidth")
            H = data.get("imageHeight")
            if not (W and H):                        # fallback: read from image
                from PIL import Image
                with Image.open(img) as im:
                    W, H = im.size
            shutil.copy2(img, dst / "images" / split / img.name)
            label = dst / "labels" / split / (img.stem + ".txt")
            lines = shapes_to_yolo_lines(data.get("shapes", []),
                                         class_to_id, W, H, task)
            label.write_text("\n".join(lines) + ("\n" if lines else ""))

    # data.yaml
    yaml = [f"path: {dst.resolve()}",
            "train: images/train",
            "val: images/val"]
    if n_test:
        yaml.append("test: images/test")
    yaml.append(f"nc: {len(class_names)}")
    yaml.append("names: [" + ", ".join(f"'{n}'" for n in class_names) + "]")
    (dst / "data.yaml").write_text("\n".join(yaml) + "\n")

    print(f"\nDone. task={task}  classes={class_names}")
    print("  " + "  ".join(f"{k}={len(v)}" for k, v in splits.items()))
    print(f"  dataset root: {dst.resolve()}")


# ---------- CLI ------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"C:\Users\alias\Documents\UNI\Thesis\GreenHouse-Dataset_Images+jsonlabels+masks\Greenhousedataset\images", help="folder of labelme pairs")
    ap.add_argument("--dst", default=r"C:\Users\alias\Documents\UNI\Thesis\GreenHouse-Dataset_Images+jsonlabels+masks\clean_yolo_dataset", help="output dataset folder")
    ap.add_argument("--task", choices=["detect", "segment"], default="detect")
    ap.add_argument("--val",  type=float, default=0.2, help="val ratio")
    ap.add_argument("--test", type=float, default=0.0, help="test ratio")
    ap.add_argument("--seed", type=int,   default=42)
    ap.add_argument("--classes", nargs="*", default="1",
                    help="fixed class name order; else auto-discover sorted")
    a = ap.parse_args()
    convert(a.src, a.dst, a.task, a.val, a.test, a.seed, a.classes)