import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

# ==== CONFIG ====
SRC_ROOT = r"C:\Users\alias\Documents\UNI\Thesis\GreenHouse-Dataset_Images+jsonlabels+masks\Greenhousedataset"
SRC_IMG_DIR = "images"   # folder with imgs + json
SRC_MASK_DIR = "masks"        # folder with masks

DST_ROOT = r"C:\Users\alias\Documents\UNI\Thesis\GreenHouse-Dataset_Images+jsonlabels+masks\clean_unet_dataset"
DST_IMG_DIR = "images"
DST_MASK_DIR = "masks"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ================


def prepare_images(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in src_dir.iterdir()
             if f.is_file() and f.suffix.lower() in IMG_EXTS]
    for f in tqdm(files, desc="Copying images", unit="img"):
        shutil.copy2(f, dst_dir / f.name)
    print(f"Copied {len(files)} images -> {dst_dir}")


def prepare_masks(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in src_dir.iterdir()
             if f.is_file() and f.suffix.lower() in IMG_EXTS]
    for f in tqdm(files, desc="Converting masks", unit="mask"):
        mask = np.array(Image.open(f))

        # convert label 1 -> 255 (keep 0 as background)
        # if you have more classes, extend the mapping below
        mask = np.where(mask == 1, 255, mask).astype(np.uint8)

        Image.fromarray(mask).save(dst_dir / f.name)
    print(f"Converted {len(files)} masks -> {dst_dir}")


def main():
    src_root = Path(SRC_ROOT)
    dst_root = Path(DST_ROOT)

    prepare_images(src_root / SRC_IMG_DIR, dst_root / DST_IMG_DIR)
    prepare_masks(src_root / SRC_MASK_DIR, dst_root / DST_MASK_DIR)

    # sanity check: image/mask filenames should match
    imgs = {p.stem for p in (dst_root / DST_IMG_DIR).iterdir()}
    masks = {p.stem for p in (dst_root / DST_MASK_DIR).iterdir()}
    missing_masks = imgs - masks
    missing_imgs = masks - imgs
    if missing_masks:
        print(f"WARNING: {len(missing_masks)} images have no matching mask")
    if missing_imgs:
        print(f"WARNING: {len(missing_imgs)} masks have no matching image")
    print(f"Paired samples: {len(imgs & masks)}")


if __name__ == "__main__":
    main()