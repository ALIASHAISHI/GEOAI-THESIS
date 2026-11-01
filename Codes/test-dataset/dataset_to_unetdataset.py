import random
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

# ==== CONFIG ====
SRC_ROOT = r"C:\Users\alias\Documents\UNI\Thesis\GreenHouse-Dataset_Images+jsonlabels+masks\Greenhousedataset"
SRC_IMG_DIR = "images"   # folder with imgs + json
SRC_MASK_DIR = "masks"   # folder with masks

DST_ROOT = r"C:\Users\alias\Documents\UNI\Thesis\GreenHouse-Dataset_Images+jsonlabels+masks\clean_unet_dataset"
DST_IMG_DIR = "images"
DST_MASK_DIR = "masks"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# split ratios (must sum to 1.0)
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42   # fixed seed -> same split every run
# ================


def list_files(folder: Path) -> dict:
    """Return {stem: path} for all image files in a folder."""
    return {f.stem: f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMG_EXTS}


def convert_mask(src: Path, dst: Path):
    mask = np.array(Image.open(src))
    # convert label 1 -> 255 (keep 0 as background)
    # if you have more classes, extend the mapping below
    mask = np.where(mask == 1, 255, mask).astype(np.uint8)
    Image.fromarray(mask).save(dst)


def split_stems(stems: list) -> dict:
    """Shuffle and split stems into train/val/test according to SPLITS."""
    assert abs(sum(SPLITS.values()) - 1.0) < 1e-6, "SPLITS must sum to 1.0"
    stems = sorted(stems)                 # deterministic order before shuffle
    random.Random(SEED).shuffle(stems)

    n = len(stems)
    n_train = int(round(n * SPLITS["train"]))
    n_val = int(round(n * SPLITS["val"]))
    # test gets the remainder so nothing is lost to rounding
    return {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }


def main():
    src_root = Path(SRC_ROOT)
    dst_root = Path(DST_ROOT)

    imgs = list_files(src_root / SRC_IMG_DIR)
    masks = list_files(src_root / SRC_MASK_DIR)

    # sanity check: image/mask filenames should match
    missing_masks = imgs.keys() - masks.keys()
    missing_imgs = masks.keys() - imgs.keys()
    if missing_masks:
        print(f"WARNING: {len(missing_masks)} images have no matching mask (skipped)")
    if missing_imgs:
        print(f"WARNING: {len(missing_imgs)} masks have no matching image (skipped)")

    paired = list(imgs.keys() & masks.keys())
    print(f"Paired samples: {len(paired)}")
    if not paired:
        return

    splits = split_stems(paired)

    for split_name, stems in splits.items():
        img_out = dst_root / split_name / DST_IMG_DIR
        mask_out = dst_root / split_name / DST_MASK_DIR
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        for stem in tqdm(stems, desc=f"{split_name:>5}", unit="pair"):
            img_src = imgs[stem]
            mask_src = masks[stem]
            shutil.copy2(img_src, img_out / img_src.name)
            convert_mask(mask_src, mask_out / mask_src.name)

        print(f"{split_name}: {len(stems)} pairs -> {dst_root / split_name}")


if __name__ == "__main__":
    main()