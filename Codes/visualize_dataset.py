"""
Visualize samples from the greenhouse dataset.

Shows for a chosen image index:
  1. Original image
  2. Image + mask overlay
  3. Image + bounding boxes drawn from the .json label
      (falls back to bboxes computed from the mask if json is missing/unparseable)
  4. A grid of augmented (image + mask) samples using the same augmentation
     pipeline as training, so you can eyeball what the model will actually see.

Usage in a Kaggle / Jupyter cell:

    visualize(index=0, n_augmentations=9)           # first image, 9 augs
    visualize(index=42, n_augmentations=12, cols=4) # 43rd image, 12 augs in 3 rows of 4
    show_sample(index=0)                            # just the top row
    show_augmentations(index=0, n=16, cols=4)       # just the augmentation grid

Install (same deps as training):
    pip install albumentations opencv-python matplotlib
"""

import json
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# CONFIG -- change these to point at your dataset
# ---------------------------------------------------------------------------
# IMAGES_DIR = r"/kaggle/input/datasets/roudybarakat/greenhouse-dataset-images-jsonlabels-masks/Greenhousedataset/images"
# MASKS_DIR  = r"/kaggle/input/datasets/roudybarakat/greenhouse-dataset-images-jsonlabels-masks/Greenhousedataset/masks"
IMAGES_DIR = r"C:\Users\alias\Documents\UNI\Thesis\GreenHouse-Dataset_Images+jsonlabels+masks\Greenhousedataset\images"
MASKS_DIR  = r"C:\Users\alias\Documents\UNI\Thesis\GreenHouse-Dataset_Images+jsonlabels+masks\Greenhousedataset\masks"
# C:\Users\alias\Documents\UNI\Thesis\GreenHouse-Dataset_Images+jsonlabels+masks
IMG_SIZE   = 256

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# 1. Discover (image, mask, json) triples
# ---------------------------------------------------------------------------
def find_samples(images_dir=IMAGES_DIR, masks_dir=MASKS_DIR):
    """
    Return a sorted list of (image_path, mask_path, json_path_or_None).
    Only images that have a matching mask are kept. .json files in the
    images folder are picked up as sibling labels when the stem matches.
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    mask_index = {
        m.stem: m for m in masks_dir.iterdir()
        if m.is_file() and m.suffix.lower() in IMG_EXTS
    }
    json_index = {
        j.stem: j for j in images_dir.iterdir()
        if j.is_file() and j.suffix.lower() == ".json"
    }

    samples = []
    for img in sorted(images_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue
        mask = mask_index.get(img.stem)
        if mask is None:
            continue
        samples.append((img, mask, json_index.get(img.stem)))
    return samples


# ---------------------------------------------------------------------------
# 2. Parse LabelMe-style JSON -> list of shapes with bboxes
# ---------------------------------------------------------------------------
def parse_labelme(json_path):
    """
    LabelMe-style JSON is the most common format for image+json pairs:
        {"shapes": [{"label", "points": [[x,y], ...], "shape_type"}], ...}

    Returns a list of dicts: {label, points (np.ndarray), bbox (x1,y1,x2,y2), shape_type}
    """
    with open(json_path) as f:
        data = json.load(f)

    out = []
    for s in data.get("shapes", []):
        pts = np.array(s.get("points", []), dtype=float)
        if pts.size == 0:
            continue
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        out.append({
            "label": str(s.get("label", "")),
            "points": pts,
            "bbox": (float(x1), float(y1), float(x2), float(y2)),
            "shape_type": s.get("shape_type", "polygon"),
        })
    return out


def bboxes_from_mask(mask, min_area=50):
    """Fallback: compute bboxes from connected components in the mask."""
    binary = (mask > 127).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, n_labels):        # 0 is background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append({
            "label": "obj",
            "points": None,
            "bbox": (float(x), float(y), float(x + w), float(y + h)),
            "shape_type": "rectangle",
        })
    return boxes


# ---------------------------------------------------------------------------
# 3. Augmentation pipeline for visualization
#    (same as training, minus Normalize + ToTensorV2 so we can plot it)
#    Keep this in sync with train_transform() in the training script.
# ---------------------------------------------------------------------------
def viz_augment(img_size=IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1),
                 rotate=(-15, 15), p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25,
                             val_shift_limit=20, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.2),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.4),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.15),
        A.RandomFog(p=0.1),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.GaussNoise(p=0.3),
        A.CoarseDropout(p=0.3),
    ])


# ---------------------------------------------------------------------------
# 4. Loaders + display helpers
# ---------------------------------------------------------------------------
def _load_sample(samples, index):
    if not samples:
        raise RuntimeError(
            "No image/mask pairs found. Check IMAGES_DIR / MASKS_DIR at the top."
        )
    if not 0 <= index < len(samples):
        raise IndexError(
            f"index {index} out of range. Dataset has {len(samples)} samples "
            f"(valid indices: 0..{len(samples) - 1})."
        )
    img_path, mask_path, json_path = samples[index]

    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask {mask_path}")

    return img_path, mask_path, json_path, image, mask


def _get_boxes(json_path, mask):
    """Try JSON first, fall back to bboxes from mask."""
    boxes = []
    if json_path is not None:
        try:
            boxes = parse_labelme(json_path)
        except Exception as e:
            print(f"[warn] failed to parse {json_path.name}: {e}")
    if not boxes:
        print("[info] no boxes from json -> deriving from mask")
        boxes = bboxes_from_mask(mask)
    return boxes


def _to_display(arr):
    """Convert an augmentation output to something imshow can render cleanly."""
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    if arr.max() <= 1.5:                    # already 0..1 float
        return np.clip(arr, 0, 1)
    return np.clip(arr, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 5. Public API: show_sample / show_augmentations / visualize
# ---------------------------------------------------------------------------
def show_sample(index=0, images_dir=IMAGES_DIR, masks_dir=MASKS_DIR):
    """Top row: original image | image + mask overlay | image + bboxes from json."""
    samples = find_samples(images_dir, masks_dir)
    img_path, mask_path, json_path, image, mask = _load_sample(samples, index)

    print(f"[sample {index} / {len(samples) - 1}]")
    print(f"  image: {img_path.name}   shape={image.shape}")
    print(f"  mask : {mask_path.name}  shape={mask.shape}")
    print(f"  json : {json_path.name if json_path else '(none)'}")

    boxes = _get_boxes(json_path, mask)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title(f"#{index}  {img_path.name}")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(mask, cmap="Reds", alpha=0.4)
    axes[1].set_title("image + mask overlay")
    axes[1].axis("off")

    axes[2].imshow(image)
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        axes[2].add_patch(rect)
        if b["label"]:
            axes[2].text(
                x1, max(y1 - 5, 0), b["label"],
                color="lime", fontsize=10, weight="bold",
                bbox=dict(facecolor="black", alpha=0.5, pad=1),
            )
    axes[2].set_title(f"image + {len(boxes)} bbox(es) from label")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def show_augmentations(
    index=0,
    n=9,
    cols=3,
    img_size=IMG_SIZE,
    images_dir=IMAGES_DIR,
    masks_dir=MASKS_DIR,
    seed=None,
):
    """Grid of `n` augmented (image, mask) samples for a chosen image index."""
    samples = find_samples(images_dir, masks_dir)
    img_path, _mask_path, _json_path, image, mask = _load_sample(samples, index)
    mask_f = (mask > 127).astype(np.float32)

    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)

    transform = viz_augment(img_size)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_1d(axes).ravel()

    for i in range(n):
        out = transform(image=image, mask=mask_f)
        aug_img = _to_display(out["image"])
        aug_mask = out["mask"]
        axes[i].imshow(aug_img)
        axes[i].imshow(aug_mask, cmap="Reds", alpha=0.35)
        axes[i].set_title(f"aug #{i + 1}")
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        f"Augmentations of #{index}  ({img_path.name})",
        y=1.02, fontsize=14,
    )
    plt.tight_layout()
    plt.show()


def visualize(
    index=0,
    n_augmentations=9,
    cols=3,
    img_size=IMG_SIZE,
    images_dir=IMAGES_DIR,
    masks_dir=MASKS_DIR,
):
    """One-call convenience: show the sample + its augmentations."""
    show_sample(index=index, images_dir=images_dir, masks_dir=masks_dir)
    show_augmentations(
        index=index,
        n=n_augmentations,
        cols=cols,
        img_size=img_size,
        images_dir=images_dir,
        masks_dir=masks_dir,
    )


# ---------------------------------------------------------------------------
# Example: first image + 9 augmentations.
# Change `index` (which image) and `n_augmentations` (how many samples).
# ---------------------------------------------------------------------------
visualize(index=0, n_augmentations=9, cols=3)
