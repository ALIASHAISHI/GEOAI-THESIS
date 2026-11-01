# dataset_to_unetdataset.py — Prepare a U-Net image/mask dataset

Copies the greenhouse images and their segmentation masks into a clean folder structure for U-Net training, converting mask label `1` to `255` so masks are visible and follow the common 0 / 255 binary convention.

---

## What it does

1. **Copies images** from `SRC_ROOT/images` to `DST_ROOT/images` (only files with an image extension — the labelme `.json` files sitting next to them are ignored).
2. **Converts masks** from `SRC_ROOT/masks` to `DST_ROOT/masks`:
   - pixel value `1` → `255`
   - `0` stays `0` (background)
   - any other value is kept unchanged
   - saved as `uint8`, same file name as the source
3. **Sanity check** — compares image and mask file names (without extension) and reports:
   - images with no matching mask
   - masks with no matching image
   - the number of correctly paired samples

---

## Requirements

```bash
pip install numpy pillow tqdm
```

---

## Configuration

Edit the constants at the top of the script:

| Constant | Default | Meaning |
|---|---|---|
| `SRC_ROOT` | `...\Greenhousedataset` | Source dataset root. |
| `SRC_IMG_DIR` | `images` | Sub-folder with images (+ labelme JSON). |
| `SRC_MASK_DIR` | `masks` | Sub-folder with masks (values 0 / 1). |
| `DST_ROOT` | `...\clean_unet_dataset` | Output dataset root. |
| `DST_IMG_DIR` | `images` | Output image sub-folder. |
| `DST_MASK_DIR` | `masks` | Output mask sub-folder. |
| `IMG_EXTS` | `.jpg .jpeg .png .bmp .tif .tiff` | Extensions treated as images / masks. |

---

## Usage

```bash
python dataset_to_unetdataset.py
```

There are no command-line arguments; all settings come from the config block.

### Example output

```
Copying images: 100%|██████████| 1200/1200
Copied 1200 images -> ...\clean_unet_dataset\images
Converting masks: 100%|██████████| 1200/1200
Converted 1200 masks -> ...\clean_unet_dataset\masks
Paired samples: 1200
```

---

## Input / output layout

```
SRC_ROOT/                         DST_ROOT/
├── images/                       ├── images/
│   ├── img_001.jpg       ──►     │   └── img_001.jpg
│   └── img_001.json              └── masks/
└── masks/                            └── img_001.png   (0 / 255)
    └── img_001.png   (0 / 1)
```

Images and masks are paired by **file stem**, so `img_001.jpg` pairs with `img_001.png`.

---

## Notes

- **No train/val split** — the script only copies and converts. Split inside your training code or extend the script.
- **Binary only** — only value `1` is remapped. For multi-class masks, extend the mapping in `prepare_masks()`.
- **Training side** — when loading these masks for a binary U-Net, divide by 255 (or threshold at > 127) to get 0 / 1 targets.
- **Masks are re-saved with their original extension.** JPEG masks would be re-compressed and get blurred edges; keep source masks as PNG/TIFF.
- Existing files in `DST_ROOT` with the same names are overwritten; other files there are left alone (and would show up in the sanity check).
