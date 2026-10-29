# # Train U-Net on a Kaggle dataset
# 
# Segmentation training with **segmentation-models-pytorch** and an
# **Albumentations** pipeline (brightness, shadows, zoom in/out, flips, blur, noise).
# 
# **Before you run:** attach your dataset via **Add Input → Datasets**, then
# update `IMAGES_DIR` and `MASKS_DIR` in the config cell.
# 
# Enable GPU: Settings → Accelerator → GPU (T4 x2 or P100).

# ## 1. Install packages
# 
# `torch`, `cv2`, `tqdm` are already on Kaggle.

# !pip install -q segmentation-models-pytorch albumentations

# ## 2. Explore the input directory

# import os
# for root, dirs, files in os.walk("/kaggle/input"):
#     depth = root.replace("/kaggle/input", "").count(os.sep)
#     if depth > 3: continue
#     print("  " * depth + os.path.basename(root) + "/")
#     if depth < 3:
#         for f in files[:5]:
#             print("  " * (depth + 1) + f)
#         if len(files) > 5:
#             print("  " * (depth + 1) + f"... ({len(files)} files total)")

# ## 3. Config
# 
# Images and masks must share filenames. `num_classes=1` → binary, else multi-class.

# ---- EDIT THESE ---------------------------------------------------
IMAGES_DIR   = r"C:\Users\alias\Documents\UNI\Thesis\GreenHouse-Dataset_Images+jsonlabels+masks\Greenhousedataset4\images_cop"
MASKS_DIR    = r"C:\Users\alias\Documents\UNI\Thesis\GreenHouse-Dataset_Images+jsonlabels+masks\Greenhousedataset4\masks_cop"

NUM_CLASSES  = 1        # 1 = binary segmentation
IMG_SIZE     = 256
BATCH_SIZE   = 16
EPOCHS       = 50
LR           = 1e-4
TRAIN_RATIO  = 0.8
ENCODER      = "resnet34"
ENCODER_W    = "imagenet"

SAVE_BEST    = r".\unet_best.pth"
SAVE_FINAL   = r".\unet_final.pth"
# -------------------------------------------------------------------

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

# ## 4. Dataset class

from pathlib import Path
import cv2, numpy as np
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.transform  = transform
        self.images = sorted(f for f in self.images_dir.iterdir()
                             if f.suffix.lower() in self.IMG_EXTS)
        if not self.images:
            raise RuntimeError(f"No images in {self.images_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_dir / (img_path.stem + ".png")

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"missing mask for {img_path.name}")
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return image, mask

# ## 5. Augmentation pipeline
# 
# Brightness, shadows, zoom in/out (`Affine(scale=…)`), flips, blur, noise, erasing.

import albumentations as A
from albumentations.pytorch import ToTensorV2

def train_transform(size=IMG_SIZE):
    return A.Compose([
        A.Resize(size, size),

        # zoom in/out + slight rotate/translate
        A.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1),
                 rotate=(-15, 15), p=0.7),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),

        # brightness / colour
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25,
                             val_shift_limit=20, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.2),

        # shadows & lighting
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.4),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.15),
        A.RandomFog(p=0.1),

        # blur / noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.GaussNoise(p=0.3),

        A.CoarseDropout(p=0.3),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def val_transform(size=IMG_SIZE):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ## 6. Train / val split + loaders

import random
from torch.utils.data import DataLoader, Subset

ds_train_view = SegmentationDataset(IMAGES_DIR, MASKS_DIR, train_transform())
ds_val_view   = SegmentationDataset(IMAGES_DIR, MASKS_DIR, val_transform())

n = len(ds_train_view)
idx = list(range(n)); random.Random(42).shuffle(idx)
cut = int(n * TRAIN_RATIO)

train_set = Subset(ds_train_view, idx[:cut])
val_set   = Subset(ds_val_view,   idx[cut:])
print(f"train: {len(train_set)} | val: {len(val_set)}")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ## 7. Preview a few augmented samples (sanity check)

# import matplotlib.pyplot as plt
# import numpy as np

# def denorm(t):
#     mean = np.array([0.485, 0.456, 0.406])
#     std  = np.array([0.229, 0.224, 0.225])
#     img = t.permute(1, 2, 0).numpy() * std + mean
#     return np.clip(img, 0, 1)

# fig, axes = plt.subplots(2, 4, figsize=(14, 6))
# for i in range(4):
#     img, mask = train_set[i]
#     axes[0, i].imshow(denorm(img)); axes[0, i].set_title("aug image"); axes[0, i].axis("off")
#     axes[1, i].imshow(mask.squeeze(), cmap="gray"); axes[1, i].set_title("mask"); axes[1, i].axis("off")
# plt.tight_layout(); plt.show()

# ## 8. Build U-Net

import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_W,
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)

if NUM_CLASSES == 1:
    criterion = smp.losses.DiceLoss(mode="binary", from_logits=True)
else:
    criterion = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ## 9. Training loop (saves best checkpoint to `/kaggle/working`)

import time
t = time.time()
imgs, masks = next(iter(train_loader))
print(f"loaded in {time.time()-t:.1f}s, imgs={imgs.shape}, masks={masks.shape}")

# ## 10. Preview predictions on val

# model.eval()
# imgs, masks = next(iter(val_loader))
# with torch.no_grad():
#     logits = model(imgs.to(DEVICE))
#     if NUM_CLASSES == 1:
#         preds = (torch.sigmoid(logits) > 0.5).float().cpu()
#     else:
#         preds = logits.argmax(1, keepdim=True).float().cpu()

# n = min(4, imgs.size(0))
# fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
# for i in range(n):
#     axes[0, i].imshow(denorm(imgs[i]));       axes[0, i].set_title("image"); axes[0, i].axis("off")
#     axes[1, i].imshow(masks[i].squeeze(), cmap="gray"); axes[1, i].set_title("gt");    axes[1, i].axis("off")
#     axes[2, i].imshow(preds[i].squeeze(), cmap="gray"); axes[2, i].set_title("pred");  axes[2, i].axis("off")
# plt.tight_layout(); plt.show()