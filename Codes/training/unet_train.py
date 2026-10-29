"""
Train a U-Net for semantic segmentation with an Albumentations pipeline.

Expects a *prepared* dataset already split into train/val:

    <images_root>/train/*.jpg   <masks_root>/train/*.png
    <images_root>/val/*.jpg     <masks_root>/val/*.png

Image and mask are paired by filename stem (e.g. tile_r14_c17.jpg ↔ tile_r14_c17.png).

Install:
    pip install torch torchvision segmentation-models-pytorch \
                albumentations opencv-python tqdm
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ======================================================================
# 1. Dataset
# ======================================================================
class SegmentationDataset(Dataset):
    """
    Pairs images with masks by filename stem, so `tile_r14_c17.jpg` matches
    `tile_r14_c17.png`. Only images with a matching mask are kept.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"images_dir does not exist: {self.images_dir}")
        if not self.masks_dir.is_dir():
            raise FileNotFoundError(f"masks_dir does not exist: {self.masks_dir}")

        # Build stem -> mask-path index once
        mask_index = {}
        for m in self.masks_dir.iterdir():
            if m.suffix.lower() in self.IMG_EXTS:
                mask_index.setdefault(m.stem, m)

        # Keep only images that have a corresponding mask
        self.pairs = []
        skipped = 0
        for img in sorted(self.images_dir.iterdir()):
            if img.suffix.lower() not in self.IMG_EXTS:
                continue
            mask = mask_index.get(img.stem)
            if mask is None:
                skipped += 1
                continue
            self.pairs.append((img, mask))

        if not self.pairs:
            raise RuntimeError(
                f"No image/mask pairs found.\n"
                f"  images_dir: {self.images_dir}\n"
                f"  masks_dir:  {self.masks_dir}\n"
                f"Check that mask filenames share the stem "
                f"(e.g. tile_r14_c17.jpg <-> tile_r14_c17.png)."
            )
        if skipped:
            print(f"[dataset] warning: {skipped} images in {self.images_dir.name} "
                  f"had no matching mask and were skipped")
        print(f"[dataset] {len(self.pairs)} pairs in "
              f"{self.images_dir.parent.name}/{self.images_dir.name}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask {mask_path}")
        mask = (mask > 127).astype(np.float32)  # binarise; drop for multi-class

        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # [1, H, W]
        return image, mask


# ======================================================================
# 2. Augmentation pipelines
# ======================================================================
def train_transform(img_size: int = 256) -> A.Compose:
    """Brightness, shadows, zoom in/out, flips, blur, noise, erasing."""
    return A.Compose([
        A.Resize(img_size, img_size),

        # --- Zoom in/out + slight rotate/translate (Affine) -----------
        A.Affine(
            scale=(0.7, 1.3),
            translate_percent=(-0.1, 0.1),
            rotate=(-15, 15),
            p=0.7,
        ),

        # --- Flips ----------------------------------------------------
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),

        # --- Brightness / contrast / colour ---------------------------
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.7,
        ),
        A.HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.5,
        ),
        A.CLAHE(clip_limit=2.0, p=0.2),

        # --- Shadows and lighting artefacts ---------------------------
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.4),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.15),
        A.RandomFog(p=0.1),

        # --- Blur / noise ---------------------------------------------
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.GaussNoise(p=0.3),

        # --- Random erasing / occlusion -------------------------------
        A.CoarseDropout(p=0.3),

        # --- Normalise + tensor ---------------------------------------
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def val_transform(img_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ======================================================================
# 3. Loaders from prepared train/val folders
# ======================================================================
def build_loaders(
    images_root,
    masks_root,
    img_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 2,
    train_subdir: str = "train",
    val_subdir: str = "val",
):
    """
    Build train/val loaders from a pre-split dataset:

        <images_root>/<train_subdir>/*.jpg   <masks_root>/<train_subdir>/*.png
        <images_root>/<val_subdir>/*.jpg     <masks_root>/<val_subdir>/*.png
    """
    images_root = Path(images_root)
    masks_root = Path(masks_root)

    train_set = SegmentationDataset(
        images_root / train_subdir,
        masks_root / train_subdir,
        transform=train_transform(img_size),
    )
    val_set = SegmentationDataset(
        images_root / val_subdir,
        masks_root / val_subdir,
        transform=val_transform(img_size),
    )
    print(f"[split] train: {len(train_set)}  val: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ======================================================================
# 4. Training
# ======================================================================
def train_unet(
    images_root: str,
    masks_root: str,
    encoder: str = "resnet34",
    encoder_weights: str = "imagenet",
    num_classes: int = 1,          # 1 = binary; N = multi-class
    img_size: int = 256,
    batch_size: int = 8,
    epochs: int = 50,
    lr: float = 1e-4,
    num_workers: int = 2,
    train_subdir: str = "train",
    val_subdir: str = "val",
    save_path: str = "unet_best.pth",
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    train_loader, val_loader = build_loaders(
        images_root=images_root,
        masks_root=masks_root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        train_subdir=train_subdir,
        val_subdir=val_subdir,
    )

    # --- Model ---------------------------------------------------------
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    ).to(device)

    # --- Loss ----------------------------------------------------------
    if num_classes == 1:
        criterion = smp.losses.DiceLoss(mode="binary", from_logits=True)
    else:
        criterion = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        run_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"epoch {epoch}/{epochs} [train]"):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if num_classes > 1:
                masks = masks.squeeze(1).long()

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * imgs.size(0)
        train_loss = run_loss / len(train_loader.dataset)

        # ---- Val ----
        model.eval()
        run_loss = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"epoch {epoch}/{epochs} [val]"):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                if num_classes > 1:
                    masks = masks.squeeze(1).long()

                preds = model(imgs)
                loss = criterion(preds, masks)
                run_loss += loss.item() * imgs.size(0)
        val_loss = run_loss / len(val_loader.dataset)

        scheduler.step()
        print(f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        # ---- Save best ----
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "encoder": encoder,
                    "num_classes": num_classes,
                    "img_size": img_size,
                },
                save_path,
            )
            print(f"  -> new best, saved to {save_path}")

    # Also save the final model
    torch.save(model.state_dict(), "unet_final.pth")
    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Best model -> {save_path} | Final model -> unet_final.pth")


# ======================================================================
# Entrypoint
# ======================================================================
if __name__ == "__main__":
    # Point these at the PARENT folders that each contain train/ and val/
    train_unet(
        images_root=r"C:\Users\alias\Documents\UNI\Thesis\QGIS-Project\unet_dataset\images",
        masks_root=r"C:\Users\alias\Documents\UNI\Thesis\QGIS-Project\unet_dataset\masks",
        encoder="resnet34",
        num_classes=1,          # binary segmentation
        img_size=256,
        batch_size=8,
        epochs=50,
        lr=1e-4,
        save_path="unet_best.pth",
    )