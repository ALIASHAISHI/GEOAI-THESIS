"""
Train an Ultralytics YOLO model with data augmentation.

Pipeline:
  1. Split a flat images/labels dataset into train/val folders
  2. Generate the data.yaml config
  3. Train YOLO with augmentations (brightness, zoom, flips, mosaic, etc.)
  4. Save the best model automatically

Install:
    pip install ultralytics pyyaml
"""

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO


def train_yolo(
    yaml_path: str,
    model_name: str = "yolo11n.pt",   # n / s / m / l / x
    suffix_name: str = "",                # optional suffix for the run name
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = r"c:\runs\train",
    name: str = "exp",
) -> Path:
    """
    Ultralytics has a rich set of built-in augmentations. The params below
    cover the ones asked for (brightness, zoom in/out, etc.).
    """
    # Load a model
    # model = YOLO("yolo26n-sem.yaml")  # build a new model from YAML
    # model = YOLO("yolo26n-sem.pt")  # load a pretrained model (recommended for training)
    model = YOLO(model_name)  # build from YAML and transfer weights

    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,

        # ---- Color / brightness ---------------------------------------
        hsv_h=0.015,   # hue shift
        hsv_s=0.70,    # saturation
        hsv_v=0.40,    # value == brightness

        # ---- Geometric (zoom lives here) ------------------------------
        degrees=10.0,      # rotation ±deg
        translate=0.10,    # shift ±fraction
        scale=0.5,         # zoom in/out ±fraction (0.5 → 50%–150%)
        shear=2.0,
        perspective=0.0001,

        # ---- Flips ----------------------------------------------------
        flipud=0.0,
        fliplr=0.5,

        # ---- Advanced -------------------------------------------------
        mosaic=1.0,          # 4-image mosaic
        mixup=0.15,
        copy_paste=0.10,
        auto_augment="randaugment",
        erasing=0.4,         # random erasing (occlusion)

        # ---- Checkpointing --------------------------------------------
        save=True,
        save_period=10,
        patience=30,         # early stopping
    )

    best = Path(project) / f"{name}{suffix_name}" / "weights" / f"best{suffix_name}.pt"
    print(f"\n✅ Training done. Best weights: {best}")
    return best


# ======================================================================
# Entrypoint
# ======================================================================
if __name__ == "__main__":
    train_yolo(
        r"C:\Users\alias\Documents\UNI\Thesis\GreenHouse-Dataset_Images+jsonlabels+masks\clean_yolo_dataset\data_segment.yaml",
        model_name="yolo12n.pt",  # for detecton: "yolo11n.pt", for segmentation: "yolo26n-sem.pt"
        suffix_name="_segment",
        project=r"C:\Users\alias\Documents\UNI\Thesis",
        epochs=1,
        imgsz=640
    )
    
    # model = YOLO(r"C:\Users\alias\Documents\UNI\Thesis\runs\semantic\runs\train\exp-6\weights\best.pt")  # build from YAML and transfer weights
    # # Predict with the model
    # results = model(r"C:\Users\alias\Pictures\Screenshots\Screenshot 2026-10-04 012319.png")  # predict on an image

    # # Access the results
    # for result in results:
    #     mask  = result.semantic_mask.data  # class map, shape (H,W), integer dtype selected by class count
    #     mask = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)  # tensor -> numpy
    #     mask = mask.squeeze()                                                   # ensure (H, W)

    #     img = result.orig_img[..., ::-1]  # BGR -> RGB

    #     n_classes = len(model.names)
    #     cmap = plt.get_cmap("tab20", n_classes)

    #     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    #     axes[0].imshow(img)
    #     axes[0].set_title("Original")

    #     im = axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=n_classes - 1, interpolation="nearest")
    #     axes[1].set_title("Semantic mask")

    #     axes[2].imshow(img)
    #     axes[2].imshow(mask, cmap=cmap, vmin=0, vmax=n_classes - 1, alpha=0.5, interpolation="nearest")
    #     axes[2].set_title("Overlay")

    #     for ax in axes:
    #         ax.axis("off")

    #     # Legend for the classes actually present in this image
    #     present = np.unique(mask)
    #     handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(int(c))) for c in present]
    #     labels = [model.names.get(int(c), str(c)) for c in present]
    #     axes[1].legend(handles, labels, loc="lower right", fontsize=8)

    #     plt.tight_layout()
    #     plt.show()
