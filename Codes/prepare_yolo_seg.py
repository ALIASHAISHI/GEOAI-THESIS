import os
import shutil
from pathlib import Path
from random import shuffle

# =========================
# USER PARAMETERS
# =========================

DATASET_ROOT = "/media/mrizk/DiskB1/GH/GH_Datasets"
OUTPUT_DATASET = "/media/mrizk/DiskB1/GH/GH_Datasets/yolo_dataset"

VAL_RATIO = 0.1
IMG_EXT = [".jpg", ".jpeg", ".png"]

CLASS_NAMES = ["GH"]

LOG_FILE = OUTPUT_DATASET + "/dataset_training_log.txt"

# =========================
# YOLO folder structure
# =========================

train_img_folder = Path(OUTPUT_DATASET) / "images/train"
val_img_folder = Path(OUTPUT_DATASET) / "images/val"
train_label_folder = Path(OUTPUT_DATASET) / "labels/train"
val_label_folder = Path(OUTPUT_DATASET) / "labels/val"

for folder in [train_img_folder, val_img_folder, train_label_folder, val_label_folder]:
    folder.mkdir(parents=True, exist_ok=True)

# ===============================
# Collect all images with labels
# ===============================

all_images = []

for i in range(1, 12):

    folder = Path(DATASET_ROOT) / f"images{i}"

    if not folder.exists():
        continue

    print("Scanning:", folder)

    for img_file in folder.iterdir():

        if img_file.suffix.lower() in IMG_EXT:

            label_file = folder / (img_file.stem + ".txt")

            if label_file.exists():

                # store folder name to avoid filename collisions later
                all_images.append((folder.name, img_file, label_file))

print(f"Total labeled images found: {len(all_images)}")

# ===============================
# Shuffle and split
# ===============================

shuffle(all_images)

num_val = int(len(all_images) * VAL_RATIO)

val_set = all_images[:num_val]
train_set = all_images[num_val:]

# ===============================
# Copy files safely
# ===============================

def copy_dataset(image_list, img_dest, label_dest):

    polygons_counts = []
    max_polygons = 0

    for folder_name, img_path, label_path in image_list:

        print(folder_name)
        # prevent overwrite by prefixing folder name
        new_img_name = f"{folder_name}_{img_path.name}"
        new_label_name = f"{folder_name}_{label_path.name}"

        shutil.copy2(img_path, img_dest / new_img_name)
        shutil.copy2(label_path, label_dest / new_label_name)

        # count polygons
        with open(label_path, "r") as f:
            lines = [l for l in f.readlines() if l.strip()]
            polygons = len(lines)

        polygons_counts.append(polygons)

        if polygons > max_polygons:
            max_polygons = polygons

    total_polygons = sum(polygons_counts)

    avg_polygons = total_polygons / len(image_list) if len(image_list) > 0 else 0

    return len(image_list), total_polygons, max_polygons, avg_polygons


train_count, train_polygons, train_max, train_avg = copy_dataset(
    train_set, train_img_folder, train_label_folder
)

val_count, val_polygons, val_max, val_avg = copy_dataset(
    val_set, val_img_folder, val_label_folder
)

# ===============================
# Save log
# ===============================

with open(LOG_FILE, "w") as f:

    f.write("YOLO Segmentation Dataset Training Log\n")
    f.write("=====================================\n\n")

    f.write(f"Total labeled images found: {len(all_images)}\n\n")

    f.write("Training Set\n")
    f.write("-----------------\n")
    f.write(f"Images: {train_count}\n")
    f.write(f"Polygons (objects): {train_polygons}\n")
    f.write(f"Max polygons in one image: {train_max}\n")
    f.write(f"Average polygons per image: {train_avg:.2f}\n\n")

    f.write("Validation Set\n")
    f.write("-----------------\n")
    f.write(f"Images: {val_count}\n")
    f.write(f"Polygons (objects): {val_polygons}\n")
    f.write(f"Max polygons in one image: {val_max}\n")
    f.write(f"Average polygons per image: {val_avg:.2f}\n\n")

print(f"Dataset split done. Log saved to {LOG_FILE}")
print(f"Train images: {train_count}, Validation images: {val_count}")
