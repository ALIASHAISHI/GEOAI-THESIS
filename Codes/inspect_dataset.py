import os
from pathlib import Path

# ========================
# User parameters
# ========================
DATASET = "/media/mrizk/DiskB1/GH/GH_Datasets/yolo_dataset"
IMG_EXT = [".jpg", ".jpeg", ".png"]

sets = ["train", "val"]

# ========================
# Function to scan a set
# ========================
def scan_dataset(set_name):
    img_folder = Path(DATASET) / "images" / set_name
    label_folder = Path(DATASET) / "labels" / set_name

    images = [f for f in img_folder.iterdir() if f.suffix.lower() in IMG_EXT]
    labels = [f for f in label_folder.iterdir() if f.suffix.lower() == ".txt"]

    images_with_labels = []
    images_without_labels = []
    total_polygons = 0

    label_names = set([f.stem for f in labels])

    for img in images:
        if img.stem in label_names:
            images_with_labels.append(img)
            # Count polygons in label file
            label_file = label_folder / f"{img.stem}.txt"
            with open(label_file, "r") as f:
                total_polygons += len([l for l in f.readlines() if l.strip()])
        else:
            images_without_labels.append(img)

    print(f"\n=== {set_name.upper()} SET ===")
    print(f"Total images found: {len(images)}")
    print(f"Images with labels: {len(images_with_labels)}")
    print(f"Images without labels: {len(images_without_labels)}")
    print(f"Total polygons (objects): {total_polygons}")

    if images_without_labels:
        print("Images without labels (sample 10):")
        for img in images_without_labels[:10]:
            print("  ", img.name)

# ========================
# Scan both train and val
# ========================
for s in sets:
    scan_dataset(s)
