

import os
from pathlib import Path

DATASET_ROOT = "/media/mrizk/DiskB1/GH/GH_Datasets"   # change to your dataset path

IMG_EXT = [".jpg", ".jpeg", ".png", ".bmp"]

OUTPUT_WITH_LABELS = "images_with_labels.txt"
OUTPUT_WITHOUT_LABELS = "images_without_labels.txt"
LOG_FILE = "dataset_log.txt"

images_with_labels = []
images_without_labels = []

total_images = 0
total_with_labels = 0
total_without_labels = 0
total_polygons = 0

log_lines = []

folders = [f"images{i}" for i in range(1, 12)]

for folder in folders:

    folder_path = os.path.join(DATASET_ROOT, folder)

    if not os.path.exists(folder_path):
        continue
    
    print(folder)
    folder_images = 0
    folder_with_labels = 0
    folder_without_labels = 0
    folder_polygons = 0

    for file in os.listdir(folder_path):
        print(file)
        ext = Path(file).suffix.lower()

        if ext not in IMG_EXT:
            continue

        folder_images += 1
        image_path = os.path.join(folder_path, file)

        label_name = Path(file).stem + ".txt"
        label_path = os.path.join(folder_path, label_name)

        if os.path.exists(label_path):

            images_with_labels.append(image_path)
            folder_with_labels += 1

            with open(label_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                folder_polygons += len(lines)

        else:

            images_without_labels.append(image_path)
            folder_without_labels += 1

    total_images += folder_images
    total_with_labels += folder_with_labels
    total_without_labels += folder_without_labels
    total_polygons += folder_polygons

    log_lines.append(f"\nFolder: {folder}")
    log_lines.append(f"Total images: {folder_images}")
    log_lines.append(f"Images with labels: {folder_with_labels}")
    log_lines.append(f"Images without labels: {folder_without_labels}")
    log_lines.append(f"Polygons in folder: {folder_polygons}")

# Save image lists

with open(OUTPUT_WITH_LABELS, "w") as f:
    for img in images_with_labels:
        f.write(img + "\n")

with open(OUTPUT_WITHOUT_LABELS, "w") as f:
    for img in images_without_labels:
        f.write(img + "\n")

# Save log file

with open(LOG_FILE, "w") as f:

    f.write("YOLO Segmentation Dataset Report\n")
    f.write("="*50 + "\n")

    for line in log_lines:
        f.write(line + "\n")

    f.write("\nSUMMARY\n")
    f.write("-"*50 + "\n")
    f.write(f"Total images: {total_images}\n")
    f.write(f"Images with labels: {total_with_labels}\n")
    f.write(f"Images without labels: {total_without_labels}\n")
    f.write(f"Total polygons (objects): {total_polygons}\n")

print("Dataset scan completed.")
print("Images with labels saved to:", OUTPUT_WITH_LABELS)
print("Images without labels saved to:", OUTPUT_WITHOUT_LABELS)
print("Log file saved to:", LOG_FILE)