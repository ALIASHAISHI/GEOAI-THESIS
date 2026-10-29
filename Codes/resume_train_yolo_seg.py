import os
from ultralytics import YOLO
from pathlib import Path

# =============================
# DATASET
# =============================

OUTPUT_DATASET = "/media/mrizk/DiskB1/GH/GH_Datasets/yolo_dataset"
yaml_path = f"{OUTPUT_DATASET}/data.yaml"

# =============================
# RESUME FROM LAST CHECKPOINT
# =============================

checkpoint = "runs/segment/train/weights/last.pt"

if not Path(checkpoint).exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

model = YOLO(checkpoint)

model.train(
    data=yaml_path,
    epochs=100,          # total target epochs
    imgsz=640,
    batch=-1,
    device=0,
    workers=4,
    cache=False,
    plots=False,
    amp=True,
    resume=True          # IMPORTANT
)

print("Training resumed and finished")
