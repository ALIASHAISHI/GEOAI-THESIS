import os
import torch
from ultralytics import YOLO

# Clear VRAM before starting
torch.cuda.empty_cache()

OUTPUT_DATASET = "/media/mrizk/DiskB1/GH/GH_Datasets/yolo_dataset"
yaml_path = os.path.join(OUTPUT_DATASET, "data.yaml")

# Load the model
model = YOLO("yolov8x-seg.pt")

model.train(
    data=yaml_path,
    epochs=100,
    imgsz=384,        
    batch=4,          
    device=0,
    workers=0,        
    cache=False,
    plots=False,
    save_json=False,
    val=False,
    mosaic=0.0,       

    # ==========================================
    # 🔥 CRITICAL FIXES FOR THE NaN / GRADIENT BLOWUP
    # ==========================================
    amp=False,        # Disable Mixed Precision. Forces FP32 to prevent NaN overflows.
    lr0=0.001,        # Lower initial learning rate (Default is 0.01). Prevents exploding gradients with batch=4.
    
    # Segmentation specifics
    mask_ratio=8,
    overlap_mask=False
)

print("Training finished successfully!")
""""


import os
from ultralytics import YOLO
from pathlib import Path
import torch

torch.cuda.empty_cache()

OUTPUT_DATASET = "/media/mrizk/DiskB1/GH/GH_Datasets/yolo_dataset"
yaml_path = f"{OUTPUT_DATASET}/data.yaml"

model = YOLO("yolov8x-seg.pt")

model.train(
    data=yaml_path,
    epochs=100,
    imgsz=384,

    batch=4,
    device=0,

    workers=0,          # reduce RAM usage further
    cache=False,

    amp=True,

    mask_ratio=8,
    overlap_mask=False,

    plots=False,
    save_json=False,

    val=False,
    mosaic=0.0 # Disable mosaic augmentation (VERY IMPORTANT to reduce memory usage Disable mosaic augmentation (VERY IMPORTANT))
    
)

print("Training finished")
"""
