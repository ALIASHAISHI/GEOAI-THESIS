# GEOAI-THESIS — Greenhouse Detection from Aerial & Satellite Imagery

Code for a thesis on finding and outlining **agricultural greenhouses** in aerial/satellite imagery with deep learning, and putting the results back on the map as GIS polygons.

The project covers the whole chain:

1. **Dataset building**: cut georeferenced rasters (GeoTIFF + shapefile of greenhouse polygons) into training tiles, or convert LabelMe-annotated images into YOLO / U-Net datasets.
2. **Model training**: YOLO (detection, instance segmentation and semantic segmentation) and U-Net (`segmentation-models-pytorch`).
3. **Foundation-model experiments**: Segment Anything (SAM, MobileSAM, SAM 2, SAM 3), used alone or prompted with YOLO boxes.
4. **GIS deployment**: a QGIS Processing tool that tiles any raster or web map layer, runs the model, and writes georeferenced greenhouse polygons with their area.

---

## Repository layout

```
GEOAI-THESIS/
├── Codes/
│   ├── geomap_ToYolo.py / geomap_ToYolo2.py   # GeoTIFF + shapefile → YOLO-seg tiles (+ LabelMe JSON)
│   ├── jasonToYolo.py                         # LabelMe JSON → YOLO polygon labels
│   ├── prepareList.py                         # Scan images1..images11, list labelled vs unlabelled images
│   ├── prepare_yolo_seg.py                    # Build train/val YOLO-seg dataset (class "GH", 10% val)
│   ├── inspect_dataset.py                     # Count images, labels and polygons per split
│   ├── visualize_dataset.py                   # Image / mask / bbox overlays + augmentation preview
│   ├── train_yolo_seg.py                      # YOLOv8x-seg training (FP32, low LR: NaN-safe settings)
│   ├── resume_train_yolo_seg.py               # Resume a YOLO-seg run from last.pt
│   │
│   ├── test-dataset/
│   │   ├── dataset_to_yolodataset.py          # LabelMe → YOLO detect + segment labels + PNG masks
│   │   ├── dataset_to_unetdataset.py          # Clean image/mask pairs for U-Net (mask 1 → 255)
│   │   └── dataset-displayer.py               # Show a raster with its LabelMe polygons
│   │
│   ├── training/
│   │   ├── yolo_train.py                      # Ultralytics YOLO training with augmentations
│   │   ├── unet_train.py                      # U-Net training (smp + Albumentations)
│   │   ├── yolo_kaggle.ipynb                  # YOLO training notebook for Kaggle GPUs
│   │   └── unet_kaggle.ipynb                  # U-Net training notebook for Kaggle GPUs
│   │
│   ├── test-sam/
│   │   ├── SAM-GEOAI-v0 … V3.ipynb            # MobileSAM experiments
│   │   ├── SAM-GEOAI-ULTRALYTICS-V1/V2.ipynb  # SAM / SAM 2 / SAM 2.1 comparison (+ sharpened inputs)
│   │   ├── SAM3-GEOAI-V1*.ipynb               # SAM 3 with text / box prompts
│   │   ├── yolo_sam3_greenhouses*.ipynb       # YOLO boxes → SAM 3 masks, with full evaluation
│   │   ├── sam_downloader.ipynb               # Download all Ultralytics SAM checkpoints
│   │   └── FSRCNN_x3.pb                       # OpenCV super-resolution model (x3 upscaling)
│   │
│   └── pipelines/                             # Earlier versions of the GIS pipeline
│       ├── greenhouse pipeline.py             # Stand-alone CLI: tile → predict → georeference
│       └── greenhouse_pipeline_qgis_v1…v3.py  # QGIS Processing versions (v2 adds XYZ/WMS, v3 adds AOI)
│
├── QGIS_Script/
│   ├── detect_greenhouses_v3.py               # ★ Main QGIS tool: YOLO / U-Net / YOLO-semantic / SAM 3
│   └── greenhouse_georef_test.py              # Georeferencing round-trip test (IoU check, no model)
│
└── results/
    ├── yolov8n-train/, yolov8s-train/         # YOLOv8-seg training runs (args.yaml + results.csv)
    ├── exp/, exp-2/                           # YOLO26x-sem and YOLO12n test runs (1 epoch)
    └── outputs/                               # YOLO vs SAM 3 evaluation metrics (CSV)
```

Datasets, model weights (`*.pt`, `*.pth`), images and the QGIS project are not in the repository (see `.gitignore`).

---

## Installation

Python 3.10+ and a CUDA GPU are recommended.

```bash
# Training and dataset tools
pip install ultralytics torch torchvision segmentation-models-pytorch \
            albumentations opencv-python pillow numpy tqdm matplotlib pyyaml

# Geospatial dataset generation (geomap_ToYolo*.py, CLI pipeline)
pip install rasterio geopandas shapely pyproj
```

SAM 3 and YOLO semantic models (`yolo26*-sem`) need a recent Ultralytics release: `pip install -U ultralytics`.

> **Paths:** most scripts have hard-coded dataset paths at the top of the file (`DATASET_ROOT`, `IMAGES_DIR`, `tif_path`, …). Edit them before running.

---

## Workflow

### 1. Build a dataset

**From a georeferenced raster and a shapefile of greenhouses:**

```bash
python Codes/geomap_ToYolo2.py
```

`generate_tiles_with_segmentation_global()` reprojects the polygons onto the raster (detecting the raster's CRS automatically if it is wrong), cuts tiles around each greenhouse at several scales (e.g. `1.0, 0.5, 2.0, 4.0, 7.0`), and writes YOLO segmentation labels (`class x1 y1 … xn yn`) and, if requested, LabelMe JSON files. Set `tif_path`, `shp_path` and `out_dir` in the `__main__` block.

**From LabelMe annotations (image + `.json` pairs):**

```bash
python Codes/test-dataset/dataset_to_yolodataset.py \
       --src <labelme_folder> --dst <output_dataset> --val 0.2 --test 0.0 --seed 42
```

This produces:

```
dataset/
├── data_detect.yaml
├── data_segment.yaml
├── images/{train,val}/
├── labels_detect/{train,val}/    # YOLO bounding boxes
├── labels_segment/{train,val}/   # YOLO polygons
└── masks/{train,val}/            # PNG masks (0 / 255)
```

For U-Net, `dataset_to_unetdataset.py` copies images and turns mask value `1` into `255`. To check a dataset, use `inspect_dataset.py` (counts) and `visualize_dataset.py` (overlays and augmentation preview).

### 2. Train a model

| Model | Script | Notes |
|---|---|---|
| YOLO detect / segment / semantic | `Codes/training/yolo_train.py` or `yolo_kaggle.ipynb` | HSV brightness, rotation, scale (zoom), flips, mosaic, mixup, copy-paste, RandAugment, random erasing; early stopping (patience 30) |
| YOLOv8x-seg | `Codes/train_yolo_seg.py` | `imgsz=384`, `batch=4`, `amp=False`, `lr0=0.001`, no mosaic: settings that avoid NaN losses on small batches |
| U-Net | `Codes/training/unet_train.py` or `unet_kaggle.ipynb` | `smp.Unet` with ResNet-34 encoder (ImageNet weights), Dice loss, AdamW with cosine LR schedule, 256 px tiles |

Resume an interrupted YOLO run with `python Codes/resume_train_yolo_seg.py`.

### 3. Experiment with Segment Anything

The `test-sam/` notebooks compare SAM variants on the greenhouse tiles:

- **MobileSAM / SAM / SAM 2 / SAM 2.1** with point and box prompts, on original and sharpened or super-resolved (FSRCNN x3) inputs.
- **SAM 3** with a text prompt (`"greenhouse"`) and/or box prompts.
- **YOLO → SAM 3** (`yolo_sam3_greenhouses_v3.ipynb`): YOLO predicts boxes, SAM 3 turns each box into a mask, and the notebook computes instance metrics (P / R / F1 / AP50 / AP50-95) and pixel metrics (IoU / Dice / accuracy), with GT vs prediction plots.

### 4. Run detection in QGIS

`QGIS_Script/detect_greenhouses_v3.py` is a QGIS Processing algorithm.

**Install**

1. Install the packages into QGIS's own Python:
   - Windows (OSGeo4W Shell): `pip install -U ultralytics pillow`
   - Linux: `python3 -m pip install --user -U ultralytics pillow`
   - macOS: `/Applications/QGIS.app/Contents/MacOS/bin/python3 -m pip install -U ultralytics pillow`
   - For U-Net state_dicts, also install `segmentation-models-pytorch`.
2. In QGIS: **Processing Toolbox → Scripts → Add Script to Toolbox…** and pick the file. It appears as **Scripts → Remote sensing → Detect greenhouses (YOLO / U-Net / SAM3)**.

**What it does**

- **Input:** any georeferenced raster, or an XYZ (Google / Bing / Esri satellite), WMS or WCS layer, which is first rendered to a temporary GeoTIFF at the chosen pixel size.
- **Models:** YOLO detection / instance segmentation, YOLO semantic segmentation, U-Net (full model or state_dict), or SAM 3 (text prompt and/or boxes from a YOLO detector, in "exemplar boxes + text" or "one mask per box" mode).
- **Area of interest:** optional AOI polygon layer. Tiles outside it are skipped, and results can be clipped to it.
- **Large areas:** tiles are sent to the GPU in small batches, rendered rasters have a megapixel cap, free disk space is checked, and GPU memory is released during long runs.
- **Output:** a vector layer of greenhouse polygons (or boxes) in the chosen CRS (EPSG:3857 by default), with `area_m2`. Use a local UTM zone for accurate areas.

**Checking the georeferencing**

`QGIS_Script/greenhouse_georef_test.py` (**Test greenhouse georeferencing (round-trip IoU)**) uses your ground-truth polygons as a "perfect model": it tiles them with exactly the same grid as the detection tool, writes them as YOLO labels, puts them back on the map with the same code, and reports IoU, centroid shift and Hausdorff distance. Exact-vertex mode should give IoU ≈ 1.0 and a shift of ≈ 0. A steady half-pixel shift in mask mode points to a pixel-corner convention problem.

---

## Results

### YOLOv8 instance segmentation (100 epochs, `imgsz=384`, `batch=4`)

Final-epoch validation metrics from `results/*/results.csv`:

| Model | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| YOLOv8n-seg | 0.965 | 0.839 | 0.903 | 0.781 | 0.814 | 0.684 | 0.730 | 0.434 |
| YOLOv8s-seg | **0.980** | **0.867** | **0.929** | **0.855** | **0.829** | **0.707** | **0.768** | **0.492** |

A 1-epoch YOLO12n detection test (`results/exp-2`) reached box mAP50 = 0.798 and mAP50-95 = 0.504.

### YOLO boxes vs SAM 3 masks (10 test images, 54 greenhouses)

From `results/outputs/`:

| Instance level | YOLO (boxes) | SAM 3 (masks) |
|---|---|---|
| Precision | 0.350 | 0.200 |
| Recall | 0.259 | 0.148 |
| F1 | 0.298 | 0.170 |
| AP50 | 0.140 | 0.083 |
| AP50-95 | 0.109 | 0.065 |
| Mean IoU of matched objects | 0.864 | 0.792 |

| Area level | YOLO (box area) | SAM 3 (mask vs GT mask) |
|---|---|---|
| IoU | 0.576 | 0.345 |
| Dice / F1 | 0.731 | 0.513 |
| Precision | 0.800 | 0.783 |
| Recall | 0.672 | 0.381 |
| Accuracy | 0.858 | 0.813 |

On this small test set, SAM 3 masks prompted by YOLO boxes did worse than the boxes alone, mainly because of low recall. Per-image results are in `per_image_metrics.csv`.

---

## Notes

- Greenhouses are one class (`GH` / `greenhouse`, id 0).
- Files named `* copy.*` and the numbered versions (`v1`, `v2`, …) are kept as a record of the development steps. Use the latest versions: `detect_greenhouses_v3.py`, `dataset_to_yolodataset.py`, `geomap_ToYolo2.py`.
- `Codes/training/yolo_seg_train.py` is currently empty.
