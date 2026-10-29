# AI-Based Greenhouse Detection, Segmentation, and Area Estimation

This repository contains research code for a thesis project developing an AI-based workflow to detect and segment greenhouses in aerial or satellite imagery, then convert the predictions into georeferenced vector polygons for mapping and area analysis.

The codebase includes dataset conversion and inspection utilities, YOLO and U-Net training experiments, Segment Anything (SAM/SAM3) notebooks, and a QGIS Processing algorithm for running models on georeferenced imagery.

## Repository contents

| Path | Purpose |
| --- | --- |
| `Codes/test-dataset/` | Dataset inspection and LabelMe-to-YOLO/U-Net dataset preparation utilities. |
| `Codes/training/` | YOLO and U-Net training scripts and notebooks. |
| `Codes/test-sam/` | Experimental SAM and SAM3 workflows in Jupyter notebooks. |
| `Codes/pipelines/` | Raster tiling and prediction pipeline experiments, including geospatial post-processing. |
| `Codes/geomap_ToYolo.py` and `Codes/geomap_ToYolo2.py` | Utilities for preparing geospatial imagery and labels for model workflows. |
| `QGIS_Script/detect_greenhouses_v3.py` | QGIS Processing algorithm for greenhouse inference and polygon output. |
| `QGIS_Script/greenhouse_georef_test.py` | Georeferencing experiment. |
| `GreenHouse-Dataset_Images+jsonlabels+masks/` | Local source and prepared datasets. |
| `results/` | Local experiment outputs and training results. |
| `weights/` | Local model-weight storage. |
| `QGIS-Project/` | Local QGIS project files. |

The notebooks and older pipeline versions are research experiments; paths, parameters, and supported model formats can vary between them. The QGIS script's installation notes and parameter descriptions are the most complete guide to using its inference workflow.

## Data and model files

The source dataset, prepared datasets, model weights, training outputs, and other large assets are excluded from Git by `.gitignore`. They are not downloaded or installed by this repository. Provide the data and weights locally before running the relevant scripts.

For dataset preparation, the current converter expects a folder of paired LabelMe image/JSON files. It writes YOLO detection and segmentation labels, segmentation masks, train/validation (and optionally test) splits, and dataset YAML files. For example, from the repository root:

```powershell
python "Codes\test-dataset\dataset_to_yolodataset.py" `
  --src "C:\path\to\labelme\images" `
  --dst "C:\path\to\prepared-dataset" `
  --val 0.2 `
  --test 0 `
  --seed 42 `
  --classes GH `
  --mask-mode class
```

Set `--classes` to the label names used in the source annotations. The converter creates `data_detect.yaml` and `data_segment.yaml` under the destination directory. Use the segmentation YAML for segmentation training.

## Python environment

There is no repository-wide dependency lockfile or requirements file. Install dependencies for the specific workflow you plan to use and ensure compatible versions of Python, PyTorch, and CUDA (if using a GPU).

The training and data-preparation scripts use packages including Ultralytics, NumPy, Pillow, Matplotlib, and PyYAML. Geospatial scripts may additionally require Rasterio, GeoPandas, Shapely, and PyProj. Notebook experiments may have further dependencies.

## Run greenhouse inference in QGIS

The supported desktop workflow is the QGIS Processing algorithm:

1. Install Ultralytics and Pillow into the Python environment used by QGIS. QGIS supplies PyQGIS, GDAL, and Shapely.
2. In QGIS, open **Processing Toolbox → Scripts → Add Script to Toolbox…** and select `QGIS_Script/detect_greenhouses_v3.py`.
3. Choose a georeferenced raster, model architecture, model-weight file, and output CRS in the algorithm dialog.
4. Configure tile size, overlap, image size, confidence, and IoU as appropriate for the model. For semantic segmentation and U-Net, use the inference size expected by the trained model; confidence and IoU do not apply to those models.
5. Run the algorithm and save the resulting greenhouse polygons to a vector layer.

The QGIS algorithm supports YOLO detection/instance segmentation, YOLO semantic segmentation, U-Net, and SAM3 (subject to the installed Ultralytics version and model requirements). SAM3 and many training/inference workflows benefit from a CUDA-capable GPU. For XYZ/WMS/WCS imagery, provide a processing extent and pixel size. Large-area jobs can require substantial memory and disk space; use the algorithm's inference batch and rendered-raster size limits.

For additional installation details and all algorithm parameters, see the header documentation in `QGIS_Script/detect_greenhouses_v3.py`.

## Training

Training scripts are in `Codes/training/`. The YOLO helper in `yolo_train.py` accepts a dataset YAML path and model/training parameters through its `train_yolo` function. Before using a script's `__main__` example, update its dataset and output paths to match your local files; examples may be configured for a particular machine or experiment.

Training outputs and checkpoints should be kept outside Git or in ignored locations. Preserve the dataset split, class ordering, model configuration, and training parameters alongside results if you need to reproduce an experiment.

## License and research context

No license or standardized contribution instructions are currently provided in this repository. Refer to the thesis proposal and project documentation in the local workspace for further research context.
