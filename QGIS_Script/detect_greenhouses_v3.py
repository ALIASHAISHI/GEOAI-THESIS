"""
Greenhouse detection pipeline — QGIS Processing Algorithm.

INSTALL
-------
1. Install the extra Python packages into QGIS's own Python.
   - Windows (OSGeo4W): open the "OSGeo4W Shell" from the Start menu, then:
         pip install -U ultralytics pillow
   - Linux (system QGIS): use the same Python that runs QGIS, e.g.
         python3 -m pip install --user -U ultralytics pillow
   - macOS (QGIS.app): use the bundled Python, e.g.
         /Applications/QGIS.app/Contents/MacOS/bin/python3 -m pip install -U ultralytics pillow

   GDAL, PyQGIS and shapely are already shipped with QGIS.

   NOTE: YOLO *semantic* segmentation models (e.g. yolo26n-sem, as trained by
   yolo_train.py) need an Ultralytics version recent enough to expose
   `result.semantic_mask`. Use the same (or newer) Ultralytics version in QGIS
   as the one you trained with — hence the `-U` above.

2. In QGIS: Processing Toolbox → Scripts (the wrench-and-python icon) →
   "Add Script to Toolbox…" → pick this file. The tool will appear under
   "Scripts → Remote sensing → Detect greenhouses (YOLO / U-Net / SAM3)".

MEMORY / STABILITY (large areas)
--------------------------------
  - Tiles are fed to the models in small chunks ("Inference batch size").
    Passing thousands of tile paths in one predict() call makes Ultralytics
    load every tile into RAM and push them through the GPU as one giant
    batch — that is what froze QGIS or rebooted the machine.
  - The rendered raster (XYZ/WMS/WCS, or a clipped GDAL raster) is capped by
    "Max rendered raster size (megapixels)". Zooming out with a small pixel
    size used to request a raster of tens of gigabytes. Raise the pixel size
    or shrink the extent / AOI if you hit the cap.
  - Free disk space is checked before rendering and tiling.
  - GPU memory is released periodically during long runs.

USAGE
-----
Fill in the dialog:
  - Input raster: any georeferenced raster loaded in QGIS. This also works
    with XYZ tile services (Google/Bing/Esri satellite) and WMS/WCS layers.
    For those the layer is rendered to a temporary GeoTIFF first, and you
    MUST also fill "Processing extent" and "Pixel size".
  - Model architecture:
      * YOLO detection / instance segmentation (yolov8*, yolo11*, *-seg).
      * U-Net (semantic segmentation). Either a full model object (from
        `torch.save(model, ...)`) OR a state_dict / checkpoint dict.
        State_dicts are loaded into an smp.Unet configured via the
        "U-Net state_dict only" parameters. Requires
        `pip install segmentation-models-pytorch` for the state_dict path.
      * YOLO semantic segmentation (e.g. yolo26n-sem trained with
        yolo_train.py). The per-pixel class map of every tile is turned
        into polygons. If you pick the first option but the weights turn
        out to be a semantic model, the script switches automatically.
      * SAM3 (Segment Anything 3, via Ultralytics). Put the path to
        sam3.pt in "Model weights". SAM3 is prompted per tile with:
          - a text prompt (default "greenhouse"; comma-separate several
            concepts — each one becomes its own class_id 0, 1, ...), and/or
          - box prompts predicted by an optional "SAM3 box detector" — any
            YOLO detection / instance-seg / OBB model (e.g. your trained
            greenhouse best.pt). The detector runs first and its boxes are
            handed to SAM3, exactly like the GT boxes in SAM3-GEOAI-V1.ipynb.
        Box prompt mode:
          - "Exemplar boxes + text" (SAM3SemanticPredictor, as in the
            notebook): the boxes are examples of the concept, SAM3 returns
            every similar object on the tile — may also find greenhouses
            the detector missed.
          - "One mask per box" (SAM3Predictor, interactive): SAM3 refines
            each detector box into exactly one mask — no new objects, the
            detector decides what is a greenhouse, SAM3 only draws the shape.
        Tiles where the detector finds nothing are skipped (fast) unless
        "Run text-only SAM3 on tiles with no detector boxes" is ticked.
        Requires an Ultralytics version with SAM3 support
        (pip install -U ultralytics) and, realistically, a CUDA GPU.
  - Model weights: path to your trained best.pt (or .pth — for YOLO, .pth
    files are auto-staged as .pt in the work folder before loading).
    For SAM3: path to sam3.pt.
  - Detection mode: polygon masks or bounding boxes. Bounding boxes are
    always axis-aligned rectangles in the raster's pixel space.
  - Tile size / overlap / imgsz / conf / iou: same meaning as in the CLI
    script. For semantic models, set "Inference image size" to the size you
    trained with (640 in yolo_train.py). conf/iou are ignored for semantic
    and U-Net models.
  - Inference batch size: how many tiles go through the GPU at once.
    Lower it (1–2) if you still run out of GPU memory.
  - Max rendered raster size (megapixels): safety cap for the rendered
    raster. 0 = no limit (not recommended).
  - Target class id (U-Net / YOLO-semantic): which class to polygonize.
    -1 = auto: for YOLO-semantic, any class whose name contains
    "greenhouse"; if none, every class except a background class.
  - Processing extent: bounding box to run detection on. Required for
    XYZ/WMS/WCS layers unless an AOI layer is provided. Optional for
    regular GDAL rasters (if given, the raster is clipped to it).
  - AOI layer: a polygon layer whose geometry constrains inference.
    Its bounding box becomes the render extent (for XYZ/WMS), tiles
    outside its shape are skipped, and detections outside it are
    dropped. Tick "Selected features only" in the layer widget to use
    just the currently selected polygons.
  - Clip results to AOI: if checked, output polygons are cut to the AOI
    boundary. If unchecked (default), whole polygons that intersect the
    AOI are kept — safer when a detection sits on the edge.
  - Pixel size: ground sample distance for the rendered raster, in the
    layer's CRS units (meters for projected CRSs, degrees for EPSG:4326).
    Used only when the layer has to be rendered.
  - Output CRS: the CRS the polygons are written in. EPSG:3857 by default.
    Change it to your raster's CRS or a local UTM zone for accurate area_m2.
  - Work folder (optional): where intermediate tiles + model output are
    kept. If empty, a temp folder is used.
  - Detected greenhouses: the output vector layer.
"""

import csv
import gc
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingParameterCrs,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterString,
    QgsProcessingParameterFeatureSink,
    QgsRasterBlockFeedback,
    QgsRasterFileWriter,
    QgsRasterPipe,
    QgsRectangle,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsWkbTypes,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
    QgsSpatialIndex,
)


class GreenhouseDetectionAlgorithm(QgsProcessingAlgorithm):
    """Tile a raster, run a YOLO / U-Net model on each tile, return polygons
    in map CRS."""

    INPUT_RASTER = "INPUT_RASTER"
    WEIGHTS = "WEIGHTS"
    ARCH = "ARCH"
    MODE = "MODE"
    TILE_SIZE = "TILE_SIZE"
    OVERLAP = "OVERLAP"
    IMGSZ = "IMGSZ"
    CONF = "CONF"
    IOU = "IOU"
    BATCH = "BATCH"
    MAX_MEGAPIXELS = "MAX_MEGAPIXELS"
    OUTPUT_CRS = "OUTPUT_CRS"
    WORK_FOLDER = "WORK_FOLDER"
    EXTENT = "EXTENT"
    PIXEL_SIZE = "PIXEL_SIZE"
    AOI_LAYER = "AOI_LAYER"
    AOI_CLIP = "AOI_CLIP"
    MERGE = "MERGE"
    MERGE_TOLERANCE = "MERGE_TOLERANCE"
    UNET_INPUT_SIZE = "UNET_INPUT_SIZE"
    UNET_THRESHOLD = "UNET_THRESHOLD"
    UNET_CLASS_ID = "UNET_CLASS_ID"
    UNET_ENCODER = "UNET_ENCODER"
    UNET_IN_CHANNELS = "UNET_IN_CHANNELS"
    UNET_NUM_CLASSES = "UNET_NUM_CLASSES"
    SAM3_TEXT = "SAM3_TEXT"
    SAM3_CONF = "SAM3_CONF"
    SAM3_DETECTOR = "SAM3_DETECTOR"
    SAM3_PROMPT_MODE = "SAM3_PROMPT_MODE"
    SAM3_TEXT_FALLBACK = "SAM3_TEXT_FALLBACK"
    OUTPUT = "OUTPUT"

    # SAM3 box-prompt modes
    SAM3_PROMPT_EXEMPLAR = 0
    SAM3_PROMPT_PER_BOX = 1

    # Mode enum values
    MODE_POLYGONS = 0
    MODE_BBOXES = 1

    # Architecture enum values
    ARCH_YOLO = 0
    ARCH_UNET = 1
    ARCH_YOLO_SEM = 2
    ARCH_SAM3 = 3

    # Release GPU memory every N chunks / tiles
    GPU_FREE_EVERY = 25
    # Warn above this many tiles
    TILE_WARN = 20000

    # ---- Algorithm metadata ---------------------------------------------------

    def tr(self, string):
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return GreenhouseDetectionAlgorithm()

    def name(self):
        return "detect_greenhouses"

    def displayName(self):
        return self.tr("Detect greenhouses (YOLO / U-Net / SAM3)")

    def group(self):
        return self.tr("Remote sensing")

    def groupId(self):
        return "remote_sensing"

    def shortHelpString(self):
        return self.tr(
            "Tile a georeferenced raster, run a YOLO (detection, instance "
            "segmentation or semantic segmentation) or U-Net model over each "
            "tile, and return the detected greenhouses as a polygon layer in "
            "map coordinates.\n\n"
            "YOLO semantic models (e.g. yolo26n-sem trained with "
            "yolo_train.py) produce a per-pixel class map that is converted "
            "to polygons. Set 'Inference image size' to the training size.\n\n"
            "SAM3: set 'Model weights' to sam3.pt, give a text prompt and/or "
            "pick a YOLO 'SAM3 box detector'. The detector predicts boxes on "
            "each tile and SAM3 uses them as prompts — either as exemplars "
            "(finds all similar objects) or one mask per box.\n\n"
            "Large areas: tiles are processed in small batches, and the "
            "rendered raster is capped by 'Max rendered raster size'. If you "
            "hit the cap, increase the pixel size or reduce the extent.\n\n"
            "Requires the ultralytics and pillow Python packages to be "
            "installed in QGIS's Python environment. See the header of the "
            "script file for install instructions."
        )

    # ---- Parameters -----------------------------------------------------------

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER, self.tr("Input raster (georeferenced)")
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.ARCH,
                self.tr("Model architecture"),
                options=[
                    self.tr("YOLO detection / instance segmentation (yolov8, yolo11, *-seg)"),
                    self.tr("U-Net (semantic segmentation)"),
                    self.tr("YOLO semantic segmentation (e.g. yolo26-sem from yolo_train.py)"),
                    self.tr("SAM3 (text and/or detector box prompts)"),
                ],
                defaultValue=self.ARCH_YOLO,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.WEIGHTS,
                self.tr("Model weights (.pt or .pth; sam3.pt for SAM3)"),
                fileFilter="PyTorch weights (*.pt *.pth);;All files (*.*)",
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODE,
                self.tr("Detection mode"),
                options=[
                    self.tr("Segmentation (polygon masks)"),
                    self.tr("Bounding boxes (rectangles)"),
                ],
                defaultValue=self.MODE_POLYGONS,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TILE_SIZE,
                self.tr("Tile size (pixels)"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=640, minValue=64,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OVERLAP,
                self.tr("Tile overlap (pixels)"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=64, minValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.IMGSZ,
                self.tr("Inference image size (YOLO / SAM3 box detector; "
                        "use training size for semantic models)"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1024, minValue=64,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CONF,
                self.tr("Confidence threshold (YOLO detection/instance and SAM3 box detector)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.35, minValue=0.0, maxValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.IOU,
                self.tr("NMS IoU threshold (YOLO detection/instance and SAM3 box detector)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.45, minValue=0.0, maxValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BATCH,
                self.tr("Inference batch size (tiles per GPU pass; lower = less memory)"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=4, minValue=1, maxValue=64,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_MEGAPIXELS,
                self.tr("Max rendered raster size (megapixels, 0 = no limit)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=400.0, minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterCrs(
                self.OUTPUT_CRS,
                self.tr("Output CRS"),
                defaultValue="EPSG:3857",
            )
        )
        self.addParameter(
            QgsProcessingParameterExtent(
                self.EXTENT,
                self.tr("Processing extent (required for XYZ/WMS/WCS layers)"),
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PIXEL_SIZE,
                self.tr("Pixel size for rendered raster (layer CRS units, e.g. metres)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.5, minValue=0.0,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.UNET_CLASS_ID,
                self.tr("[U-Net / YOLO-semantic] Target class id (-1 = auto / any foreground)"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=-1, minValue=-1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.UNET_INPUT_SIZE,
                self.tr("[U-Net only] Model input size (px, 0 = use tile size)"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=0, minValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.UNET_THRESHOLD,
                self.tr("[U-Net only] Probability threshold (binary output)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.5, minValue=0.0, maxValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.UNET_ENCODER,
                self.tr("[U-Net state_dict only] smp encoder name (e.g. resnet34, "
                        "resnet50, efficientnet-b0, mobilenet_v2)"),
                defaultValue="resnet34",
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.UNET_IN_CHANNELS,
                self.tr("[U-Net state_dict only] Input channels (3 for RGB, 4 for RGBI, ...)"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3, minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.UNET_NUM_CLASSES,
                self.tr("[U-Net state_dict only] Number of output classes (1 = binary)"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1, minValue=1,
            )
        )
        # ---- SAM3 ----
        self.addParameter(
            QgsProcessingParameterString(
                self.SAM3_TEXT,
                self.tr("[SAM3] Text prompt (comma-separated concepts; empty = "
                        "box prompts only)"),
                defaultValue="greenhouse",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SAM3_CONF,
                self.tr("[SAM3] Mask confidence threshold"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.5, minValue=0.0, maxValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.SAM3_DETECTOR,
                self.tr("[SAM3] Box detector weights — YOLO model that predicts "
                        "boxes to prompt SAM3 (optional)"),
                fileFilter="PyTorch weights (*.pt *.pth);;All files (*.*)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.SAM3_PROMPT_MODE,
                self.tr("[SAM3] How detector boxes are used"),
                options=[
                    self.tr("Exemplar boxes + text — find all similar objects "
                            "(SAM3 semantic, as in the notebook)"),
                    self.tr("One mask per detector box (SAM3 interactive)"),
                ],
                defaultValue=self.SAM3_PROMPT_EXEMPLAR,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SAM3_TEXT_FALLBACK,
                self.tr("[SAM3] Run text-only SAM3 on tiles where the detector "
                        "finds no boxes (slower, higher recall)"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.AOI_LAYER,
                self.tr("AOI layer (polygons — use 'Selected features only' if desired)"),
                types=[QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.AOI_CLIP,
                self.tr("Clip results to AOI boundary"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.MERGE,
                self.tr("Merge intersecting polygons (per class)"),
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MERGE_TOLERANCE,
                self.tr("Merge tolerance — also merge polygons closer than this "
                        "(output CRS units, 0 = only touching/overlapping)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0, minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.WORK_FOLDER,
                self.tr("Work folder for tiles + model output (optional)"),
                behavior=QgsProcessingParameterFile.Folder,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr("Detected greenhouses"),
                type=QgsProcessing.TypeVectorPolygon,
            )
        )

    # ---- Helpers --------------------------------------------------------------

    @staticmethod
    def _pick_device():
        """GPU 0 if torch sees CUDA, else CPU (Ultralytics device syntax)."""
        try:
            import torch
            return 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @staticmethod
    def _free_gpu():
        """Release Python garbage and cached CUDA memory."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def _clear_labels_dir(labels_dir):
        """Create labels_dir, removing label files from an earlier run."""
        if labels_dir.exists():
            for old in labels_dir.glob("*.txt"):
                old.unlink()
        labels_dir.mkdir(parents=True, exist_ok=True)

    def _predict_chunked(self, model, tile_paths, batch, feedback, **kwargs):
        """Run model.predict over tile_paths a few tiles at a time.

        Yields (index, tile_path, result). Ultralytics loads every image of a
        list source into RAM up front and treats the whole list as ONE batch,
        so passing thousands of paths at once exhausts RAM and GPU memory.
        Feeding small chunks with stream=True keeps both bounded.
        """
        n = len(tile_paths)
        for n_chunk, start in enumerate(range(0, n, batch)):
            if feedback.isCanceled():
                return
            chunk = tile_paths[start:start + batch]
            results = model.predict(
                source=[str(p) for p in chunk],
                stream=True,
                save=False,
                verbose=False,
                **kwargs,
            )
            for j, r in enumerate(results):
                yield start + j, chunk[j], r
                del r
            del results
            if (n_chunk + 1) % self.GPU_FREE_EVERY == 0:
                self._free_gpu()

    def _load_yolo_model(self, weights_path, work_dir, feedback):
        """Import ultralytics and load the YOLO weights (any task)."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise QgsProcessingException(
                "The 'ultralytics' package is not installed in QGIS's "
                "Python. Install it with:\n    pip install -U ultralytics"
            )

        # Ultralytics' YOLO() loader keys off the '.pt' extension. If the
        # user picked a .pth (or anything else), stage a copy with a .pt
        # extension in the work folder so it loads cleanly.
        if weights_path.suffix.lower() != ".pt":
            staged_weights = work_dir / (weights_path.stem + ".pt")
            feedback.pushInfo(
                f"Staging '{weights_path.name}' as '{staged_weights.name}' "
                "(Ultralytics expects a .pt extension)."
            )
            shutil.copy2(str(weights_path), str(staged_weights))
            load_path = staged_weights
        else:
            load_path = weights_path

        feedback.pushInfo(f"Loading YOLO model: {load_path}")
        try:
            model = YOLO(str(load_path))
        except Exception as e:
            raise QgsProcessingException(
                f"Ultralytics could not load {load_path.name}: {e}\n"
                "If this is a semantic model (e.g. yolo26-sem), upgrade "
                "Ultralytics in QGIS's Python to the version you trained "
                "with:  pip install -U ultralytics"
            )
        return model

    @staticmethod
    def _yolo_result_to_lines(r):
        """Turn one Ultralytics detect / segment / OBB result into YOLO label
        lines (normalized to the tile): polygons for masks and OBB, 'c cx cy
        w h' for plain boxes."""
        lines = []
        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)
        obb = getattr(r, "obb", None)

        if masks is not None and len(masks) and boxes is not None:
            cls = boxes.cls.cpu().numpy().astype(int).reshape(-1)
            for c, poly in zip(cls, masks.xyn):
                poly = np.asarray(poly, dtype=float).reshape(-1, 2)
                if len(poly) < 3:
                    continue
                coords = " ".join(f"{v:.6f}" for v in poly.flatten())
                lines.append(f"{int(c)} {coords}")
        elif obb is not None and len(obb):
            cls = obb.cls.cpu().numpy().astype(int).reshape(-1)
            pts = obb.xyxyxyxyn.cpu().numpy().reshape(-1, 4, 2)
            for c, poly in zip(cls, pts):
                coords = " ".join(f"{v:.6f}" for v in poly.flatten())
                lines.append(f"{int(c)} {coords}")
        elif boxes is not None and len(boxes):
            cls = boxes.cls.cpu().numpy().astype(int).reshape(-1)
            xywhn = boxes.xywhn.cpu().numpy().reshape(-1, 4)
            for c, (cx, cy, bw, bh) in zip(cls, xywhn):
                lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return lines

    def _run_yolo_inference(self, model, tile_paths, preds_project, imgsz,
                            device, conf, iou, batch, feedback):
        """YOLO detection / instance segmentation / OBB, chunked and
        streamed; writes YOLO-format label files."""
        labels_dir = preds_project / "yolo_seg" / "labels"
        self._clear_labels_dir(labels_dir)

        feedback.pushInfo(
            f"Running YOLO inference on {len(tile_paths)} tiles "
            f"(imgsz={imgsz}, batch={batch}, device={device})..."
        )
        n_total = 0
        n_tiles = len(tile_paths)
        for i, tp, r in self._predict_chunked(
            model, tile_paths, batch, feedback,
            imgsz=imgsz, device=device, conf=conf, iou=iou,
        ):
            lines = self._yolo_result_to_lines(r)
            if lines:
                (labels_dir / (tp.stem + ".txt")).write_text("\n".join(lines))
                n_total += len(lines)
            if (i + 1) % 20 == 0:
                feedback.setProgress(40 + int((i + 1) / n_tiles * 40))

        feedback.pushInfo(f"YOLO produced {n_total} detections across tiles.")
        return labels_dir

    @staticmethod
    def _semantic_target_classes(names, class_id, feedback):
        """Decide which class ids of a semantic class map count as greenhouse.

        class_id >= 0 → exactly that class.
        class_id == -1 → classes whose name contains 'greenhouse'; if none,
        every class except the background class (a class named
        background/bg/void/..., or class 0 if no such name exists).
        """
        if isinstance(names, (list, tuple)):
            names = dict(enumerate(names))
        names = {int(k): str(v) for k, v in (names or {}).items()}
        if names:
            feedback.pushInfo(f"Model classes: {names}")

        if class_id >= 0:
            if names and class_id not in names:
                raise QgsProcessingException(
                    f"Target class id {class_id} is not in the model's "
                    f"classes: {names}"
                )
            return [class_id]

        def norm(s):
            return s.lower().replace("_", "").replace("-", "").replace(" ", "")

        gh = [k for k, v in names.items() if "greenhouse" in norm(v)]
        if gh:
            return sorted(gh)

        if len(names) <= 1:
            return sorted(names.keys()) or [1]

        bg_words = {"background", "bg", "background_", "_background_",
                    "void", "unlabeled", "unlabelled", "none", "other"}
        bg = [k for k, v in names.items() if v.lower().strip() in bg_words]
        if not bg:
            bg = [0]
        return sorted(k for k in names if k not in bg)

    def _run_yolo_semantic_inference(self, model, tile_paths, preds_project,
                                     imgsz, device, class_id, batch, feedback):
        """Run a YOLO semantic-segmentation model (e.g. yolo26n-sem) over every
        tile, polygonize the per-pixel class map, and write YOLO-format
        polygon label files so the rest of the pipeline is unchanged."""
        from PIL import Image

        targets = self._semantic_target_classes(
            getattr(model, "names", {}), class_id, feedback
        )
        name_map = getattr(model, "names", {}) or {}
        if isinstance(name_map, (list, tuple)):
            name_map = dict(enumerate(name_map))
        feedback.pushInfo(
            "Polygonizing class(es): "
            + ", ".join(f"{t} ({name_map.get(t, '?')})" for t in targets)
        )

        labels_dir = preds_project / "yolo_semantic" / "labels"
        self._clear_labels_dir(labels_dir)

        feedback.pushInfo(
            f"Running YOLO semantic inference on {len(tile_paths)} tiles "
            f"(imgsz={imgsz}, batch={batch}, device={device})..."
        )

        n_total = 0
        n_tiles = len(tile_paths)
        for i, tile_path, result in self._predict_chunked(
            model, tile_paths, batch, feedback, imgsz=imgsz, device=device,
        ):
            sem = getattr(result, "semantic_mask", None)
            if sem is None:
                raise QgsProcessingException(
                    "The model returned no semantic mask. Either these are "
                    "not semantic-segmentation weights (pick 'YOLO detection "
                    "/ instance segmentation' instead), or the Ultralytics "
                    "version in QGIS is too old: pip install -U ultralytics"
                )

            data = getattr(sem, "data", sem)
            if hasattr(data, "cpu"):
                data = data.cpu().numpy()
            data = np.asarray(data).squeeze()
            if data.ndim == 3:
                # Per-class scores / one-hot [C, H, W] → class map
                data = data.argmax(axis=0)
            if data.ndim != 2:
                raise QgsProcessingException(
                    f"Unexpected semantic mask shape {data.shape} on "
                    f"{tile_path.name}; expected (H, W)."
                )
            class_map = data.astype(np.int32)
            del data

            # Tile size in pixels (the class map may be at inference size)
            orig_shape = getattr(result, "orig_shape", None)
            if orig_shape is not None and len(orig_shape) >= 2:
                tile_h, tile_w = int(orig_shape[0]), int(orig_shape[1])
            else:
                with Image.open(tile_path) as im:
                    tile_w, tile_h = im.size

            if class_map.shape != (tile_h, tile_w):
                cm_img = Image.fromarray(class_map.astype(np.int32), mode="I")
                cm_img = cm_img.resize((tile_w, tile_h), Image.NEAREST)
                class_map = np.array(cm_img, dtype=np.int32)

            lines = []
            for t in targets:
                mask = (class_map == t).astype(np.uint8)
                for poly in self._mask_to_polygons(mask):
                    if len(poly) < 3:
                        continue
                    poly_n = poly.astype(float).copy()
                    poly_n[:, 0] /= float(tile_w)
                    poly_n[:, 1] /= float(tile_h)
                    coords_str = " ".join(f"{v:.6f}" for v in poly_n.flatten())
                    lines.append(f"{t} {coords_str}")

            if lines:
                (labels_dir / (tile_path.stem + ".txt")).write_text(
                    "\n".join(lines)
                )
                n_total += len(lines)

            if (i + 1) % 20 == 0:
                feedback.setProgress(40 + int((i + 1) / n_tiles * 40))

        feedback.pushInfo(
            f"YOLO semantic model produced {n_total} mask polygons across tiles."
        )
        return labels_dir

    # ---- SAM3 -----------------------------------------------------------------

    @staticmethod
    def _make_sam3_predictor(kind, weights_path, conf, device, feedback):
        """Build an Ultralytics SAM3 predictor.

        kind = "semantic" → SAM3SemanticPredictor (text / exemplar boxes,
               returns all matching instances — as in SAM3-GEOAI-V1.ipynb)
        kind = "interactive" → SAM3Predictor (one mask per box prompt)

        The model itself is loaded lazily by Ultralytics on the first
        set_image() call.
        """
        try:
            if kind == "semantic":
                from ultralytics.models.sam import SAM3SemanticPredictor as Pred
            else:
                from ultralytics.models.sam import SAM3Predictor as Pred
        except ImportError as e:
            raise QgsProcessingException(
                f"This Ultralytics version has no SAM3 support ({e}). "
                "Upgrade it in QGIS's Python:  pip install -U ultralytics"
            )

        base = {
            "conf": conf,
            "task": "segment",
            "mode": "predict",
            "model": str(weights_path),
            "save": False,
            "verbose": False,
            "device": device,
        }
        # FP16 on GPU. The notebook uses "quantize": 16; older Ultralytics
        # versions call it "half". Try both, then plain FP32.
        extras = [{"quantize": 16}, {"half": True}, {}] if device != "cpu" \
            else [{}]
        last_err = None
        for extra in extras:
            try:
                pred = Pred(overrides={**base, **extra})
                feedback.pushInfo(
                    f"SAM3 {kind} predictor ready "
                    f"(conf={conf}, device={device}"
                    + (f", {extra}" if extra else "") + ")."
                )
                return pred
            except Exception as e:  # unknown cfg key → retry without it
                last_err = e
        raise QgsProcessingException(
            f"Could not create the SAM3 {kind} predictor: {last_err}"
        )

    def _run_box_detector(self, detector, tile_paths, imgsz, conf, iou,
                          device, batch, feedback):
        """Run a YOLO detection / seg / OBB model over the tiles and return
        {tile_path: (boxes_xyxy float32 (N,4), class_ids int (N,))} in tile
        pixel coordinates — the box prompts for SAM3."""
        feedback.pushInfo(
            f"Running box detector on {len(tile_paths)} tiles "
            f"(imgsz={imgsz}, conf={conf}, iou={iou}, batch={batch}, "
            f"device={device})..."
        )
        out = {}
        n_boxes = 0
        for i, tp, r in self._predict_chunked(
            detector, tile_paths, batch, feedback,
            imgsz=imgsz, conf=conf, iou=iou, device=device,
        ):
            b = getattr(r, "boxes", None)
            if b is None or len(b) == 0:
                b = getattr(r, "obb", None)       # OBB models
            if b is None or len(b) == 0:
                out[tp] = (np.zeros((0, 4), np.float32), np.zeros(0, int))
                continue
            xyxy = b.xyxy.cpu().numpy().astype(np.float32).reshape(-1, 4)
            cls = b.cls.cpu().numpy().astype(int).reshape(-1)
            out[tp] = (xyxy, cls)
            n_boxes += len(xyxy)
            if (i + 1) % 20 == 0:
                feedback.setProgress(40 + int((i + 1) / len(tile_paths) * 15))
        n_hit = sum(1 for v in out.values() if len(v[0]))
        feedback.pushInfo(
            f"Detector found {n_boxes} boxes on {n_hit}/{len(tile_paths)} tiles."
        )
        return out

    @staticmethod
    def _resize_mask(mask, w, h):
        """Nearest-neighbour resize of a (h0, w0) binary mask to (h, w)."""
        if mask.shape == (h, w):
            return mask.astype(np.uint8)
        try:
            import cv2
            return cv2.resize(mask.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST)
        except ImportError:
            from PIL import Image
            m = Image.fromarray(mask.astype(np.uint8) * 255)
            m = m.resize((w, h), Image.NEAREST)
            return (np.array(m) > 127).astype(np.uint8)

    def _run_sam3_inference(self, weights_path, tile_paths, preds_project,
                            text_prompts, sam3_conf, detector_path,
                            prompt_mode, text_fallback, imgsz, conf, iou,
                            batch, work_dir, feedback):
        """Segment every tile with SAM3 and write YOLO-format polygon label
        files (one line per instance mask) so the rest of the pipeline is
        unchanged.

        Prompts per tile:
          - text_prompts (list of str, may be empty)
          - boxes from the optional YOLO detector (detector_path)
        """
        from PIL import Image

        device = self._pick_device()
        if device == "cpu":
            try:
                import sys
                import torch
                cuda_build = torch.version.cuda
                diag = (f"torch {torch.__version__}, CUDA build: "
                        f"{cuda_build or 'NONE (CPU-only wheel)'}, "
                        f"python: {sys.executable}")
                if not cuda_build:
                    hint = ("QGIS's Python has the CPU-only PyTorch. From the "
                            "OSGeo4W Shell run:\n"
                            "    pip uninstall -y torch torchvision\n"
                            "    pip install torch torchvision --index-url "
                            "https://download.pytorch.org/whl/cu128\n"
                            "then restart QGIS.")
                else:
                    hint = ("PyTorch has CUDA support but sees no GPU — check "
                            "the NVIDIA driver (run nvidia-smi) and that it "
                            f"supports CUDA {cuda_build}.")
            except ImportError:
                diag, hint = "torch not importable", "pip install torch"
            feedback.pushWarning(
                "No CUDA GPU found — SAM3 on CPU is very slow "
                f"(minutes per tile). [{diag}]\n{hint}"
            )

        has_detector = detector_path is not None
        if not text_prompts and not has_detector:
            raise QgsProcessingException(
                "SAM3 needs a prompt: give a text prompt and/or a box "
                "detector model."
            )

        # 1) Box prompts from the detector -------------------------------
        boxes_by_tile = {}
        if has_detector:
            feedback.pushInfo(f"Loading SAM3 box detector (YOLO): {detector_path.name}")
            detector = self._load_yolo_model(detector_path, work_dir, feedback)
            det_task = str(getattr(detector, "task", "") or "").lower()
            feedback.pushInfo(f"Box detector task: '{det_task or 'unknown'}'")
            if "sem" in det_task:
                raise QgsProcessingException(
                    "The SAM3 box detector must output boxes (YOLO detect / "
                    "segment / obb). A semantic model gives no boxes."
                )
            boxes_by_tile = self._run_box_detector(
                detector, tile_paths, imgsz, conf, iou, device, batch, feedback
            )
            del detector
            self._free_gpu()
        feedback.setProgress(55)

        if prompt_mode == self.SAM3_PROMPT_PER_BOX and not has_detector:
            feedback.pushWarning(
                "'One mask per detector box' needs a box detector — falling "
                "back to text-only SAM3."
            )
        if prompt_mode == self.SAM3_PROMPT_PER_BOX and has_detector \
                and text_prompts:
            feedback.pushInfo(
                "Per-box mode: the text prompt is only used for the text-only "
                "fallback on tiles without detector boxes."
            )

        # 2) SAM3 --------------------------------------------------------
        labels_dir = preds_project / "sam3" / "labels"
        self._clear_labels_dir(labels_dir)

        predictors = {}

        def get_pred(kind):
            if kind not in predictors:
                predictors[kind] = self._make_sam3_predictor(
                    kind, weights_path, sam3_conf, device, feedback
                )
            return predictors[kind]

        feedback.pushInfo(
            f"Running SAM3 on {len(tile_paths)} tiles "
            f"(text={text_prompts or '—'}, "
            f"boxes={'detector' if has_detector else '—'})..."
        )

        n_total = n_skipped = n_failed = n_run = 0
        n_tiles = len(tile_paths)
        first_call = True
        for i, tp in enumerate(tile_paths):
            if feedback.isCanceled():
                break

            boxes = det_cls = None
            if has_detector:
                boxes, det_cls = boxes_by_tile.get(
                    tp, (np.zeros((0, 4), np.float32), np.zeros(0, int))
                )
                if len(boxes) == 0:
                    if not (text_fallback and text_prompts):
                        n_skipped += 1
                        continue
                    boxes = det_cls = None   # text-only on this tile

            per_box = (prompt_mode == self.SAM3_PROMPT_PER_BOX
                       and boxes is not None)
            kind = "interactive" if per_box else "semantic"

            try:
                pred = get_pred(kind)
                pred.set_image(str(tp))
                if per_box:
                    results = pred(bboxes=boxes.tolist())
                else:
                    kwargs = {}
                    if text_prompts:
                        kwargs["text"] = text_prompts
                    if boxes is not None:
                        kwargs["bboxes"] = boxes
                        kwargs["labels"] = np.ones(len(boxes), dtype=np.int32)
                    results = pred(**kwargs)
            except QgsProcessingException:
                raise
            except Exception as e:
                if first_call:
                    raise QgsProcessingException(
                        f"SAM3 failed on the first tile ({tp.name}): {e}\n"
                        "Check that 'Model weights' points to sam3.pt and that "
                        "Ultralytics is up to date (pip install -U ultralytics)."
                    )
                n_failed += 1
                feedback.pushWarning(f"SAM3 failed on {tp.name}: {e}")
                self._free_gpu()   # a failure is often an OOM — recover
                continue
            first_call = False
            n_run += 1

            res = results[0] if isinstance(results, (list, tuple)) else results
            masks = getattr(res, "masks", None)
            if masks is None or len(masks) == 0:
                del results, res
                continue

            data = masks.data
            if hasattr(data, "cpu"):
                data = data.cpu().numpy()
            data = np.asarray(data)
            if data.ndim == 2:
                data = data[None]

            orig_shape = getattr(res, "orig_shape", None)
            if orig_shape is not None and len(orig_shape) >= 2:
                tile_h, tile_w = int(orig_shape[0]), int(orig_shape[1])
            else:
                with Image.open(tp) as im:
                    tile_w, tile_h = im.size

            # Class id per mask: detector class (per-box mode) or the index
            # of the matching text prompt (semantic mode).
            mask_cls = np.zeros(len(data), dtype=int)
            if per_box and det_cls is not None and len(det_cls) == len(data):
                mask_cls = det_cls
            else:
                rb = getattr(res, "boxes", None)
                if rb is not None and len(rb) == len(data) \
                        and getattr(rb, "cls", None) is not None:
                    mask_cls = rb.cls.cpu().numpy().astype(int).reshape(-1)
                    if text_prompts:
                        mask_cls = np.clip(mask_cls, 0, len(text_prompts) - 1)

            lines = []
            for m, c in zip(data, mask_cls):
                m = self._resize_mask(m > 0.5, tile_w, tile_h)
                for poly in self._mask_to_polygons(m):
                    if len(poly) < 3:
                        continue
                    poly_n = poly.astype(float).copy()
                    poly_n[:, 0] /= float(tile_w)
                    poly_n[:, 1] /= float(tile_h)
                    coords = " ".join(f"{v:.6f}" for v in poly_n.flatten())
                    lines.append(f"{int(c)} {coords}")

            del results, res, masks, data

            if lines:
                (labels_dir / (tp.stem + ".txt")).write_text("\n".join(lines))
                n_total += len(lines)

            if n_run % self.GPU_FREE_EVERY == 0:
                self._free_gpu()
            if (i + 1) % 5 == 0:
                feedback.setProgress(55 + int((i + 1) / n_tiles * 25))

        predictors.clear()
        self._free_gpu()

        if has_detector:
            feedback.pushInfo(
                f"{n_skipped} tiles skipped (no detector boxes"
                + ("" if text_fallback else
                   "; tick the text-only fallback to run them anyway") + ")."
            )
        if n_failed:
            feedback.pushWarning(f"SAM3 failed on {n_failed} tiles.")
        feedback.pushInfo(f"SAM3 produced {n_total} mask polygons across tiles.")
        return labels_dir

    def _run_unet_inference(self, weights_path, tile_paths, preds_project,
                            unet_input_size, unet_threshold, unet_class_id,
                            unet_encoder, unet_in_channels, unet_num_classes,
                            feedback):
        """Run a U-Net semantic segmentation model over every tile and write
        YOLO-format .txt label files (one polygon per line, normalized coords)
        so the rest of the pipeline can consume them unchanged.

        Supports two save formats:
          - Full model object saved with `torch.save(model, ...)`
          - state_dict (with or without a wrapping dict) — rebuilt into an
            `smp.Unet` using the encoder/channels/classes parameters.
        """
        try:
            import torch
        except ImportError:
            raise QgsProcessingException(
                "PyTorch is not installed in QGIS's Python. Install it with:\n"
                "    pip install torch"
            )
        from PIL import Image

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feedback.pushInfo(f"U-Net device: {device}")

        # Load whatever's in the .pth
        try:
            try:
                obj = torch.load(
                    str(weights_path), map_location=device, weights_only=False
                )
            except TypeError:
                obj = torch.load(str(weights_path), map_location=device)
        except Exception as e:
            raise QgsProcessingException(
                f"torch.load failed on {weights_path.name}: {e}"
            )

        model = self._resolve_unet_model(
            obj, torch, unet_encoder, unet_in_channels, unet_num_classes, feedback
        )
        model = model.to(device)
        model.eval()

        # Set up a labels dir mimicking YOLO's folder layout
        labels_dir = preds_project / "unet_seg" / "labels"
        self._clear_labels_dir(labels_dir)

        feedback.pushInfo(
            f"Running U-Net inference on {len(tile_paths)} tiles "
            f"(threshold={unet_threshold}, target_class={unet_class_id})..."
        )
        n_polys_total = 0
        for i, tp in enumerate(tile_paths):
            if feedback.isCanceled():
                break

            with Image.open(tp) as im:
                img = im.convert("RGB")
            tile_w, tile_h = img.size

            # Optionally resize input to model's expected size
            if unet_input_size > 0 and (tile_w, tile_h) != (unet_input_size,
                                                             unet_input_size):
                img_r = img.resize(
                    (unet_input_size, unet_input_size), Image.BILINEAR
                )
            else:
                img_r = img

            arr = np.array(img_r, dtype=np.float32) / 255.0  # H, W, C
            tensor = (
                torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            )

            try:
                with torch.no_grad():
                    out = model(tensor)
            except Exception as e:
                raise QgsProcessingException(
                    f"U-Net forward pass failed on tile {tp.name}: {e}\n"
                    "Try setting '[U-Net only] Model input size' to the exact "
                    "size the model was trained on (e.g. 256, 384, 512)."
                )

            # Normalize output to a 3D tensor [C, H, W]
            if isinstance(out, (list, tuple)):
                out = out[0]
            if isinstance(out, dict):
                for k in ("out", "logits", "mask", "pred"):
                    if k in out:
                        out = out[k]
                        break
                else:
                    out = next(iter(out.values()))

            if out.dim() == 4 and out.shape[0] == 1:
                out = out.squeeze(0)
            elif out.dim() != 3:
                raise QgsProcessingException(
                    f"Unexpected U-Net output shape: {tuple(out.shape)}. "
                    "Expected [1, C, H, W] or [C, H, W]."
                )

            num_classes = out.shape[0]

            if num_classes == 1:
                # Binary head with sigmoid
                prob = torch.sigmoid(out[0])
                mask = (prob > unet_threshold).cpu().numpy().astype(np.uint8)
                target_class = 0
            else:
                # Multi-class head with softmax → argmax
                argmax = out.argmax(dim=0).cpu().numpy().astype(np.int32)
                if unet_class_id < 0:
                    # "Any foreground": treat class 0 as background
                    mask = (argmax > 0).astype(np.uint8)
                    target_class = 0
                else:
                    mask = (argmax == unet_class_id).astype(np.uint8)
                    target_class = unet_class_id

            del out, tensor

            # Resize mask back to tile pixel space if we resized the input
            if mask.shape != (tile_h, tile_w):
                m_img = Image.fromarray(mask * 255)
                m_img = m_img.resize((tile_w, tile_h), Image.NEAREST)
                mask = (np.array(m_img) > 127).astype(np.uint8)

            polys = self._mask_to_polygons(mask)
            if polys:
                lines = []
                for poly in polys:
                    if len(poly) < 3:
                        continue
                    poly_n = poly.astype(float).copy()
                    poly_n[:, 0] /= float(tile_w)
                    poly_n[:, 1] /= float(tile_h)
                    coords_str = " ".join(f"{v:.6f}" for v in poly_n.flatten())
                    lines.append(f"{target_class} {coords_str}")
                if lines:
                    (labels_dir / (tp.stem + ".txt")).write_text(
                        "\n".join(lines)
                    )
                    n_polys_total += len(lines)

            if (i + 1) % (self.GPU_FREE_EVERY * 4) == 0:
                self._free_gpu()
            if (i + 1) % 20 == 0:
                feedback.setProgress(40 + int((i + 1) / len(tile_paths) * 40))

        del model
        self._free_gpu()
        feedback.pushInfo(
            f"U-Net produced {n_polys_total} mask polygons across tiles."
        )
        return labels_dir

    @classmethod
    def _resolve_unet_model(cls, obj, torch, encoder_name, in_channels,
                             num_classes, feedback):
        """Given whatever torch.load returned, produce a callable model.

        - Full model object → returned directly.
        - Dict with a wrapped model under 'model' / 'net' → returned.
        - Dict with a 'state_dict' / 'model_state_dict' / bare tensors →
          built into an smp.Unet from the user-supplied encoder/channels/
          classes and loaded via load_state_dict.
        """
        # Case 1: already a callable module
        if hasattr(obj, "eval") and callable(obj):
            return obj

        if not isinstance(obj, dict):
            raise QgsProcessingException(
                f"Unexpected object loaded from .pth: {type(obj).__name__}"
            )

        # Case 2: wrapped full model under a common key
        for key in ("model", "net", "network", "generator"):
            if key in obj and hasattr(obj[key], "eval") and callable(obj[key]):
                feedback.pushInfo(f"Using full model object under key '{key}'.")
                return obj[key]

        # Case 3: dict containing a state_dict (or is itself one)
        state_dict = None
        for key in ("state_dict", "model_state_dict", "net_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                state_dict = obj[key]
                feedback.pushInfo(f"Found state_dict under key '{key}'.")
                break

        if state_dict is None and obj and all(
            isinstance(v, torch.Tensor) for v in obj.values()
        ):
            state_dict = obj
            feedback.pushInfo("Treating .pth as a bare state_dict.")

        if state_dict is None:
            raise QgsProcessingException(
                f"The .pth is a dict with keys {list(obj.keys())[:10]} but "
                "no model or state_dict was found."
            )

        return cls._build_unet_from_state_dict(
            state_dict, torch, encoder_name, in_channels, num_classes, feedback
        )

    @staticmethod
    def _build_unet_from_state_dict(state_dict, torch, encoder_name,
                                     in_channels, num_classes, feedback):
        """Instantiate an smp.Unet with the given hyperparameters and load
        the state_dict into it. Handles the DataParallel 'module.' prefix
        and falls back to non-strict loading with a warning on mismatch."""
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise QgsProcessingException(
                "To load a state_dict, we need the U-Net architecture. This "
                "pipeline uses segmentation_models_pytorch (smp). Install it "
                "with:\n"
                "    pip install segmentation-models-pytorch\n\n"
                "If your model was NOT trained with smp, save it as a full "
                "model object instead:  torch.save(model, 'unet_best.pth')."
            )

        feedback.pushInfo(
            f"Building smp.Unet(encoder='{encoder_name}', "
            f"in_channels={in_channels}, classes={num_classes})..."
        )
        try:
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,  # loading our own weights
                in_channels=in_channels,
                classes=num_classes,
            )
        except Exception as e:
            raise QgsProcessingException(
                f"smp.Unet construction failed with encoder='{encoder_name}': {e}\n"
                "Check the encoder name against the smp docs "
                "(https://smp.readthedocs.io/en/latest/encoders.html)."
            )

        # DataParallel-trained checkpoints have a 'module.' prefix on every key
        if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            feedback.pushInfo("Stripped 'module.' prefix from state_dict keys.")

        # Try strict first — the honest error
        try:
            model.load_state_dict(state_dict, strict=True)
            feedback.pushInfo("state_dict loaded (strict).")
            return model
        except RuntimeError as strict_err:
            feedback.pushWarning(
                f"Strict load_state_dict failed: {strict_err}"
            )

        # Fall back to non-strict and report what didn't match
        result = model.load_state_dict(state_dict, strict=False)
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))
        if missing or unexpected:
            feedback.pushWarning(
                f"Non-strict load: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys. "
                f"First missing: {missing[:3]}. First unexpected: {unexpected[:3]}. "
                "If output looks wrong, the encoder/channels/classes settings "
                "likely don't match how the model was trained."
            )
        return model

    @staticmethod
    def _to_multipolygon(geom):
        """Return a strictly MultiPolygon-typed QgsGeometry from any input.

        Works when the input is a Polygon, MultiPolygon, or GeometryCollection
        (which makeValid/intersection can produce). Non-polygon parts (points,
        lines) are dropped. Returns None if no polygon parts exist.
        """
        if geom is None or geom.isEmpty():
            return None

        polys = []
        for part in geom.asGeometryCollection():
            if part is None or part.isEmpty():
                continue
            ptype = QgsWkbTypes.flatType(part.wkbType())
            if ptype == QgsWkbTypes.Polygon:
                polys.append(part)
            elif ptype == QgsWkbTypes.MultiPolygon:
                for sub in part.asGeometryCollection():
                    if (sub is not None and not sub.isEmpty()
                            and QgsWkbTypes.flatType(sub.wkbType())
                            == QgsWkbTypes.Polygon):
                        polys.append(sub)
            # Points and lines are silently dropped.

        if not polys:
            return None

        combined = QgsGeometry.collectGeometry(polys)
        if combined is None or combined.isEmpty():
            return None
        if not combined.isMultipart():
            combined.convertToMultiType()
        if QgsWkbTypes.flatType(combined.wkbType()) != QgsWkbTypes.MultiPolygon:
            return None
        return combined

    def _merge_intersecting(self, records, tolerance, feedback):
        """Dissolve polygons that overlap or touch (or lie within `tolerance`
        of each other) into single polygons, separately for each class.

        Polygons are first grouped into connected clusters with a spatial
        index, and each cluster is unioned on its own. This scales to tens of
        thousands of polygons, unlike one giant unaryUnion.

        records: list of (QgsGeometry, class_id, tile_name)
        returns: list of (QgsGeometry MultiPolygon, class_id, tiles_str, n_merged)
        """
        from collections import defaultdict

        by_cls = defaultdict(list)
        for geom, cls, tile in records:
            by_cls[cls].append((geom, tile))

        # Buffering each polygon by `tolerance` fuses polygons up to
        # 2 * tolerance apart (same behaviour as before).
        reach = 2.0 * tolerance

        out = []
        for cls, items in by_cls.items():
            if feedback.isCanceled():
                break
            geoms = [g for g, _ in items]
            tiles = [t for _, t in items]
            n = len(geoms)

            index = QgsSpatialIndex()
            for i, g in enumerate(geoms):
                f = QgsFeature(i)
                f.setGeometry(g)
                index.addFeature(f)

            parent = list(range(n))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            for i, g in enumerate(geoms):
                if i % 1000 == 0 and feedback.isCanceled():
                    break
                bb = QgsRectangle(g.boundingBox())
                if reach > 0:
                    bb.grow(reach)
                for j in index.intersects(bb):
                    if j <= i:
                        continue
                    ri, rj = find(i), find(j)
                    if ri == rj:
                        continue
                    if reach > 0:
                        close = g.distance(geoms[j]) <= reach
                    else:
                        close = g.intersects(geoms[j])
                    if close:
                        parent[rj] = ri

            clusters = defaultdict(list)
            for i in range(n):
                clusters[find(i)].append(i)

            for members in clusters.values():
                if len(members) == 1:
                    final = geoms[members[0]]
                else:
                    parts = [geoms[i] for i in members]
                    if tolerance > 0:
                        parts = [p.buffer(tolerance, 4) for p in parts]
                    final = QgsGeometry.unaryUnion(parts)
                    if final is None or final.isEmpty():
                        continue
                    if tolerance > 0:
                        final = final.buffer(-tolerance, 4)
                    if final is None or final.isEmpty():
                        continue
                    if not final.isGeosValid():
                        final = final.makeValid()
                mp = self._to_multipolygon(final)
                if mp is None:
                    continue

                tile_str = ",".join(sorted({tiles[i] for i in members}))
                if len(tile_str) > 254:          # Shapefile string limit
                    tile_str = tile_str[:251] + "..."
                out.append((mp, cls, tile_str, len(members)))

        feedback.pushInfo(
            f"Merged {len(records)} polygons into {len(out)} "
            f"(tolerance {tolerance})."
        )
        return out

    @staticmethod
    def _mask_to_polygons(mask):
        """Convert a (H, W) uint8 binary mask to a list of (N, 2) polygon
        vertex arrays in (col, row) tile-pixel coords. Uses cv2 when
        available (nicer contours), falls back to GDAL polygonize."""
        if mask.sum() == 0:
            return []

        # Preferred: cv2.findContours
        try:
            import cv2
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            out = []
            for c in contours:
                if len(c) < 3:
                    continue
                out.append(c.reshape(-1, 2).astype(float))
            return out
        except ImportError:
            pass

        # Fallback: GDAL polygonize on an in-memory raster
        try:
            from osgeo import gdal, ogr
        except ImportError:
            return []

        h, w = mask.shape
        mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
        mem.GetRasterBand(1).WriteArray(mask.astype(np.uint8))
        band = mem.GetRasterBand(1)

        ogr_ds = ogr.GetDriverByName("Memory").CreateDataSource("")
        layer = ogr_ds.CreateLayer("poly", srs=None, geom_type=ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn("DN", ogr.OFTInteger))
        gdal.Polygonize(band, band, layer, 0, [], callback=None)

        out = []
        for feat in layer:
            if feat.GetField("DN") == 0:
                continue
            geom = feat.GetGeometryRef()
            if geom is None:
                continue
            ring = geom.GetGeometryRef(0)
            if ring is None:
                continue
            n = ring.GetPointCount()
            if n < 3:
                continue
            out.append(
                np.array(
                    [(ring.GetX(i), ring.GetY(i)) for i in range(n)],
                    dtype=float,
                )
            )
        return out

    @staticmethod
    def _parse_label_line(line, tile_w, tile_h, mode):
        """Parse one line of a YOLO label file into tile-pixel vertices.

        Auto-detects whether the line is detection (5 values: class cx cy w h)
        or segmentation (odd count >= 7: class x1 y1 x2 y2 ...). Both are
        normalized to [0, 1] relative to the tile.

        mode = MODE_POLYGONS: keep polygon vertices as-is; if the line is a
            bbox, expand to a 4-corner rectangle.
        mode = MODE_BBOXES: always emit a 4-corner rectangle. For polygon
            lines, that rectangle is the vertex-wise bbox.

        Returns (class_id, ndarray shape (n, 2)) in tile pixels, or None.
        """
        vals = line.split()
        if len(vals) < 5:
            return None
        try:
            cls = int(float(vals[0]))
            coords = [float(v) for v in vals[1:]]
        except ValueError:
            return None

        # --- Detection (bbox) format ---
        if len(vals) == 5:
            cx, cy, bw, bh = coords
            x0 = (cx - bw / 2.0) * tile_w
            y0 = (cy - bh / 2.0) * tile_h
            x1 = (cx + bw / 2.0) * tile_w
            y1 = (cy + bh / 2.0) * tile_h
            return cls, np.array(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float
            )

        # --- Segmentation (polygon) format ---
        if len(vals) >= 7 and len(coords) % 2 == 0:
            pts = np.array(coords, dtype=float).reshape(-1, 2)
            pts[:, 0] *= tile_w
            pts[:, 1] *= tile_h
            if mode == GreenhouseDetectionAlgorithm.MODE_BBOXES:
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                pts = np.array(
                    [[x_min, y_min], [x_max, y_min],
                     [x_max, y_max], [x_min, y_max]], dtype=float,
                )
            return cls, pts

        return None

    @staticmethod
    def _check_disk_space(folder, needed_bytes, what, feedback):
        """Refuse to start if the work folder's disk can't hold the output."""
        try:
            free = shutil.disk_usage(str(folder)).free
        except OSError:
            return
        if needed_bytes > free:
            raise QgsProcessingException(
                f"Not enough disk space for {what}: needs about "
                f"{needed_bytes / 1e9:.1f} GB, only {free / 1e9:.1f} GB free in "
                f"{folder}. Choose another work folder, increase the pixel "
                "size or reduce the extent."
            )
        feedback.pushInfo(
            f"Disk check for {what}: ~{needed_bytes / 1e9:.2f} GB needed, "
            f"{free / 1e9:.1f} GB free."
        )

    def _render_layer_to_geotiff(self, raster_layer, extent, pixel_size,
                                  target_crs, output_path, max_megapixels,
                                  feedback):
        """Render any QgsRasterLayer (XYZ, WMS, GDAL) to a GeoTIFF at the given
        extent/pixel size. Triggers on-the-fly tile downloads for XYZ layers."""
        provider = raster_layer.dataProvider()
        if provider is None:
            raise QgsProcessingException("Input layer has no data provider.")

        width_px = max(1, int(round(extent.width() / pixel_size)))
        height_px = max(1, int(round(extent.height() / pixel_size)))
        mpix = width_px * height_px / 1e6
        feedback.pushInfo(
            f"Render size: {width_px} x {height_px} px ({mpix:.0f} MP)"
        )

        # Safety cap: a zoomed-out extent at a fine pixel size can ask for a
        # raster of tens of GB (and, for XYZ, hundreds of thousands of tile
        # downloads), which takes QGIS — or the whole machine — down.
        if max_megapixels > 0 and mpix > max_megapixels:
            suggested = pixel_size * math.sqrt(mpix / max_megapixels)
            raise QgsProcessingException(
                f"The rendered raster would be {width_px} x {height_px} px "
                f"({mpix:.0f} MP, ~{width_px * height_px * 3 / 1e9:.1f} GB), "
                f"above the limit of {max_megapixels:.0f} MP.\n"
                f"Either shrink the processing extent / AOI, increase 'Pixel "
                f"size' to at least {suggested:.3g}, or raise 'Max rendered "
                "raster size' if you are sure the machine can handle it."
            )

        out_dir = Path(output_path).parent
        # Rendered raster + tiles (PNG, roughly half of raw) + headroom
        self._check_disk_space(
            out_dir, int(width_px * height_px * 3 * 1.8), "the rendered raster "
            "and tiles", feedback
        )

        pipe = QgsRasterPipe()
        if not pipe.set(provider.clone()):
            raise QgsProcessingException(
                "Failed to build a raster pipe from the input layer's provider."
            )

        writer = QgsRasterFileWriter(output_path)
        writer.setOutputFormat("GTiff")
        create_opts = ["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"]
        setter = getattr(writer, "setCreationOptions", None) or \
            getattr(writer, "setCreateOptions", None)
        if setter is not None:
            setter(create_opts)

        # Let the Processing "Cancel" button stop the render too.
        block_fb = QgsRasterBlockFeedback()
        try:
            feedback.canceled.connect(block_fb.cancel)
        except Exception:
            pass

        err = writer.writeRaster(
            pipe, width_px, height_px, extent, target_crs,
            QgsProject.instance().transformContext(), block_fb,
        )
        if feedback.isCanceled():
            raise QgsProcessingException("Canceled while rendering.")
        # QgsRasterFileWriter.NoError == 0
        if err != 0:
            raise QgsProcessingException(
                f"Rendering the layer to GeoTIFF failed (error code {err}). "
                "For XYZ layers, check your internet connection and that the "
                "extent is reasonable."
            )

    # ---- Main -----------------------------------------------------------------

    def processAlgorithm(self, parameters, context, feedback):
        # Deferred imports so the algorithm can still be registered without them,
        # and so we can give a clean error message.
        try:
            from PIL import Image
        except ImportError:
            raise QgsProcessingException(
                "Pillow is not installed in QGIS's Python. Install it with:\n"
                "    pip install pillow"
            )
        try:
            from osgeo import gdal
        except ImportError:
            raise QgsProcessingException(
                "GDAL Python bindings are unexpectedly missing inside QGIS."
            )

        # -- Read parameters ---------------------------------------------------
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        if raster_layer is None:
            raise QgsProcessingException("Invalid input raster.")

        arch = self.parameterAsEnum(parameters, self.ARCH, context)
        weights_path = Path(self.parameterAsFile(parameters, self.WEIGHTS, context))
        mode = self.parameterAsEnum(parameters, self.MODE, context)
        mode_name = "bounding boxes" if mode == self.MODE_BBOXES else "polygons"
        arch_name = {
            self.ARCH_YOLO: "YOLO (detection / instance segmentation)",
            self.ARCH_UNET: "U-Net",
            self.ARCH_YOLO_SEM: "YOLO semantic segmentation",
            self.ARCH_SAM3: "SAM3",
        }.get(arch, "YOLO")
        feedback.pushInfo(f"Architecture: {arch_name}   Detection mode: {mode_name}")
        tile_size = self.parameterAsInt(parameters, self.TILE_SIZE, context)
        overlap = self.parameterAsInt(parameters, self.OVERLAP, context)
        imgsz = self.parameterAsInt(parameters, self.IMGSZ, context)
        conf = self.parameterAsDouble(parameters, self.CONF, context)
        iou = self.parameterAsDouble(parameters, self.IOU, context)
        batch = max(1, self.parameterAsInt(parameters, self.BATCH, context))
        max_mp = self.parameterAsDouble(parameters, self.MAX_MEGAPIXELS, context)
        out_crs = self.parameterAsCrs(parameters, self.OUTPUT_CRS, context)

        unet_input_size = self.parameterAsInt(parameters, self.UNET_INPUT_SIZE, context)
        unet_threshold = self.parameterAsDouble(parameters, self.UNET_THRESHOLD, context)
        unet_class_id = self.parameterAsInt(parameters, self.UNET_CLASS_ID, context)
        unet_encoder = self.parameterAsString(parameters, self.UNET_ENCODER, context)
        unet_in_channels = self.parameterAsInt(parameters, self.UNET_IN_CHANNELS, context)
        unet_num_classes = self.parameterAsInt(parameters, self.UNET_NUM_CLASSES, context)

        sam3_text_raw = self.parameterAsString(parameters, self.SAM3_TEXT, context) or ""
        sam3_text = [t.strip() for t in sam3_text_raw.split(",") if t.strip()]
        sam3_conf = self.parameterAsDouble(parameters, self.SAM3_CONF, context)
        det_param = self.parameterAsFile(parameters, self.SAM3_DETECTOR, context)
        sam3_detector = Path(det_param) if det_param else None
        sam3_prompt_mode = self.parameterAsEnum(parameters, self.SAM3_PROMPT_MODE, context)
        sam3_text_fallback = self.parameterAsBool(parameters, self.SAM3_TEXT_FALLBACK, context)

        # Fail fast on missing files, before the (possibly long) tiling step.
        if not weights_path.exists():
            raise QgsProcessingException(f"Weights not found: {weights_path}")

        # sam3.pt can't be loaded by YOLO() — if the user left the
        # architecture on a YOLO option, switch to SAM3 automatically.
        if arch != self.ARCH_SAM3 and "sam3" in weights_path.stem.lower():
            feedback.pushWarning(
                f"'{weights_path.name}' looks like SAM3 weights but the "
                f"architecture was set to '{arch_name}'. Switching to SAM3."
            )
            arch = self.ARCH_SAM3

        if arch == self.ARCH_SAM3:
            if sam3_detector is not None and \
                    "sam" in sam3_detector.stem.lower():
                raise QgsProcessingException(
                    f"'{sam3_detector.name}' was given as the box detector, but "
                    "the box detector must be a YOLO model (e.g. your trained "
                    "greenhouse best.pt). Put sam3.pt in 'Model weights' and "
                    "the YOLO model in '[SAM3] Box detector weights' — or leave "
                    "the detector empty for text-only SAM3."
                )
            if sam3_detector is not None and \
                    sam3_detector.resolve() == weights_path.resolve():
                raise QgsProcessingException(
                    "'Model weights' and '[SAM3] Box detector weights' point to "
                    "the same file. Model weights = sam3.pt, box detector = "
                    "your YOLO model."
                )
            if sam3_detector is not None and not sam3_detector.exists():
                raise QgsProcessingException(
                    f"Box detector weights not found: {sam3_detector}"
                )
            if not sam3_text and sam3_detector is None:
                raise QgsProcessingException(
                    "SAM3 needs a prompt: fill '[SAM3] Text prompt' and/or "
                    "'[SAM3] Box detector weights'."
                )
            feedback.pushInfo(
                f"SAM3 prompts — text: {sam3_text or 'none'}; detector: "
                f"{sam3_detector.name if sam3_detector else 'none'}; box mode: "
                + ("one mask per box" if sam3_prompt_mode == self.SAM3_PROMPT_PER_BOX
                   else "exemplar + text")
            )

        step = tile_size - overlap
        if step <= 0:
            raise QgsProcessingException("Overlap must be smaller than tile size.")

        work_param = self.parameterAsString(parameters, self.WORK_FOLDER, context)
        if work_param:
            work_dir = Path(work_param)
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="greenhouse_"))
        feedback.pushInfo(f"Work folder: {work_dir}")

        # -- Read AOI layer (optional) ----------------------------------------
        aoi_source = self.parameterAsSource(parameters, self.AOI_LAYER, context)
        aoi_clip = self.parameterAsBool(parameters, self.AOI_CLIP, context)
        do_merge = self.parameterAsBool(parameters, self.MERGE, context)
        merge_tol = self.parameterAsDouble(parameters, self.MERGE_TOLERANCE, context)
        aoi_geom_native = None
        aoi_native_crs = None
        if aoi_source is not None:
            aoi_native_crs = aoi_source.sourceCrs()
            parts = []
            for feat in aoi_source.getFeatures():
                if feat.hasGeometry():
                    parts.append(QgsGeometry(feat.geometry()))
            if not parts:
                raise QgsProcessingException(
                    "The AOI layer has no usable features. If you enabled "
                    "'Selected features only', make sure at least one polygon "
                    "is selected."
                )
            if len(parts) == 1:
                aoi_geom_native = QgsGeometry(parts[0])
            else:
                aoi_geom_native = QgsGeometry.unaryUnion(parts)
            if aoi_geom_native is None or aoi_geom_native.isEmpty():
                raise QgsProcessingException("AOI dissolved geometry is empty.")
            feedback.pushInfo(
                f"AOI: {len(parts)} feature(s) from '{aoi_source.sourceName()}' "
                f"[{aoi_native_crs.authid()}]."
            )

        # -- Resolve input to a real GeoTIFF that GDAL can open ---------------
        raster_source = raster_layer.source()
        provider_name = (
            raster_layer.dataProvider().name() if raster_layer.dataProvider() else ""
        )

        extent = None
        if parameters.get(self.EXTENT):
            layer_crs = raster_layer.crs() if raster_layer.crs().isValid() \
                else QgsCoordinateReferenceSystem("EPSG:3857")
            extent = self.parameterAsExtent(parameters, self.EXTENT, context, layer_crs)

        pixel_size = self.parameterAsDouble(parameters, self.PIXEL_SIZE, context)

        # If AOI given, its bbox (in the layer's CRS) becomes the render extent.
        # Explicit EXTENT still wins if both are provided.
        if aoi_geom_native is not None and (extent is None or extent.isEmpty()):
            render_crs_for_aoi = raster_layer.crs() if raster_layer.crs().isValid() \
                else QgsCoordinateReferenceSystem("EPSG:3857")
            aoi_geom_in_render = QgsGeometry(aoi_geom_native)
            if aoi_native_crs != render_crs_for_aoi:
                aoi_geom_in_render.transform(QgsCoordinateTransform(
                    aoi_native_crs, render_crs_for_aoi, QgsProject.instance()
                ))
            extent = aoi_geom_in_render.boundingBox()
            feedback.pushInfo(
                f"Using AOI bbox as processing extent: {extent.width():.1f} x "
                f"{extent.height():.1f} {render_crs_for_aoi.authid()} units."
            )

        needs_render = provider_name != "gdal"
        if not needs_render:
            probe = gdal.Open(raster_source)
            if probe is None:
                needs_render = True
            probe = None
        if extent is not None and not extent.isEmpty():
            # Even a plain GDAL raster gets rendered/clipped when an extent is set.
            needs_render = True

        if needs_render:
            if extent is None or extent.isEmpty():
                raise QgsProcessingException(
                    f"This layer isn't a plain GDAL raster (provider = "
                    f"'{provider_name}'). To use it, set both 'Processing extent' "
                    "and 'Pixel size' so it can be rendered to a temporary GeoTIFF."
                )
            if pixel_size <= 0:
                raise QgsProcessingException(
                    "'Pixel size' must be greater than 0 to render the layer."
                )

            render_crs = raster_layer.crs() if raster_layer.crs().isValid() \
                else QgsCoordinateReferenceSystem("EPSG:3857")
            raster_path = str(work_dir / "input_render.tif")
            feedback.pushInfo(
                f"Rendering layer to {raster_path} "
                f"(extent {extent.width():.1f} x {extent.height():.1f} "
                f"{render_crs.authid()} units, pixel {pixel_size})..."
            )
            self._render_layer_to_geotiff(
                raster_layer, extent, pixel_size, render_crs, raster_path,
                max_mp, feedback,
            )
        else:
            raster_path = raster_source

        feedback.pushInfo(f"Input raster: {raster_path}")
        ds = gdal.Open(raster_path)
        if ds is None:
            raise QgsProcessingException(f"GDAL could not open raster: {raster_path}")

        gt = ds.GetGeoTransform()
        is_identity = gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        if is_identity:
            gcps = ds.GetGCPs()
            if gcps:
                gt = gdal.GCPsToGeoTransform(gcps)
                if gt is None:
                    raise QgsProcessingException(
                        "GDAL failed to compute a geotransform from the raster's GCPs."
                    )
                feedback.pushInfo("Using geotransform derived from GCPs.")
            else:
                raise QgsProcessingException(
                    f"{raster_path} has no georeferencing (identity transform, no GCPs)."
                )

        src_crs = raster_layer.crs()
        if not src_crs.isValid():
            raise QgsProcessingException("Input raster has no valid CRS.")

        width = ds.RasterXSize
        height = ds.RasterYSize
        feedback.pushInfo(f"Raster CRS: {src_crs.authid()}   size: {width} x {height}")

        if not needs_render:
            # Tiles alone for a big source raster (PNG ≈ half of raw RGB)
            self._check_disk_space(
                work_dir, int(width * height * 3 * 0.8), "the tiles", feedback
            )

        # Prepare AOI geometry in raster (source) CRS for tile-skip tests.
        aoi_geom_src = None
        if aoi_geom_native is not None:
            aoi_geom_src = QgsGeometry(aoi_geom_native)
            if aoi_native_crs != src_crs:
                try:
                    aoi_geom_src.transform(QgsCoordinateTransform(
                        aoi_native_crs, src_crs, QgsProject.instance()
                    ))
                except Exception as e:
                    raise QgsProcessingException(
                        f"Failed to reproject AOI to raster CRS: {e}"
                    )
            aoi_bbox_src = aoi_geom_src.boundingBox()
        else:
            aoi_bbox_src = None

        # -- Tile --------------------------------------------------------------
        tiles_dir = work_dir / "tiles"
        if tiles_dir.exists():
            # Don't mix tiles from an earlier run into this one.
            for old in tiles_dir.glob("tile_*.png"):
                old.unlink()
        tiles_dir.mkdir(exist_ok=True)
        index_file = work_dir / "tile_index.csv"

        n_rows = (height + step - 1) // step
        n_cols = (width + step - 1) // step
        total_tiles = max(1, n_rows * n_cols)
        feedback.pushInfo(
            f"Tiling into ~{total_tiles} tiles of {tile_size}px (overlap {overlap}px)..."
        )
        if total_tiles > self.TILE_WARN:
            feedback.pushWarning(
                f"{total_tiles} tiles is a lot — this will take a long time. "
                "Consider a coarser pixel size or a smaller extent/AOI."
            )

        count = 0
        skipped_aoi = 0
        with open(index_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tile_name", "col_off", "row_off", "width", "height"])
            for row in range(0, height, step):
                if feedback.isCanceled():
                    return {}
                for col in range(0, width, step):
                    win_w = min(tile_size, width - col)
                    win_h = min(tile_size, height - row)
                    if win_w <= 0 or win_h <= 0:
                        continue

                    # AOI tile-skip: compute this tile's map extent and test.
                    if aoi_geom_src is not None:
                        tx0 = gt[0] + col * gt[1] + row * gt[2]
                        ty0 = gt[3] + col * gt[4] + row * gt[5]
                        tx1 = gt[0] + (col + win_w) * gt[1] + (row + win_h) * gt[2]
                        ty1 = gt[3] + (col + win_w) * gt[4] + (row + win_h) * gt[5]
                        tile_rect = QgsRectangle(
                            min(tx0, tx1), min(ty0, ty1),
                            max(tx0, tx1), max(ty0, ty1),
                        )
                        if not aoi_bbox_src.intersects(tile_rect):
                            skipped_aoi += 1
                            continue
                        if not aoi_geom_src.intersects(
                            QgsGeometry.fromRect(tile_rect)
                        ):
                            skipped_aoi += 1
                            continue

                    arr = ds.ReadAsArray(col, row, win_w, win_h)
                    if arr is None:
                        continue

                    # Normalize to HxWx3 uint8
                    if arr.ndim == 2:
                        arr = np.stack([arr, arr, arr], axis=-1)
                    else:
                        arr = np.transpose(arr, (1, 2, 0))
                        if arr.shape[2] >= 3:
                            arr = arr[:, :, :3]
                        elif arr.shape[2] == 1:
                            arr = np.repeat(arr, 3, axis=-1)
                        else:  # 2 bands
                            pad = np.zeros_like(arr[..., :1])
                            arr = np.concatenate([arr, pad], axis=-1)

                    if arr.dtype != np.uint8:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)

                    # Skip completely empty (nodata/black) tiles — common at
                    # the edges of rendered XYZ rasters.
                    if not arr.any():
                        continue

                    tile_name = f"tile_{row}_{col}.png"
                    Image.fromarray(np.ascontiguousarray(arr)).save(
                        tiles_dir / tile_name
                    )
                    writer.writerow([tile_name, col, row, win_w, win_h])
                    count += 1
                    del arr

                    if count % 20 == 0:
                        feedback.setProgress(int(count / total_tiles * 40))

        feedback.pushInfo(f"{count} tiles written.")
        if aoi_geom_src is not None:
            feedback.pushInfo(f"{skipped_aoi} tiles skipped (outside AOI).")
        feedback.setProgress(40)

        # -- Predict -----------------------------------------------------------
        if not weights_path.exists():
            raise QgsProcessingException(f"Weights not found: {weights_path}")

        images = sorted(
            p for p in tiles_dir.glob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        )
        if not images:
            raise QgsProcessingException(f"No tiles found in {tiles_dir}")

        preds_project = work_dir / "results"
        preds_project.mkdir(exist_ok=True)

        if arch == self.ARCH_SAM3:
            labels_dir = self._run_sam3_inference(
                weights_path, images, preds_project,
                sam3_text, sam3_conf, sam3_detector,
                sam3_prompt_mode, sam3_text_fallback,
                imgsz, conf, iou, batch, work_dir, feedback,
            )
        elif arch == self.ARCH_UNET:
            labels_dir = self._run_unet_inference(
                weights_path, images, preds_project,
                unet_input_size, unet_threshold, unet_class_id,
                unet_encoder, unet_in_channels, unet_num_classes,
                feedback,
            )
        else:
            model = self._load_yolo_model(weights_path, work_dir, feedback)
            device = self._pick_device()
            task = str(getattr(model, "task", "") or "").lower()
            feedback.pushInfo(f"YOLO model task: '{task or 'unknown'}'")

            use_semantic = arch == self.ARCH_YOLO_SEM or "sem" in task
            if use_semantic and arch != self.ARCH_YOLO_SEM:
                feedback.pushInfo(
                    "These are semantic-segmentation weights — switching to "
                    "YOLO semantic handling automatically."
                )

            if use_semantic:
                labels_dir = self._run_yolo_semantic_inference(
                    model, images, preds_project, imgsz, device,
                    unet_class_id, batch, feedback,
                )
            else:
                labels_dir = self._run_yolo_inference(
                    model, images, preds_project, imgsz, device,
                    conf, iou, batch, feedback,
                )
            del model
            self._free_gpu()

        if feedback.isCanceled():
            ds = None
            return {}
        feedback.setProgress(80)

        # -- Postprocess -> feature sink --------------------------------------
        index = {}
        with open(index_file, newline="") as f:
            for row in csv.DictReader(f):
                index[row["tile_name"]] = {
                    "col_off": int(row["col_off"]),
                    "row_off": int(row["row_off"]),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                }

        xform = QgsCoordinateTransform(src_crs, out_crs, QgsProject.instance())

        # AOI in output CRS, for filtering/clipping polygons at the end.
        aoi_geom_out = None
        if aoi_geom_native is not None:
            aoi_geom_out = QgsGeometry(aoi_geom_native)
            if aoi_native_crs != out_crs:
                try:
                    aoi_geom_out.transform(QgsCoordinateTransform(
                        aoi_native_crs, out_crs, QgsProject.instance()
                    ))
                except Exception as e:
                    raise QgsProcessingException(
                        f"Failed to reproject AOI to output CRS: {e}"
                    )

        fields = QgsFields()
        fields.append(QgsField("class_id", QVariant.Int))
        fields.append(QgsField("tile", QVariant.String))
        fields.append(QgsField("area_m2", QVariant.Double))
        fields.append(QgsField("n_merged", QVariant.Int))

        sink, dest_id = self.parameterAsSink(
            parameters, self.OUTPUT, context,
            fields, QgsWkbTypes.MultiPolygon, out_crs,
        )
        if sink is None:
            raise QgsProcessingException("Could not create output sink.")

        label_files = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []
        if not label_files:
            feedback.pushWarning("No label files found — the model detected nothing.")
            ds = None
            return {self.OUTPUT: dest_id}

        n_polys = 0
        dropped_aoi = 0
        dropped_shape = 0
        total_area = 0.0
        records = []   # (geom, class_id, tile_name) before merging
        for lf in label_files:
            if feedback.isCanceled():
                break

            tile_name = lf.stem + ".png"
            if tile_name not in index:
                tile_name = lf.stem + ".jpg"
            if tile_name not in index:
                feedback.pushInfo(f"Skipping {lf.name} — no matching tile in index.")
                continue

            meta = index[tile_name]
            w, h = meta["width"], meta["height"]
            col0, row0 = meta["col_off"], meta["row_off"]

            for line in lf.read_text().strip().splitlines():
                parsed = self._parse_label_line(line, w, h, mode)
                if parsed is None:
                    continue
                cls, pts = parsed

                # tile px -> global px
                px_col = pts[:, 0] + col0
                px_row = pts[:, 1] + row0
                # global px -> map coords via GDAL geotransform
                x = gt[0] + px_col * gt[1] + px_row * gt[2]
                y = gt[3] + px_col * gt[4] + px_row * gt[5]

                # source-CRS -> output-CRS, vertex by vertex
                qgs_points = []
                for xi, yi in zip(x, y):
                    try:
                        qgs_points.append(
                            xform.transform(QgsPointXY(float(xi), float(yi)))
                        )
                    except Exception:
                        pass
                if len(qgs_points) < 3:
                    continue

                geom = QgsGeometry.fromPolygonXY([qgs_points])
                if geom.isEmpty():
                    continue
                if not geom.isGeosValid():
                    geom = geom.makeValid()
                    if geom.isEmpty():
                        continue

                # AOI filter / clip
                if aoi_geom_out is not None:
                    if not aoi_geom_out.intersects(geom):
                        dropped_aoi += 1
                        continue
                    if aoi_clip:
                        geom = geom.intersection(aoi_geom_out)
                        if geom.isEmpty():
                            dropped_aoi += 1
                            continue

                mp = self._to_multipolygon(geom)
                if mp is None:
                    dropped_shape += 1
                    continue
                geom = mp

                if geom.area() <= 0:
                    dropped_shape += 1
                    continue

                records.append((geom, cls, tile_name))

        # -- Merge intersecting polygons (optional) ---------------------------
        if do_merge and records:
            if merge_tol > 0 and out_crs.isGeographic():
                feedback.pushWarning(
                    "Output CRS is geographic, so the merge tolerance is in "
                    "DEGREES. Use a projected output CRS for a metric tolerance."
                )
            final = self._merge_intersecting(records, merge_tol, feedback)
        else:
            final = [(g, c, t, 1) for g, c, t in records]

        for geom, cls, tile_str, n_merged in final:
            area = geom.area()
            if area <= 0:
                continue
            feat = QgsFeature(fields)
            feat.setGeometry(geom)
            feat.setAttributes([cls, tile_str, float(area), int(n_merged)])
            sink.addFeature(feat, QgsFeatureSink.FastInsert)
            n_polys += 1
            total_area += area

        feedback.pushInfo(
            f"Collected {n_polys} polygons. "
            f"Total area ({out_crs.authid()} units^2): {total_area:.2f}"
        )
        if aoi_geom_out is not None:
            feedback.pushInfo(f"{dropped_aoi} polygons dropped by AOI filter.")
        if dropped_shape:
            feedback.pushInfo(
                f"{dropped_shape} polygons dropped (no polygonal parts after "
                "makeValid/clip — likely degenerate mask outlines)."
            )
        feedback.setProgress(100)

        ds = None  # close GDAL dataset
        return {self.OUTPUT: dest_id}