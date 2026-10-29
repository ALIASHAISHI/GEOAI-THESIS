"""
Greenhouse detection pipeline — QGIS Processing Algorithm.

INSTALL
-------
1. Install the extra Python packages into QGIS's own Python.
   - Windows (OSGeo4W): open the "OSGeo4W Shell" from the Start menu, then:
         pip install ultralytics pillow
   - Linux (system QGIS): use the same Python that runs QGIS, e.g.
         python3 -m pip install --user ultralytics pillow
   - macOS (QGIS.app): use the bundled Python, e.g.
         /Applications/QGIS.app/Contents/MacOS/bin/python3 -m pip install ultralytics pillow

   GDAL, PyQGIS and shapely are already shipped with QGIS.

2. In QGIS: Processing Toolbox → Scripts (the wrench-and-python icon) →
   "Add Script to Toolbox…" → pick this file. The tool will appear under
   "Scripts → Remote sensing → Detect greenhouses (YOLOv8-seg-multipolygon)".

USAGE
-----
Fill in the dialog:
  - Input raster: any georeferenced raster loaded in QGIS. This also works
    with XYZ tile services (Google/Bing/Esri satellite) and WMS/WCS layers.
    For those the layer is rendered to a temporary GeoTIFF first, and you
    MUST also fill "Processing extent" and "Pixel size".
  - YOLOv8 weights: path to your trained best.pt.
  - Tile size / overlap / imgsz / conf / iou: same meaning as in the CLI script.
  - Processing extent: bounding box to run detection on. Required for
    XYZ/WMS/WCS layers. Optional for regular GDAL rasters (if given, the
    raster is clipped to it).
  - Pixel size: ground sample distance for the rendered raster, in the
    layer's CRS units (meters for projected CRSs, degrees for EPSG:4326).
    Used only when the layer has to be rendered.
  - Output CRS: the CRS the polygons are written in. EPSG:3857 by default
    (matches the original script). Change it to your raster's CRS or a local
    UTM zone for accurate area_m2.
  - Work folder (optional): where intermediate tiles + YOLO output are kept.
    If empty, a temp folder is used.
  - Detected greenhouses: the output vector layer (GeoPackage, Shapefile,
    memory layer — whatever the user picks in the dialog).
"""

import csv
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
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterCrs,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFeatureSink,
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
)


class GreenhouseDetectionAlgorithm(QgsProcessingAlgorithm):
    """Tile a raster, run YOLOv8-seg on each tile, return polygons in map CRS."""

    INPUT_RASTER = "INPUT_RASTER"
    WEIGHTS = "WEIGHTS"
    TILE_SIZE = "TILE_SIZE"
    OVERLAP = "OVERLAP"
    IMGSZ = "IMGSZ"
    CONF = "CONF"
    IOU = "IOU"
    OUTPUT_CRS = "OUTPUT_CRS"
    WORK_FOLDER = "WORK_FOLDER"
    EXTENT = "EXTENT"
    PIXEL_SIZE = "PIXEL_SIZE"
    OUTPUT = "OUTPUT"
    SHAPE_MODE = "SHAPE_MODE"
    SIMPLIFY_TOL = "SIMPLIFY_TOL"

    SHAPE_CHOICES = [
        "Raw polygon (no reshape)",
        "Oriented minimum bounding box (rotated rectangle)",
        "Axis-aligned bounding box",
        "Convex hull",
        "Simplify (Douglas-Peucker)",
    ]

    # ---- Algorithm metadata ---------------------------------------------------

    def tr(self, string):
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return GreenhouseDetectionAlgorithm()

    def name(self):
        return "detect_greenhouses"

    def displayName(self):
        return self.tr("Detect greenhouses (YOLOv8-seg-multipolygonsssss)")

    def group(self):
        return self.tr("Remote sensing")

    def groupId(self):
        return "remote_sensing"

    def shortHelpString(self):
        return self.tr(
            "Tile a georeferenced raster, run a YOLOv8 segmentation model over "
            "each tile, and return the detected polygons as a vector layer in "
            "map coordinates.\n\n"
            "Requires the ultralytics and pillow Python packages to be installed "
            "in QGIS's Python environment. See the header of the script file for "
            "install instructions."
        )

    # ---- Parameters -----------------------------------------------------------

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER, self.tr("Input raster (georeferenced)")
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.WEIGHTS,
                self.tr("YOLOv8 weights (.pt)"),
                extension="pt",
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
                self.tr("Inference image size"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1024, minValue=64,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CONF,
                self.tr("Confidence threshold"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.35, minValue=0.0, maxValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.IOU,
                self.tr("NMS IoU threshold"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.45, minValue=0.0, maxValue=1.0,
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
            QgsProcessingParameterFile(
                self.WORK_FOLDER,
                self.tr("Work folder for tiles + YOLO output (optional)"),
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
        self.addParameter(
            QgsProcessingParameterEnum(
                self.SHAPE_MODE,
                self.tr("Polygon reshape"),
                options=self.SHAPE_CHOICES,
                defaultValue=1,   # oriented bbox
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SIMPLIFY_TOL,
                self.tr("Simplify tolerance (output CRS units — only used with 'Simplify')"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.5, minValue=0.0,
                optional=True,
            )
        )

    # ---- Helpers --------------------------------------------------------------

    def _render_layer_to_geotiff(self, raster_layer, extent, pixel_size,
                                  target_crs, output_path, feedback):
        """Render any QgsRasterLayer (XYZ, WMS, GDAL) to a GeoTIFF at the given
        extent/pixel size. Triggers on-the-fly tile downloads for XYZ layers."""
        provider = raster_layer.dataProvider()
        if provider is None:
            raise QgsProcessingException("Input layer has no data provider.")

        pipe = QgsRasterPipe()
        if not pipe.set(provider.clone()):
            raise QgsProcessingException(
                "Failed to build a raster pipe from the input layer's provider."
            )

        width_px = max(1, int(round(extent.width() / pixel_size)))
        height_px = max(1, int(round(extent.height() / pixel_size)))
        feedback.pushInfo(f"Render size: {width_px} x {height_px} px")

        writer = QgsRasterFileWriter(output_path)
        writer.setOutputFormat("GTiff")

        err = writer.writeRaster(
            pipe, width_px, height_px, extent, target_crs,
            QgsProject.instance().transformContext(),
        )
        # QgsRasterFileWriter.NoError == 0
        if err != 0:
            raise QgsProcessingException(
                f"Rendering the layer to GeoTIFF failed (error code {err}). "
                "For XYZ layers, check your internet connection and that the "
                "extent is reasonable."
            )


    def _reshape_geometry(self, geom, mode, tol):
        """Apply the chosen reshape and return a MultiPolygon (or None if empty)."""
        out = geom
        # if geom is None or geom.isEmpty():
        #     return None

        # if mode == 0:                                       # raw
        #     out = geom
        # elif mode == 1:                                     # oriented minimum bbox
        #     result = geom.orientedMinimumBoundingBox()
        #     out = result[0] if isinstance(result, tuple) else result
        # elif mode == 2:                                     # axis-aligned bbox
        #     out = QgsGeometry.fromRect(geom.boundingBox())
        # elif mode == 3:                                     # convex hull
        #     out = geom.convexHull()
        # elif mode == 4:                                     # simplify
        #     out = geom.simplify(tol) if tol > 0 else geom
        # else:
        #     out = geom

        # if out is None or out.isEmpty():
        #     return None
        out.convertToMultiType()
        return out
    
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
            from ultralytics import YOLO
        except ImportError:
            raise QgsProcessingException(
                "The 'ultralytics' package is not installed in QGIS's Python. "
                "Install it with:\n    pip install ultralytics"
            )
        try:
            from osgeo import gdal
        except ImportError:
            raise QgsProcessingException(
                "GDAL Python bindings are unexpectedly missing inside QGIS."
            )

        feedback.pushInfo("=== greenhouse script v2 (MultiPolygon + coerceToType) ===")

        # -- Read parameters ---------------------------------------------------
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        if raster_layer is None:
            raise QgsProcessingException("Invalid input raster.")

        weights_path = Path(self.parameterAsFile(parameters, self.WEIGHTS, context))
        tile_size = self.parameterAsInt(parameters, self.TILE_SIZE, context)
        overlap = self.parameterAsInt(parameters, self.OVERLAP, context)
        imgsz = self.parameterAsInt(parameters, self.IMGSZ, context)
        conf = self.parameterAsDouble(parameters, self.CONF, context)
        iou = self.parameterAsDouble(parameters, self.IOU, context)
        out_crs = self.parameterAsCrs(parameters, self.OUTPUT_CRS, context)
        shape_mode = self.parameterAsEnum(parameters, self.SHAPE_MODE, context)
        simplify_tol = self.parameterAsDouble(parameters, self.SIMPLIFY_TOL, context)

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

        # -- Resolve input to a real GeoTIFF that GDAL can open ---------------
        # XYZ tile layers (Google Satellite etc.) and WMS/WCS layers don't have
        # a file GDAL can open — their source() is a URL/URI. In that case we
        # render the requested extent to a temporary GeoTIFF and use that.
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
                raster_layer, extent, pixel_size, render_crs, raster_path, feedback
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
            # Fall back to GCPs if present, mirroring the original script.
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

        # -- Tile --------------------------------------------------------------
        tiles_dir = work_dir / "tiles"
        tiles_dir.mkdir(exist_ok=True)
        index_file = work_dir / "tile_index.csv"

        n_rows = (height + step - 1) // step
        n_cols = (width + step - 1) // step
        total_tiles = max(1, n_rows * n_cols)
        feedback.pushInfo(
            f"Tiling into ~{total_tiles} tiles of {tile_size}px (overlap {overlap}px)..."
        )

        count = 0
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
                        # Same clip-to-uint8 behavior as the original script.
                        arr = np.clip(arr, 0, 255).astype(np.uint8)

                    tile_name = f"tile_{row}_{col}.png"
                    Image.fromarray(arr).save(tiles_dir / tile_name)
                    writer.writerow([tile_name, col, row, win_w, win_h])
                    count += 1

                    if count % 20 == 0:
                        feedback.setProgress(int(count / total_tiles * 40))

        feedback.pushInfo(f"{count} tiles written.")
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

        feedback.pushInfo(f"Loading YOLO model: {weights_path}")
        model = YOLO(str(weights_path))

        preds_project = work_dir / "results"
        preds_project.mkdir(exist_ok=True)

        feedback.pushInfo(f"Running inference on {len(images)} tiles...")
        model.predict(
            source=[str(p) for p in images],
            imgsz=imgsz,
            device="cpu",
            conf=conf,
            iou=iou,
            save=False,
            save_txt=True,
            project=str(preds_project),
            name="yolo_seg",
            exist_ok=True,
            verbose=False,
        )
        labels_dir = preds_project / "yolo_seg" / "labels"
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

        fields = QgsFields()
        fields.append(QgsField("class_id", QVariant.Int))
        fields.append(QgsField("tile", QVariant.String))
        fields.append(QgsField("area_m2", QVariant.Double))

        sink, dest_id = self.parameterAsSink(
            parameters, self.OUTPUT, context,
            fields, QgsWkbTypes.MultiPolygon, out_crs,
        )
        if sink is None:
            raise QgsProcessingException("Could not create output sink.")

        label_files = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []
        if not label_files:
            feedback.pushWarning("No YOLO label files found — model detected nothing.")
            ds = None
            return {self.OUTPUT: dest_id}

        n_polys = 0
        total_area = 0.0
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
                vals = line.split()
                # YOLOv8-seg format: class x1 y1 x2 y2 ... (normalized tile pixels)
                if len(vals) < 7:
                    continue
                try:
                    cls = int(float(vals[0]))
                    coords = [float(v) for v in vals[1:]]
                except ValueError:
                    continue
                if len(coords) < 6 or len(coords) % 2 != 0:
                    continue

                pts = np.array(coords, dtype=float).reshape(-1, 2)
                # normalized tile px -> global px
                px_col = pts[:, 0] * w + col0
                px_row = pts[:, 1] * h + row0
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
                    if geom is None or geom.isEmpty():
                        continue

                # Walk parts and keep only polygonal ones. asGeometryCollection() returns
                # [geom] for single geometries and the parts list for real collections,
                # so this handles Polygon, MultiPolygon, and mixed GeometryCollection alike.
                polygonal = [
                    p for p in geom.asGeometryCollection()
                    if p is not None and not p.isEmpty()
                    and p.type() == QgsWkbTypes.PolygonGeometry
                ]
                if not polygonal:
                    continue

                # Merge into one clean MultiPolygon
                merged = QgsGeometry.unaryUnion(polygonal)
                if merged is None or merged.isEmpty():
                    continue

                shaped = self._reshape_geometry(merged, shape_mode, simplify_tol)
                if shaped is None:
                    continue

                area = shaped.area()
                if area <= 0:
                    continue

                feat = QgsFeature(fields)
                feat.setGeometry(shaped)
                feat.setAttributes([cls, tile_name, float(area)])
                sink.addFeature(feat, QgsFeatureSink.FastInsert)

                n_polys += 1
                total_area += area

        feedback.pushInfo(
            f"Collected {n_polys} polygons. "
            f"Total area ({out_crs.authid()} units^2): {total_area:.2f}"
        )
        feedback.setProgress(100)

        ds = None  # close GDAL dataset
        return {self.OUTPUT: dest_id}