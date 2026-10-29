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
   "Scripts → Remote sensing → Detect greenhouses (YOLOv8-seg)".

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
    QgsProcessingParameterNumber,
    QgsProcessingParameterCrs,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterBoolean,
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
    AOI_LAYER = "AOI_LAYER"
    AOI_CLIP = "AOI_CLIP"
    OUTPUT = "OUTPUT"

    # ---- Algorithm metadata ---------------------------------------------------

    def tr(self, string):
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return GreenhouseDetectionAlgorithm()

    def name(self):
        return "detect_greenhouses"

    def displayName(self):
        return self.tr("Detect greenhouses (YOLOv8-seg)")

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

        # If AOI given, its bbox (in the layer's CRS) becomes the render extent.
        # Explicit EXTENT still wins if both are provided, so the user can
        # narrow down further.
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
        tiles_dir.mkdir(exist_ok=True)
        index_file = work_dir / "tile_index.csv"

        n_rows = (height + step - 1) // step
        n_cols = (width + step - 1) // step
        total_tiles = max(1, n_rows * n_cols)
        feedback.pushInfo(
            f"Tiling into ~{total_tiles} tiles of {tile_size}px (overlap {overlap}px)..."
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
                        # Same clip-to-uint8 behavior as the original script.
                        arr = np.clip(arr, 0, 255).astype(np.uint8)

                    tile_name = f"tile_{row}_{col}.png"
                    Image.fromarray(arr).save(tiles_dir / tile_name)
                    writer.writerow([tile_name, col, row, win_w, win_h])
                    count += 1

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
        dropped_aoi = 0
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
                    if geom.isEmpty():
                        continue

                area = geom.area()
                if area <= 0:
                    continue

                # AOI filter / clip
                if aoi_geom_out is not None:
                    if not aoi_geom_out.intersects(geom):
                        dropped_aoi += 1
                        continue
                    if aoi_clip:
                        clipped = geom.intersection(aoi_geom_out)
                        if clipped.isEmpty():
                            dropped_aoi += 1
                            continue
                        geom = clipped
                        area = geom.area()
                        if area <= 0:
                            dropped_aoi += 1
                            continue

                if not geom.isMultipart():
                    geom.convertToMultiType()

                feat = QgsFeature(fields)
                feat.setGeometry(geom)
                feat.setAttributes([cls, tile_name, float(area)])
                sink.addFeature(feat, QgsFeatureSink.FastInsert)

                n_polys += 1
                total_area += area

        feedback.pushInfo(
            f"Collected {n_polys} polygons. "
            f"Total area ({out_crs.authid()} units^2): {total_area:.2f}"
        )
        if aoi_geom_out is not None:
            feedback.pushInfo(f"{dropped_aoi} polygons dropped by AOI filter.")
        feedback.setProgress(100)

        ds = None  # close GDAL dataset
        return {self.OUTPUT: dest_id}