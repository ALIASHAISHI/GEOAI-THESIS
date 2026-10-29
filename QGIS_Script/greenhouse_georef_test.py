"""
Georeferencing round-trip test for the greenhouse detection pipeline.

WHAT IT DOES
------------
It checks that the code which places predictions back on the map is
correct, without running any model. It uses your polygons as a "perfect
model":

  1. Your polygon layer (ground truth) is projected into the raster's
     pixel space with the inverse GDAL geotransform.
  2. The raster is tiled with EXACTLY the same grid as the detection tool
     (same tile size / overlap / step / tile names / tile_index.csv, and
     the same rendering for XYZ / WMS / WCS layers).
  3. On every tile, each polygon is clipped to the tile and written as a
     YOLO polygon label line (normalized 0-1, 6 decimals), like the models'
     output. Two routes:
       - "Exact vertices": the clipped polygon outline itself
         (tests only the geometry math).
       - "Rasterized mask": the clipped polygon is burnt into a tile mask
         and turned back into polygons with the same _mask_to_polygons()
         as U-Net / YOLO-semantic / SAM3 (also tests pixel quantization).
  4. The label files are read back and placed on the map with the SAME
     code as the detection tool (tile px -> global px -> geotransform ->
     output CRS, vertex by vertex).
  5. The pieces of every feature are unioned and compared with the
     original feature: IoU, centroid shift (dx, dy, in map units and in
     pixels), Hausdorff distance, coverage by the raster.

If georeferencing is right you get IoU ~ 1.0 and shift ~ 0 with the
"Exact" route. The "Mask" route gives slightly lower IoU (pixel
quantization) — a systematic shift of about half a pixel there means the
contour/pixel-corner convention is off (see the half-pixel option).

INSTALL
-------
Processing Toolbox -> Scripts -> "Add Script to Toolbox..." -> pick this
file. It appears under "Scripts -> Remote sensing -> Test greenhouse
georeferencing (round-trip IoU)". Needs only numpy, pillow and GDAL
(opencv is used when present, like in the detection tool).

TIPS
----
  - Use the SAME raster, tile size, overlap, extent/AOI bbox, pixel size
    and output CRS as your detection run — the tile grid depends on them.
  - Tick "Selected features only" on the polygon input to test a few
    features.
  - "Write debug tiles" saves each tile PNG with the label outlines drawn
    in red, so you can see whether the outlines sit on the greenhouses.
  - If you change the placement code in the detection tool, update the
    copied functions below (_parse_label_line, _to_multipolygon,
    _mask_to_polygons, placement block) so the test stays faithful.
"""

import csv
import math
import shutil
import statistics
import tempfile
from pathlib import Path

import numpy as np

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.PyQt.QtGui import QTransform
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingParameterCrs,
    QgsProcessingParameterExtent,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFile,
    QgsProcessingParameterFeatureSink,
    QgsProcessingOutputNumber,
    QgsProcessingOutputFile,
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


class GreenhouseGeorefTestAlgorithm(QgsProcessingAlgorithm):

    INPUT_RASTER = "INPUT_RASTER"
    INPUT_POLYGONS = "INPUT_POLYGONS"
    ROUTE = "ROUTE"
    TILE_SIZE = "TILE_SIZE"
    OVERLAP = "OVERLAP"
    EXTENT = "EXTENT"
    PIXEL_SIZE = "PIXEL_SIZE"
    MAX_MEGAPIXELS = "MAX_MEGAPIXELS"
    OUTPUT_CRS = "OUTPUT_CRS"
    SKIP_EMPTY = "SKIP_EMPTY"
    HALF_PIXEL = "HALF_PIXEL"
    IOU_PASS = "IOU_PASS"
    WRITE_DEBUG = "WRITE_DEBUG"
    WORK_FOLDER = "WORK_FOLDER"
    OUTPUT = "OUTPUT"
    OUTPUT_DIFF = "OUTPUT_DIFF"

    ROUTE_EXACT = 0
    ROUTE_MASK = 1

    # same value as GreenhouseDetectionAlgorithm.MODE_POLYGONS
    MODE_POLYGONS = 0
    MODE_BBOXES = 1

    # ---- metadata -------------------------------------------------------------

    def tr(self, s):
        return QCoreApplication.translate("Processing", s)

    def createInstance(self):
        return GreenhouseGeorefTestAlgorithm()

    def name(self):
        return "test_greenhouse_georef"

    def displayName(self):
        return self.tr("Test greenhouse georeferencing (round-trip IoU)")

    def group(self):
        return self.tr("Remote sensing")

    def groupId(self):
        return "remote_sensing"

    def shortHelpString(self):
        return self.tr(
            "Checks the placement code of 'Detect greenhouses' without a "
            "model. Your polygons are tiled with the same grid as the "
            "detector, written as YOLO label lines per tile, then placed "
            "back on the map with the detector's code and compared with the "
            "originals.\n\n"
            "Output layer: the placed polygons with per-feature IoU, "
            "centroid shift (map units and pixels) and Hausdorff distance. "
            "Optional difference layer: 'missed' (original not covered) and "
            "'extra' (placed outside the original).\n\n"
            "Exact route → IoU should be ~1.0. Mask route → slightly lower "
            "(pixel quantization); a constant ~0.5 px shift points to a "
            "pixel-centre vs pixel-corner offset.\n\n"
            "Use the same raster, tile size, overlap, extent and pixel size "
            "as the detection run."
        )

    # ---- parameters -----------------------------------------------------------

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT_RASTER, self.tr("Raster used for detection")))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT_POLYGONS,
            self.tr("Polygons to test (ground truth or predictions — "
                    "'Selected features only' supported)"),
            types=[QgsProcessing.TypeVectorPolygon]))
        self.addParameter(QgsProcessingParameterEnum(
            self.ROUTE, self.tr("Label route"),
            options=[
                self.tr("Exact vertices (tests geometry math only)"),
                self.tr("Rasterized mask → contours (like U-Net / "
                        "YOLO-semantic / SAM3)"),
            ],
            defaultValue=self.ROUTE_EXACT))
        self.addParameter(QgsProcessingParameterNumber(
            self.TILE_SIZE, self.tr("Tile size (pixels)"),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=640, minValue=64))
        self.addParameter(QgsProcessingParameterNumber(
            self.OVERLAP, self.tr("Tile overlap (pixels)"),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=64, minValue=0))
        self.addParameter(QgsProcessingParameterExtent(
            self.EXTENT,
            self.tr("Processing extent (same as detection run; empty = "
                    "polygons' bbox for XYZ/WMS)"),
            optional=True))
        self.addParameter(QgsProcessingParameterNumber(
            self.PIXEL_SIZE,
            self.tr("Pixel size for rendered raster (layer CRS units)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.5, minValue=0.0, optional=True))
        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_MEGAPIXELS,
            self.tr("Max rendered raster size (megapixels, 0 = no limit)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=400.0, minValue=0.0))
        self.addParameter(QgsProcessingParameterCrs(
            self.OUTPUT_CRS, self.tr("Output CRS (as in detection run)"),
            defaultValue="EPSG:3857"))
        self.addParameter(QgsProcessingParameterBoolean(
            self.SKIP_EMPTY,
            self.tr("Skip all-black tiles like the detector does (reads pixels)"),
            defaultValue=True))
        self.addParameter(QgsProcessingParameterBoolean(
            self.HALF_PIXEL,
            self.tr("[Mask route] Add +0.5 px to contour vertices "
                    "(test the pixel-centre fix)"),
            defaultValue=False))
        self.addParameter(QgsProcessingParameterNumber(
            self.IOU_PASS, self.tr("IoU pass threshold"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.95, minValue=0.0, maxValue=1.0))
        self.addParameter(QgsProcessingParameterBoolean(
            self.WRITE_DEBUG,
            self.tr("Write debug tiles (PNG with label outlines in red)"),
            defaultValue=False))
        self.addParameter(QgsProcessingParameterFile(
            self.WORK_FOLDER,
            self.tr("Work folder (labels, tile index, report; optional)"),
            behavior=QgsProcessingParameterFile.Folder, optional=True))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, self.tr("Placed polygons with IoU"),
            type=QgsProcessing.TypeVectorPolygon))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_DIFF,
            self.tr("Differences (missed / extra areas)"),
            type=QgsProcessing.TypeVectorPolygon,
            optional=True, createByDefault=True))

        self.addOutput(QgsProcessingOutputNumber("MEAN_IOU", self.tr("Mean IoU")))
        self.addOutput(QgsProcessingOutputNumber("MEDIAN_IOU", self.tr("Median IoU")))
        self.addOutput(QgsProcessingOutputNumber("GLOBAL_IOU", self.tr("Global IoU")))
        self.addOutput(QgsProcessingOutputNumber("MEAN_SHIFT_PX",
                                                 self.tr("Mean centroid shift (px)")))
        self.addOutput(QgsProcessingOutputNumber("N_PASS", self.tr("Features passing")))
        self.addOutput(QgsProcessingOutputFile("REPORT", self.tr("CSV report")))

    # ---- helpers copied from the detection tool (keep in sync) --------------

    @staticmethod
    def _to_multipolygon(geom):
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

    @staticmethod
    def _mask_to_polygons(mask):
        if mask.sum() == 0:
            return []
        try:
            import cv2
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            out = []
            for c in contours:
                if len(c) < 3:
                    continue
                out.append(c.reshape(-1, 2).astype(float))
            return out
        except ImportError:
            pass
        try:
            from osgeo import gdal, ogr
        except ImportError:
            return []
        h, w = mask.shape
        mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
        mem.GetRasterBand(1).WriteArray(mask.astype(np.uint8))
        band = mem.GetRasterBand(1)
        drv = ogr.GetDriverByName("Memory") or ogr.GetDriverByName("MEM")
        ogr_ds = drv.CreateDataSource("")
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
            if ring is None or ring.GetPointCount() < 3:
                continue
            out.append(np.array(
                [(ring.GetX(i), ring.GetY(i)) for i in range(ring.GetPointCount())],
                dtype=float))
        return out

    @staticmethod
    def _parse_label_line(line, tile_w, tile_h, mode):
        vals = line.split()
        if len(vals) < 5:
            return None
        try:
            cls = int(float(vals[0]))
            coords = [float(v) for v in vals[1:]]
        except ValueError:
            return None
        if len(vals) == 5:
            cx, cy, bw, bh = coords
            x0 = (cx - bw / 2.0) * tile_w
            y0 = (cy - bh / 2.0) * tile_h
            x1 = (cx + bw / 2.0) * tile_w
            y1 = (cy + bh / 2.0) * tile_h
            return cls, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                                 dtype=float)
        if len(vals) >= 7 and len(coords) % 2 == 0:
            pts = np.array(coords, dtype=float).reshape(-1, 2)
            pts[:, 0] *= tile_w
            pts[:, 1] *= tile_h
            if mode == GreenhouseGeorefTestAlgorithm.MODE_BBOXES:
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                pts = np.array([[x_min, y_min], [x_max, y_min],
                                [x_max, y_max], [x_min, y_max]], dtype=float)
            return cls, pts
        return None

    @staticmethod
    def _check_disk_space(folder, needed_bytes, what, feedback):
        try:
            free = shutil.disk_usage(str(folder)).free
        except OSError:
            return
        if needed_bytes > free:
            raise QgsProcessingException(
                f"Not enough disk space for {what}: needs about "
                f"{needed_bytes / 1e9:.1f} GB, only {free / 1e9:.1f} GB free.")

    def _render_layer_to_geotiff(self, raster_layer, extent, pixel_size,
                                 target_crs, output_path, max_megapixels,
                                 feedback):
        provider = raster_layer.dataProvider()
        if provider is None:
            raise QgsProcessingException("Input layer has no data provider.")
        width_px = max(1, int(round(extent.width() / pixel_size)))
        height_px = max(1, int(round(extent.height() / pixel_size)))
        mpix = width_px * height_px / 1e6
        feedback.pushInfo(f"Render size: {width_px} x {height_px} px ({mpix:.0f} MP)")
        if max_megapixels > 0 and mpix > max_megapixels:
            suggested = pixel_size * math.sqrt(mpix / max_megapixels)
            raise QgsProcessingException(
                f"The rendered raster would be {mpix:.0f} MP, above the limit "
                f"of {max_megapixels:.0f} MP. Shrink the extent or use a pixel "
                f"size of at least {suggested:.3g}.")
        self._check_disk_space(Path(output_path).parent,
                               int(width_px * height_px * 3 * 1.2),
                               "the rendered raster", feedback)
        pipe = QgsRasterPipe()
        if not pipe.set(provider.clone()):
            raise QgsProcessingException("Failed to build a raster pipe.")
        writer = QgsRasterFileWriter(output_path)
        writer.setOutputFormat("GTiff")
        setter = getattr(writer, "setCreationOptions", None) or \
            getattr(writer, "setCreateOptions", None)
        if setter is not None:
            setter(["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"])
        block_fb = QgsRasterBlockFeedback()
        try:
            feedback.canceled.connect(block_fb.cancel)
        except Exception:
            pass
        err = writer.writeRaster(pipe, width_px, height_px, extent, target_crs,
                                 QgsProject.instance().transformContext(),
                                 block_fb)
        if feedback.isCanceled():
            raise QgsProcessingException("Canceled while rendering.")
        if err != 0:
            raise QgsProcessingException(
                f"Rendering the layer to GeoTIFF failed (error code {err}).")

    # ---- helpers specific to the test ----------------------------------------

    @staticmethod
    def _tile_rgb(ds, col, row, w, h):
        """Read a tile and normalize it exactly like the detector does."""
        arr = ds.ReadAsArray(col, row, w, h)
        if arr is None:
            return None
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[2] >= 3:
                arr = arr[:, :, :3]
            elif arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            else:
                pad = np.zeros_like(arr[..., :1])
                arr = np.concatenate([arr, pad], axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _rasterize_piece(piece, col0, row0, w, h):
        """Burn a pixel-space polygon into a (h, w) tile mask (pixel-centre
        rule, holes respected)."""
        from osgeo import gdal, ogr
        mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
        mem.SetGeoTransform((float(col0), 1.0, 0.0, float(row0), 0.0, 1.0))
        drv = ogr.GetDriverByName("Memory") or ogr.GetDriverByName("MEM")
        vds = drv.CreateDataSource("")
        lyr = vds.CreateLayer("p", srs=None, geom_type=ogr.wkbUnknown)
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetGeometry(ogr.CreateGeometryFromWkt(piece.asWkt()))
        lyr.CreateFeature(f)
        gdal.RasterizeLayer(mem, [1], lyr, burn_values=[1])
        return mem.GetRasterBand(1).ReadAsArray().astype(np.uint8)

    @staticmethod
    def _safe_area(g):
        return g.area() if g is not None and not g.isEmpty() else 0.0

    @staticmethod
    def _valid(g):
        if g is None or g.isEmpty():
            return g
        if not g.isGeosValid():
            g = g.makeValid()
        return g

    # ---- main -----------------------------------------------------------------

    def processAlgorithm(self, parameters, context, feedback):
        try:
            from PIL import Image, ImageDraw
            from osgeo import gdal
        except ImportError as e:
            raise QgsProcessingException(f"Missing dependency: {e}")

        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        if raster_layer is None:
            raise QgsProcessingException("Invalid input raster.")
        source = self.parameterAsSource(parameters, self.INPUT_POLYGONS, context)
        if source is None:
            raise QgsProcessingException("Invalid polygon layer.")

        route = self.parameterAsEnum(parameters, self.ROUTE, context)
        tile_size = self.parameterAsInt(parameters, self.TILE_SIZE, context)
        overlap = self.parameterAsInt(parameters, self.OVERLAP, context)
        pixel_size = self.parameterAsDouble(parameters, self.PIXEL_SIZE, context)
        max_mp = self.parameterAsDouble(parameters, self.MAX_MEGAPIXELS, context)
        out_crs = self.parameterAsCrs(parameters, self.OUTPUT_CRS, context)
        skip_empty = self.parameterAsBool(parameters, self.SKIP_EMPTY, context)
        half_px = self.parameterAsBool(parameters, self.HALF_PIXEL, context)
        iou_pass = self.parameterAsDouble(parameters, self.IOU_PASS, context)
        write_debug = self.parameterAsBool(parameters, self.WRITE_DEBUG, context)

        step = tile_size - overlap
        if step <= 0:
            raise QgsProcessingException("Overlap must be smaller than tile size.")

        work_param = self.parameterAsString(parameters, self.WORK_FOLDER, context)
        if work_param:
            work_dir = Path(work_param)
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="georef_test_"))
        feedback.pushInfo(f"Work folder: {work_dir}")
        feedback.pushInfo("Route: " + ("exact vertices" if route == self.ROUTE_EXACT
                                       else "rasterized mask → contours"
                                       + (" (+0.5 px)" if half_px else "")))

        # -- Read the test polygons -------------------------------------------
        gt_crs = source.sourceCrs()
        gt_fids, gt_geoms = [], []
        for f in source.getFeatures():
            if not f.hasGeometry():
                continue
            g = self._valid(QgsGeometry(f.geometry()))
            if g is None or g.isEmpty():
                continue
            gt_fids.append(f.id())
            gt_geoms.append(g)
        if not gt_geoms:
            raise QgsProcessingException(
                "No polygons to test (check 'Selected features only').")
        feedback.pushInfo(f"{len(gt_geoms)} polygon feature(s) "
                          f"[{gt_crs.authid()}] to test.")

        # -- Resolve the raster exactly like the detector ----------------------
        layer_crs = raster_layer.crs() if raster_layer.crs().isValid() \
            else QgsCoordinateReferenceSystem("EPSG:3857")
        provider_name = raster_layer.dataProvider().name() \
            if raster_layer.dataProvider() else ""

        extent = None
        if parameters.get(self.EXTENT):
            extent = self.parameterAsExtent(parameters, self.EXTENT, context, layer_crs)

        needs_render = provider_name != "gdal"
        if not needs_render and gdal.Open(raster_layer.source()) is None:
            needs_render = True
        if extent is not None and not extent.isEmpty():
            needs_render = True

        if needs_render:
            if extent is None or extent.isEmpty():
                # No extent given → bbox of the test polygons (+ margin).
                to_layer = QgsCoordinateTransform(gt_crs, layer_crs, QgsProject.instance())
                bbox = None
                for g in gt_geoms:
                    gg = QgsGeometry(g)
                    gg.transform(to_layer)
                    b = gg.boundingBox()
                    if bbox is None:
                        bbox = QgsRectangle(b)
                    else:
                        bbox.combineExtentWith(b)
                if pixel_size > 0:
                    bbox.grow(pixel_size * max(overlap, 32))
                extent = bbox
                feedback.pushWarning(
                    "No processing extent given — using the polygons' bbox. "
                    "The tile grid then differs from your detection run; give "
                    "the same extent / AOI bbox to reproduce it exactly.")
            if pixel_size <= 0:
                raise QgsProcessingException("'Pixel size' must be > 0 to render.")
            raster_path = str(work_dir / "input_render.tif")
            feedback.pushInfo(f"Rendering layer to {raster_path} ...")
            self._render_layer_to_geotiff(raster_layer, extent, pixel_size,
                                          layer_crs, raster_path, max_mp, feedback)
        else:
            raster_path = raster_layer.source()

        ds = gdal.Open(raster_path)
        if ds is None:
            raise QgsProcessingException(f"GDAL could not open {raster_path}")
        gt = ds.GetGeoTransform()
        if gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            gcps = ds.GetGCPs()
            if not gcps:
                raise QgsProcessingException("Raster has no georeferencing.")
            gt = gdal.GCPsToGeoTransform(gcps)
            feedback.pushInfo("Using geotransform derived from GCPs.")
        src_crs = raster_layer.crs()
        if not src_crs.isValid():
            raise QgsProcessingException("Input raster has no valid CRS.")
        width, height = ds.RasterXSize, ds.RasterYSize
        feedback.pushInfo(f"Raster {src_crs.authid()} {width} x {height}, "
                          f"geotransform {tuple(round(v, 6) for v in gt)}")

        inv = gdal.InvGeoTransform(gt)
        if inv is not None and len(inv) == 2:      # GDAL 2 returns (ok, gt)
            ok, inv = inv
            inv = inv if ok else None
        if inv is None:
            raise QgsProcessingException("Geotransform is not invertible.")
        # map -> pixel:  col = c0 + c1*x + c2*y ; row = r0 + r1*x + r2*y
        c0, c1, c2, r0, r1, r2 = inv
        to_pixel = QTransform(c1, r1, c2, r2, c0, r0)

        # -- Polygons in pixel space and in output CRS (independent path) -----
        to_src = QgsCoordinateTransform(gt_crs, src_crs, QgsProject.instance())
        to_out_direct = QgsCoordinateTransform(gt_crs, out_crs, QgsProject.instance())
        gt_px, gt_out = [], []
        index = QgsSpatialIndex()
        for i, g in enumerate(gt_geoms):
            gp = QgsGeometry(g)
            gp.transform(to_src)
            gp.transform(to_pixel)
            gp = self._valid(gp)
            gt_px.append(gp)
            go = QgsGeometry(g)
            go.transform(to_out_direct)
            gt_out.append(self._valid(go))
            feat = QgsFeature(i)
            feat.setGeometry(gp)
            index.addFeature(feat)

        # -- Tile with the detector's grid, write YOLO labels ------------------
        labels_dir = work_dir / "labels"
        if labels_dir.exists():
            for old in labels_dir.glob("*.txt"):
                old.unlink()
        labels_dir.mkdir(parents=True, exist_ok=True)
        debug_dir = work_dir / "debug_tiles"
        if write_debug:
            debug_dir.mkdir(exist_ok=True)
        index_file = work_dir / "tile_index.csv"

        n_rows = (height + step - 1) // step
        n_cols = (width + step - 1) // step
        total_tiles = max(1, n_rows * n_cols)
        feedback.pushInfo(f"Tile grid: {n_cols} x {n_rows} = {total_tiles} tiles "
                          f"({tile_size}px, overlap {overlap}px).")

        n_tiles_used = n_empty = holes_dropped = n_lines = 0
        lost_in_empty = set()
        done = 0
        with open(index_file, "w", newline="") as fidx:
            writer = csv.writer(fidx)
            writer.writerow(["tile_name", "col_off", "row_off", "width", "height"])
            for row in range(0, height, step):
                if feedback.isCanceled():
                    return {}
                for col in range(0, width, step):
                    done += 1
                    win_w = min(tile_size, width - col)
                    win_h = min(tile_size, height - row)
                    if win_w <= 0 or win_h <= 0:
                        continue
                    rect = QgsRectangle(col, row, col + win_w, row + win_h)
                    hits = index.intersects(rect)
                    if not hits:
                        continue

                    rgb = None
                    if skip_empty or write_debug:
                        rgb = self._tile_rgb(ds, col, row, win_w, win_h)
                        if skip_empty and (rgb is None or not rgb.any()):
                            n_empty += 1
                            lost_in_empty.update(hits)
                            continue

                    tile_name = f"tile_{row}_{col}.png"
                    rect_geom = QgsGeometry.fromRect(rect)
                    lines, debug_polys = [], []
                    for i in sorted(hits):
                        piece = gt_px[i].intersection(rect_geom)
                        piece = self._to_multipolygon(self._valid(piece))
                        if piece is None:
                            continue

                        polys = []  # (N,2) arrays in tile pixel coords
                        if route == self.ROUTE_EXACT:
                            for poly in piece.asMultiPolygon():
                                if not poly:
                                    continue
                                holes_dropped += len(poly) - 1  # YOLO lines have no holes
                                ring = np.array([[p.x(), p.y()] for p in poly[0]], dtype=float)
                                if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
                                    ring = ring[:-1]
                                ring[:, 0] -= col
                                ring[:, 1] -= row
                                polys.append(ring)
                        else:
                            mask = self._rasterize_piece(piece, col, row, win_w, win_h)
                            for ring in self._mask_to_polygons(mask):
                                if half_px:
                                    ring = ring + 0.5
                                polys.append(ring)

                        for ring in polys:
                            if len(ring) < 3:
                                continue
                            n = ring.copy()
                            n[:, 0] /= float(win_w)
                            n[:, 1] /= float(win_h)
                            # class column = test-feature index, so we can
                            # match placed pieces back to their feature
                            lines.append(f"{i} " + " ".join(f"{v:.6f}" for v in n.flatten()))
                            debug_polys.append(ring)

                    writer.writerow([tile_name, col, row, win_w, win_h])
                    if lines:
                        (labels_dir / (tile_name[:-4] + ".txt")).write_text("\n".join(lines))
                        n_tiles_used += 1
                        n_lines += len(lines)
                        if write_debug and rgb is not None:
                            img = Image.fromarray(rgb)
                            draw = ImageDraw.Draw(img)
                            for ring in debug_polys:
                                pts = [(float(x), float(y)) for x, y in ring]
                                draw.line(pts + [pts[0]], fill=(255, 0, 0), width=2)
                            img.save(debug_dir / tile_name)
                if total_tiles:
                    feedback.setProgress(int(done / total_tiles * 50))

        feedback.pushInfo(f"{n_lines} label lines on {n_tiles_used} tiles.")
        if n_empty:
            feedback.pushWarning(
                f"{n_empty} tiles skipped as all-black (the detector skips "
                "them too) — polygons there cannot be placed.")
        if holes_dropped:
            feedback.pushInfo(
                f"{holes_dropped} polygon hole(s) dropped — YOLO polygon "
                "lines carry outer rings only (same as the detector).")

        # -- Place labels with the detector's code ------------------------------
        tile_index = {}
        with open(index_file, newline="") as f:
            for r in csv.DictReader(f):
                tile_index[r["tile_name"]] = {
                    "col_off": int(r["col_off"]), "row_off": int(r["row_off"]),
                    "width": int(r["width"]), "height": int(r["height"])}

        xform = QgsCoordinateTransform(src_crs, out_crs, QgsProject.instance())
        placed = {}        # feature index -> list of geometries
        placed_tiles = {}  # feature index -> set of tile names
        for lf in sorted(labels_dir.glob("*.txt")):
            if feedback.isCanceled():
                return {}
            tile_name = lf.stem + ".png"
            if tile_name not in tile_index:
                continue
            meta = tile_index[tile_name]
            w, h = meta["width"], meta["height"]
            col0, row0 = meta["col_off"], meta["row_off"]
            for line in lf.read_text().strip().splitlines():
                parsed = self._parse_label_line(line, w, h, self.MODE_POLYGONS)
                if parsed is None:
                    continue
                cls, pts = parsed
                # ---- identical to the detector -------------------------
                px_col = pts[:, 0] + col0
                px_row = pts[:, 1] + row0
                x = gt[0] + px_col * gt[1] + px_row * gt[2]
                y = gt[3] + px_col * gt[4] + px_row * gt[5]
                qgs_points = []
                for xi, yi in zip(x, y):
                    try:
                        qgs_points.append(xform.transform(QgsPointXY(float(xi), float(yi))))
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
                mp = self._to_multipolygon(geom)
                if mp is None or mp.area() <= 0:
                    continue
                # ---------------------------------------------------------
                placed.setdefault(cls, []).append(mp)
                placed_tiles.setdefault(cls, set()).add(tile_name)
        feedback.setProgress(70)

        # -- Raster footprint in output CRS (densified edges) ------------------
        ring = []
        nd = 50
        for k in range(nd):
            ring.append((width * k / nd, 0.0))
        for k in range(nd):
            ring.append((float(width), height * k / nd))
        for k in range(nd):
            ring.append((width * (1 - k / nd), float(height)))
        for k in range(nd):
            ring.append((0.0, height * (1 - k / nd)))
        fp_pts = []
        for c, r in ring:
            mx = gt[0] + c * gt[1] + r * gt[2]
            my = gt[3] + c * gt[4] + r * gt[5]
            fp_pts.append(xform.transform(QgsPointXY(mx, my)))
        footprint = self._valid(QgsGeometry.fromPolygonXY([fp_pts]))

        # Pixel size expressed in output CRS units (at the raster centre)
        cc, cr = width / 2.0, height / 2.0
        pa = xform.transform(QgsPointXY(gt[0] + cc * gt[1] + cr * gt[2],
                                        gt[3] + cc * gt[4] + cr * gt[5]))
        pb = xform.transform(QgsPointXY(gt[0] + (cc + 1) * gt[1] + cr * gt[2],
                                        gt[3] + (cc + 1) * gt[4] + cr * gt[5]))
        px_out = math.hypot(pb.x() - pa.x(), pb.y() - pa.y()) or 1.0
        feedback.pushInfo(f"One raster pixel ≈ {px_out:.4g} {out_crs.authid()} units.")

        # -- Metrics + outputs ---------------------------------------------------
        fields = QgsFields()
        for name, t in [("src_fid", QVariant.LongLong), ("iou", QVariant.Double),
                        ("iou_in", QVariant.Double), ("coverage", QVariant.Double),
                        ("area_gt", QVariant.Double), ("area_pred", QVariant.Double),
                        ("dx", QVariant.Double), ("dy", QVariant.Double),
                        ("shift", QVariant.Double), ("shift_px", QVariant.Double),
                        ("hausdorff", QVariant.Double), ("n_pieces", QVariant.Int),
                        ("n_tiles", QVariant.Int), ("passed", QVariant.Int),
                        ("note", QVariant.String)]:
            fields.append(QgsField(name, t))
        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context,
                                             fields, QgsWkbTypes.MultiPolygon, out_crs)
        if sink is None:
            raise QgsProcessingException("Could not create output layer.")

        dfields = QgsFields()
        dfields.append(QgsField("src_fid", QVariant.LongLong))
        dfields.append(QgsField("kind", QVariant.String))
        dfields.append(QgsField("area", QVariant.Double))
        dsink, ddest = self.parameterAsSink(parameters, self.OUTPUT_DIFF, context,
                                            dfields, QgsWkbTypes.MultiPolygon, out_crs)

        report_path = work_dir / "georef_report.csv"
        rows = []
        ious, shifts_px, dxs_px, dys_px = [], [], [], []
        all_pred, all_gt_in = [], []
        n_pass = 0
        n = len(gt_geoms)
        for i in range(n):
            if feedback.isCanceled():
                break
            g_ref = gt_out[i]
            a_gt = self._safe_area(g_ref)
            g_in = self._valid(g_ref.intersection(footprint)) if footprint else g_ref
            a_in = self._safe_area(g_in)
            coverage = a_in / a_gt if a_gt > 0 else 0.0
            if a_in > 0:
                all_gt_in.append(g_in)

            pieces = placed.get(i, [])
            note = ""
            pred = None
            if pieces:
                pred = self._valid(QgsGeometry.unaryUnion(pieces))
                pred = self._to_multipolygon(pred)
            a_pred = self._safe_area(pred)

            if pred is None or a_pred <= 0:
                iou = iou_in = 0.0
                dx = dy = shift = shift_px = haus = None
                if i in lost_in_empty:
                    note = "lost: only on all-black tiles"
                elif coverage <= 0:
                    note = "outside raster"
                else:
                    note = "not placed"
            else:
                all_pred.append(pred)
                inter = self._safe_area(pred.intersection(g_ref))
                uni = self._safe_area(pred.combine(g_ref))
                iou = inter / uni if uni > 0 else 0.0
                if a_in > 0:
                    inter_in = self._safe_area(pred.intersection(g_in))
                    uni_in = self._safe_area(pred.combine(g_in))
                    iou_in = inter_in / uni_in if uni_in > 0 else 0.0
                    cp = pred.centroid().asPoint()
                    cg = g_in.centroid().asPoint()
                    dx, dy = cp.x() - cg.x(), cp.y() - cg.y()
                    shift = math.hypot(dx, dy)
                    shift_px = shift / px_out
                    haus = pred.hausdorffDistance(g_in)
                    ious.append(iou_in)
                    shifts_px.append(shift_px)
                    dxs_px.append(dx / px_out)
                    dys_px.append(dy / px_out)
                else:
                    iou_in = 0.0
                    dx = dy = shift = shift_px = haus = None
                if coverage < 0.999:
                    note = f"partly outside raster ({coverage:.0%} covered)"

            passed = int(iou_in >= iou_pass)
            n_pass += passed
            if (pred is None or a_pred <= 0) and coverage > 0:
                ious.append(0.0)   # inside the raster but never placed

            feat = QgsFeature(fields)
            if pred is not None and a_pred > 0:
                feat.setGeometry(pred)
            feat.setAttributes([
                int(gt_fids[i]), float(iou), float(iou_in), float(coverage),
                float(a_gt), float(a_pred), dx, dy, shift, shift_px, haus,
                len(pieces), len(placed_tiles.get(i, ())), passed, note])
            sink.addFeature(feat, QgsFeatureSink.FastInsert)

            if dsink is not None and a_in > 0:
                for kind, dg in (
                    ("missed", g_in.difference(pred) if pred is not None else g_in),
                    ("extra", pred.difference(g_in) if pred is not None else None),
                ):
                    dg = self._to_multipolygon(self._valid(dg)) if dg is not None else None
                    if dg is None or dg.area() <= 0:
                        continue
                    df = QgsFeature(dfields)
                    df.setGeometry(dg)
                    df.setAttributes([int(gt_fids[i]), kind, float(dg.area())])
                    dsink.addFeature(df, QgsFeatureSink.FastInsert)

            rows.append([gt_fids[i], f"{iou:.6f}", f"{iou_in:.6f}", f"{coverage:.4f}",
                         f"{a_gt:.4f}", f"{a_pred:.4f}",
                         "" if dx is None else f"{dx:.6g}",
                         "" if dy is None else f"{dy:.6g}",
                         "" if shift_px is None else f"{shift_px:.4f}",
                         "" if haus is None else f"{haus:.6g}",
                         len(pieces), len(placed_tiles.get(i, ())), passed, note])
            feedback.setProgress(70 + int((i + 1) / n * 28))

        with open(report_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["src_fid", "iou", "iou_in", "coverage", "area_gt",
                        "area_pred", "dx", "dy", "shift_px", "hausdorff",
                        "n_pieces", "n_tiles", "passed", "note"])
            w.writerows(rows)

        # Global IoU: everything placed vs everything inside the raster
        global_iou = 0.0
        if all_pred and all_gt_in:
            up = self._valid(QgsGeometry.unaryUnion(all_pred))
            ug = self._valid(QgsGeometry.unaryUnion(all_gt_in))
            inter = self._safe_area(up.intersection(ug))
            uni = self._safe_area(up.combine(ug))
            global_iou = inter / uni if uni > 0 else 0.0

        mean_iou = statistics.fmean(ious) if ious else 0.0
        median_iou = statistics.median(ious) if ious else 0.0
        mean_shift = statistics.fmean(shifts_px) if shifts_px else 0.0
        mdx = statistics.fmean(dxs_px) if dxs_px else 0.0
        mdy = statistics.fmean(dys_px) if dys_px else 0.0

        feedback.pushInfo("=" * 60)
        feedback.pushInfo(f"Features tested      : {n}")
        feedback.pushInfo(f"Passing (IoU ≥ {iou_pass}) : {n_pass}")
        feedback.pushInfo(f"Mean / median / min IoU : {mean_iou:.4f} / "
                          f"{median_iou:.4f} / {min(ious) if ious else 0:.4f}")
        feedback.pushInfo(f"Global IoU           : {global_iou:.4f}")
        feedback.pushInfo(f"Mean centroid shift  : {mean_shift:.3f} px "
                          f"(mean dx {mdx:+.3f} px, dy {mdy:+.3f} px, map axes)")
        feedback.pushInfo(f"Report               : {report_path}")
        feedback.pushInfo("=" * 60)

        if abs(mdx) > 0.25 or abs(mdy) > 0.25:
            feedback.pushWarning(
                f"Systematic shift of ({mdx:+.2f}, {mdy:+.2f}) px. "
                + ("≈ half a pixel with the mask route: contour vertices are "
                   "pixel indices (centres) but the geotransform maps them as "
                   "pixel corners. Re-run with the +0.5 px option to confirm."
                   if route == self.ROUTE_MASK and not half_px else
                   "Check the geotransform / CRS / rendered extent."))
        elif route == self.ROUTE_EXACT and mean_iou < 0.99:
            feedback.pushWarning(
                "Exact route should give IoU ≈ 1. Look at low-IoU features "
                "and the debug tiles — likely a CRS or tiling offset.")

        feedback.setProgress(100)
        ds = None
        return {
            self.OUTPUT: dest_id,
            self.OUTPUT_DIFF: ddest,
            "MEAN_IOU": mean_iou,
            "MEDIAN_IOU": median_iou,
            "GLOBAL_IOU": global_iou,
            "MEAN_SHIFT_PX": mean_shift,
            "N_PASS": n_pass,
            "REPORT": str(report_path),
        }
