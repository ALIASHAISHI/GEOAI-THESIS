# dataset_to_yolodataset.py — Labelme → YOLO dataset (detect + segment + masks)

Converts a labelme dataset (images paired with `.json` annotations) into a YOLO-ready dataset containing, for every image:

- a **detection** label (bounding box per object),
- a **segmentation** label (polygon per object),
- a **rasterised mask** PNG,

split into `train` / `val` (and optionally `test`), plus two dataset YAML files.

---

## Requirements

```bash
pip install pillow tqdm
```

---

## Usage

```bash
python dataset_to_yolodataset.py --src <labelme folder> --dst <output folder> \
                                 --val 0.2 --test 0.0 --seed 42
```

Running it with no arguments uses the thesis paths set as defaults.

### Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--src` | `...\Greenhousedataset\images` | Folder with labelme pairs (searched recursively for `*.json`). |
| `--dst` | `...\clean_yolo_dataset` | Output dataset folder. |
| `--val` | `0.2` | Validation fraction. |
| `--test` | `0.0` | Test fraction (0 = no test split). |
| `--seed` | `42` | Random seed for the shuffle / split. |
| `--classes` | `1` | Fixed class name order. Pass with no values to auto-discover sorted labels from the JSON files. |
| `--mask-mode` | `class` | `binary`: masks are 0 / 255. `class`: 0 = background, `class_id + 1` per object. |

---

## Output layout

```
dst/
├── data_detect.yaml
├── data_segment.yaml
├── images/{train,val[,test]}/          ← copied images
├── labels_detect/{train,val[,test]}/   ← YOLO bbox .txt
├── labels_segment/{train,val[,test]}/  ← YOLO polygon .txt
└── masks/{train,val[,test]}/           ← PNG masks
```

### Label formats (all coordinates normalised to 0–1)

**Detect** — one line per object:
```
<class> <cx> <cy> <w> <h>
```

**Segment** — one line per object:
```
<class> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

Images with no shapes get an empty `.txt` file (a valid "background" image for YOLO).

### YAML files

```yaml
path: /abs/path/to/dst
train: images/train
val: images/val
labels: labels_detect        # or labels_segment
nc: 1
names: ['1']
```

`data_segment.yaml` also lists `masks_train`, `masks_val` (and `masks_test`).

---

## How it works

1. **Discover pairs** — every `*.json` under `--src` is read; the image is located from `imagePath`, or by trying the same stem with each image extension. JSONs without an image are skipped with a message.
2. **Collect classes** — from `--classes`, or from all shape labels (sorted).
3. **Split** — pairs are shuffled with `--seed`, then divided into train / val / test. At least one sample always goes to train.
4. **Per sample**:
   - image size from `imageWidth` / `imageHeight` in the JSON, or read from the image;
   - only `polygon` and `rectangle` shapes are used (rectangles become 4-point polygons); circles, lines, points etc. are ignored; polygons with < 3 points are dropped;
   - coordinates are clamped to the image;
   - the image is copied, the detect and segment `.txt` files are written, and the mask is rasterised with PIL.
5. **Write YAMLs** and print a summary (class list, split sizes, output root).

---

## Notes and caveats

- **Single class in the labels.** `shapes_to_detect_lines` and `shapes_to_segment_lines` always write class id `0`, whatever the labelme label. That matches the single-class greenhouse setup; for multi-class, use `class_to_id[_label]` instead of `0`.
- **Default mask values are 0 / 1, not 0 / 255.** With the default `--mask-mode class` and one class, objects are drawn with value `class_id + 1 = 1`, so masks look black in an image viewer. Use `--mask-mode binary` for 0 / 255.
- **`labels:` in the YAML is not read by Ultralytics.** Ultralytics finds labels by replacing `images` with `labels` in the image path. To train, either rename the folder you want (`labels_detect` or `labels_segment`) to `labels`, or create a `labels` link/copy to it.
- **Overlapping polygons in masks** — later shapes overwrite earlier ones where they overlap.
- **Same stem, different folders** — images are copied by file name into one split folder, so two source images with the same name in different sub-folders would overwrite each other.
- The module docstring refers to the script as `labelme2yolo.py`; the usage is the same with this file name.
