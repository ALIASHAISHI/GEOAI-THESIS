import os
import json
import glob

# ⚙️ USER SETTINGS
root_dir = r"/media/mrizk/DiskB1/GH"  # 🔸 Change this to your dataset root
classes = {}  # will fill automatically from JSON labels

def get_class_id(label_name):
    """Map label name to integer ID."""
    if label_name not in classes:
        classes[label_name] = len(classes)
    return classes[label_name]

def convert_labelme_to_yolo(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_h = data.get("imageHeight", None)
    image_w = data.get("imageWidth", None)
    if not image_h or not image_w:
        print(f"⚠️ Skipping {json_path} (missing image size)")
        return

    yolo_lines = []
    for shape in data.get("shapes", []):
        if shape["shape_type"] != "polygon":
            continue

        label = shape["label"]
        points = shape["points"]
        class_id = get_class_id(label)

        # Flatten and normalize polygon coordinates
        coords = []
        for x, y in points:
            coords.append(x / image_w)
            coords.append(y / image_h)

        line = f"{class_id} " + " ".join([f"{c:.6f}" for c in coords])
        yolo_lines.append(line)

    # Save YOLO .txt label next to the image
    txt_path = os.path.splitext(json_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))
    print(f"✅ Converted: {os.path.basename(json_path)}")

def process_all_jsons():
    json_files = glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True)
    print(f"🔍 Found {len(json_files)} JSON files across subfolders.\n")

    for json_file in json_files:
        convert_labelme_to_yolo(json_file)

    # Save classes.txt at the root folder
    class_file = os.path.join(root_dir, "classes.txt")
    with open(class_file, "w", encoding="utf-8") as f:
        for cls_name in classes.keys():
            f.write(cls_name + "\n")

    print("\n🎯 Conversion done!")
    print(f"📂 classes.txt saved at: {class_file}")
    print(f"📚 Classes found: {classes}")

if __name__ == "__main__":
    process_all_jsons()

