"""
Display a raster with its labelme polygon annotations overlaid.
Usage:  python show_polygons.py image.png image.json
"""
import json
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def show(image_path, json_path):
    img = mpimg.imread(image_path)
    with open(json_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)

    # one color per unique label
    labels = sorted({s["label"] for s in data["shapes"]})
    cmap = plt.get_cmap("tab10")
    colors = {lab: cmap(i % 10) for i, lab in enumerate(labels)}

    for shp in data["shapes"]:
        if shp["shape_type"] != "polygon":
            continue
        color = colors[shp["label"]]
        poly = Polygon(shp["points"], closed=True,
                       facecolor=color, edgecolor=color,
                       alpha=0.35, linewidth=2)
        ax.add_patch(poly)
        # label near the polygon's first vertex
        x, y = shp["points"][0]
        ax.text(x, y - 4, shp["label"], color="white", fontsize=9,
                bbox=dict(facecolor=color, edgecolor="none",
                          alpha=0.8, pad=1.5))

    ax.set_title(f"{image_path}  —  {len(data['shapes'])} polygons")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show(
        r"C:\Users\alias\Documents\UNI\Thesis\GreenHouse-Dataset_Images+jsonlabels+masks\Greenhousedataset\images\2.png", 
        r"C:\Users\alias\Documents\UNI\Thesis\GreenHouse-Dataset_Images+jsonlabels+masks\Greenhousedataset\images\2.json")