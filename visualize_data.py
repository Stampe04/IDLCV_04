from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# Paths
DATASET = Path("/dtu/datasets1/02516/potholes")
OUTPUT = Path(__file__).resolve().parent / "visualizations"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Images to inspect
EXAMPLES = [
    "potholes0",
    "potholes58",
    "potholes214",
    "potholes100",
]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, name in zip(axes.flat, EXAMPLES):
    image_path = DATASET / "images" / f"{name}.png"
    annotation_path = DATASET / "annotations" / f"{name}.xml"

    # Read original image and annotations
    image = Image.open(image_path)
    root = ET.parse(annotation_path).getroot()

    ax.imshow(image)

    # Display all original bounding boxes
    objects = root.findall("object")

    for obj in objects:
        box = obj.find("bndbox")

        xmin = int(box.findtext("xmin"))
        ymin = int(box.findtext("ymin"))
        xmax = int(box.findtext("xmax"))
        ymax = int(box.findtext("ymax"))

        rectangle = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rectangle)

    ax.set_title(f"{name} ({len(objects)} annotations)")
    ax.axis("off")

plt.tight_layout()

# Save visualization only
save_path = OUTPUT / "ground_truth_examples.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Visualization saved to: {save_path}")