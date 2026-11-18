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

# Choose 6 examples
# You can change these later if you find better-looking report examples
EXAMPLES = [
    "potholes0",
    "potholes58",
    "potholes100",
    "potholes214",
    "potholes231",
    "potholes277",
]

# 3 rows x 2 columns
fig, axes = plt.subplots(3, 2, figsize=(8, 12))
axes = axes.flatten()

for ax, name in zip(axes, EXAMPLES):
    image_path = DATASET / "images" / f"{name}.png"
    annotation_path = DATASET / "annotations" / f"{name}.xml"

    # Load original image
    image = Image.open(image_path).convert("RGB")
    root = ET.parse(annotation_path).getroot()

    ax.imshow(image)

    objects = root.findall("object")

    for obj in objects:
        box = obj.find("bndbox")

        xmin = int(box.findtext("xmin"))
        ymin = int(box.findtext("ymin"))
        xmax = int(box.findtext("xmax"))
        ymax = int(box.findtext("ymax"))

        # Draw annotation exactly as given
        rect = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=1.2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

    ax.set_title(f"{name} ({len(objects)} potholes)", fontsize=10)
    ax.axis("off")

plt.tight_layout()

save_path = OUTPUT / "ground_truth_examples_6.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Visualization saved to: {save_path}")