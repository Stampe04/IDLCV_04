from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")  # Save figures without a display

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


# Paths
DATASET = Path("/dtu/datasets1/02516/potholes")
OUTPUT = Path(__file__).resolve().parent / "visualizations"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Images to visualize
EXAMPLES = [
    "potholes0",
    "potholes58",
    "potholes214",
    "potholes100",
]

# Create a 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, name in zip(axes.flat, EXAMPLES):
    image_path = DATASET / "images" / f"{name}.png"
    annotation_path = DATASET / "annotations" / f"{name}.xml"

    # Load image and annotation
    image = Image.open(image_path).convert("RGB")
    root = ET.parse(annotation_path).getroot()

    ax.imshow(image)

    valid_boxes = 0

    for obj in root.findall("object"):
        box = obj.find("bndbox")

        xmin = int(box.findtext("xmin"))
        ymin = int(box.findtext("ymin"))
        xmax = int(box.findtext("xmax"))
        ymax = int(box.findtext("ymax"))

        # Clip coordinates to image boundaries
        xmin = max(0, min(xmin, image.width))
        xmax = max(0, min(xmax, image.width))
        ymin = max(0, min(ymin, image.height))
        ymax = max(0, min(ymax, image.height))

        # Skip invalid boxes
        if xmax <= xmin or ymax <= ymin:
            continue

        # Draw bounding box
        rectangle = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
        )

        ax.add_patch(rectangle)
        valid_boxes += 1

    ax.set_title(f"{name} ({valid_boxes} potholes)")
    ax.axis("off")

# Save figure
plt.tight_layout()

save_path = OUTPUT / "ground_truth_examples.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Visualization saved to: {save_path}")