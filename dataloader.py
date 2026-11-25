from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import ResNet18_Weights

# Configuration
DATA_PATH = Path("/dtu/datasets1/02516/potholes")
CACHE_PATH = Path(__file__).resolve().parent / "proposal_cache"

NUM_PROPOSALS = 750
NEGATIVE_THRESHOLD = 0.3
POSITIVE_THRESHOLD = 0.5

# Preprocessing for pretrained ResNet-18
TRANSFORM = ResNet18_Weights.DEFAULT.transforms()

def load_annotations(image_id, width, height):
    
    # load bounding boxes
    xml_path = DATA_PATH / "annotations" / f"{image_id}.xml"
    root = ET.parse(xml_path).getroot()

    boxes = []

    for obj in root.findall("object"):
        box = obj.find("bndbox")

        x1 = int(box.findtext("xmin"))
        y1 = int(box.findtext("ymin"))
        x2 = int(box.findtext("xmax"))
        y2 = int(box.findtext("ymax"))

        # Clip coordinates to image boundaries
        x1 = np.clip(x1, 0, width)
        x2 = np.clip(x2, 0, width)
        y1 = np.clip(y1, 0, height)
        y2 = np.clip(y2, 0, height)

        # Ignore invalid bounding boxes
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2, y2])

    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

def compute_iou(box, gt_boxes):
    
    if len(gt_boxes) == 0:
        return np.array([])

    x1 = np.maximum(box[0], gt_boxes[:, 0])
    y1 = np.maximum(box[1], gt_boxes[:, 1])
    x2 = np.minimum(box[2], gt_boxes[:, 2])
    y2 = np.minimum(box[3], gt_boxes[:, 3])

    intersection = (
        np.maximum(0, x2 - x1) *
        np.maximum(0, y2 - y1)
    )

    area_box = (
        (box[2] - box[0]) *
        (box[3] - box[1])
    )

    area_gt = (
        (gt_boxes[:, 2] - gt_boxes[:, 0]) *
        (gt_boxes[:, 3] - gt_boxes[:, 1])
    )

    union = area_box + area_gt - intersection

    return intersection / np.maximum(union, 1e-8)

def assign_labels(proposals, gt_boxes):

    labels = np.full(len(proposals), -1, dtype=np.int64)

    for i, proposal in enumerate(proposals):
        overlaps = compute_iou(proposal, gt_boxes)
        max_iou = overlaps.max() if len(overlaps) else 0.0

        if max_iou >= POSITIVE_THRESHOLD:
            labels[i] = 1
        elif max_iou < NEGATIVE_THRESHOLD:
            labels[i] = 0

    return labels

def generate_proposals(image):

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()

    rects = np.asarray(ss.process(), dtype=np.int32).reshape(-1, 4)
    rects = rects[:NUM_PROPOSALS]

    # Convert (x, y, w, h) to (xmin, ymin, xmax, ymax)
    boxes = rects.copy()
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    height, width = image.shape[:2]

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)

    valid = (
        (boxes[:, 2] > boxes[:, 0]) &
        (boxes[:, 3] > boxes[:, 1])
    )

    return boxes[valid]

def prepare_proposals(image_ids):

    CACHE_PATH.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        cache_file = CACHE_PATH / f"{image_id}_ss{NUM_PROPOSALS}.npz"

        if cache_file.exists():
            continue

        image_path = DATA_PATH / "images" / f"{image_id}.png"
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        height, width = image.shape[:2]

        proposals = generate_proposals(image)
        gt_boxes = load_annotations(image_id, width, height)
        labels = assign_labels(proposals, gt_boxes)

        np.savez_compressed(
            cache_file,
            boxes=proposals,
            labels=labels
        )

        print(
            f"{image_id}: "
            f"{np.sum(labels == 1)} positive, "
            f"{np.sum(labels == 0)} background"
        )

class PotholeDataset(Dataset):

    def __init__(self, image_ids):
        self.samples = []
        self.transform = TRANSFORM

        for image_id in image_ids:
            cache_file = (
                CACHE_PATH / f"{image_id}_ss{NUM_PROPOSALS}.npz"
            )

            if not cache_file.exists():
                raise FileNotFoundError(
                    f"Missing cache for {image_id}. "
                    "Run prepare_proposals() first."
                )

            with np.load(cache_file) as data:
                boxes = data["boxes"]
                labels = data["labels"]

            for box, label in zip(boxes, labels):
                if label == -1:
                    continue

                self.samples.append((
                    image_id,
                    box.astype(int).tolist(),
                    int(label)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, box, label = self.samples[idx]

        image_path = DATA_PATH / "images" / f"{image_id}.png"

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            crop = image.crop(tuple(box))

        crop = self.transform(crop)

        return crop, torch.tensor(label, dtype=torch.long)

def create_dataloader(dataset, batch_size=64, training=True):

    if training:
        labels = np.array(
            [sample[2] for sample in dataset.samples],
            dtype=np.int64
        )

        counts = np.bincount(labels, minlength=2)

        if np.any(counts == 0):
            raise ValueError(
                "Training dataset must contain both classes.")

        # Target 75% background, 25% pothole
        weights = np.array([0.75, 0.25]) / counts
        sample_weights = weights[labels]

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(
                sample_weights, dtype=torch.double
            ),
            num_samples=len(dataset),
            replacement=True)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=torch.cuda.is_available())

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available())