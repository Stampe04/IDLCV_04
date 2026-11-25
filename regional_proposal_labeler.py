
from pathlib import Path

import cv2
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from datasets import potholes, read_content
from region_proposal_labeler import RegionProposalLabeler


# Configuration
CACHE_PATH = Path(__file__).resolve().parent / "proposal_cache"

NUM_PROPOSALS = 750
IOU_THRESHOLD = 0.5
BATCH_SIZE = 64

# Preprocessing for pretrained ResNet-18
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def generate_proposals(image):
    """Generate Selective Search proposals in xyxy format."""

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()

    # Select first 750 proposals
    rects = np.asarray(
        ss.process(), dtype=np.int32
    ).reshape(-1, 4)

    rects = rects[:NUM_PROPOSALS]

    # Convert (x, y, w, h) to (xmin, ymin, xmax, ymax)
    boxes = rects.copy()
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    height, width = image.shape[:2]

    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]], 0, width
    )
    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]], 0, height
    )

    # Remove invalid proposals
    valid = (
        (boxes[:, 2] > boxes[:, 0]) &
        (boxes[:, 3] > boxes[:, 1])
    )

    return boxes[valid]


def prepare_proposals(dataset, indices):
    """Generate and cache proposals for selected images."""

    CACHE_PATH.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        image_path = Path(dataset.images_path[idx])
        image_id = image_path.stem

        cache_file = (
            CACHE_PATH / f"{image_id}_ss{NUM_PROPOSALS}.npy"
        )

        if cache_file.exists():
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        proposals = generate_proposals(image)

        np.save(cache_file, proposals)

        print(f"{image_id}: {len(proposals)} proposals cached")


class ProposalDataset(Dataset):
    """Dataset of labeled region proposal crops."""

    def __init__(self, original_dataset, indices):
        self.samples = []
        self.transform = TRANSFORM

        labeler = RegionProposalLabeler(
            iou_threshold=IOU_THRESHOLD
        )

        for idx in indices:
            image_path = Path(original_dataset.images_path[idx])
            image_id = image_path.stem

            annotation_path = (
                Path(original_dataset.DATA_PATH) /
                "annotations" /
                f"{image_id}.xml"
            )

            # Reuse the group's XML parser
            _, gt_boxes = read_content(str(annotation_path))

            gt_boxes = np.asarray(
                gt_boxes, dtype=np.float32
            ).reshape(-1, 4)

            # Validate ground-truth boxes in memory
            with Image.open(image_path) as image:
                width, height = image.size

            gt_boxes[:, [0, 2]] = np.clip(
                gt_boxes[:, [0, 2]], 0, width
            )
            gt_boxes[:, [1, 3]] = np.clip(
                gt_boxes[:, [1, 3]], 0, height
            )

            valid_gt = (
                (gt_boxes[:, 2] > gt_boxes[:, 0]) &
                (gt_boxes[:, 3] > gt_boxes[:, 1])
            )
            gt_boxes = gt_boxes[valid_gt]

            # Load cached proposals
            cache_file = (
                CACHE_PATH / f"{image_id}_ss{NUM_PROPOSALS}.npy"
            )

            if not cache_file.exists():
                raise FileNotFoundError(
                    f"Missing proposals for {image_id}. "
                    "Run prepare_proposals() first."
                )

            proposals = np.load(cache_file)

            # Reuse the group's proposal labeler
            labels = labeler.label(gt_boxes, proposals)

            for box, label in zip(proposals, labels):

                # Supports ignored samples if added later
                if label == -1:
                    continue

                self.samples.append((
                    str(image_path),
                    box.astype(int).tolist(),
                    int(label)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, box, label = self.samples[idx]

        # Crop proposal from original image
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            crop = image.crop(tuple(box))

        crop = self.transform(crop)

        return crop, torch.tensor(label, dtype=torch.long)


def create_dataloader(dataset, batch_size=BATCH_SIZE, training=True):
    """Create balanced training or standard validation loader."""

    if training:
        labels = np.asarray(
            [sample[2] for sample in dataset.samples],
            dtype=np.int64
        )

        counts = np.bincount(labels, minlength=2)

        if np.any(counts == 0):
            raise ValueError(
                "Training data must contain both classes."
            )

        # Target distribution: 75% background, 25% pothole
        class_weights = np.array([0.75, 0.25]) / counts
        sample_weights = class_weights[labels]

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
            pin_memory=torch.cuda.is_available()
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available())