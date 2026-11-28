import numpy as np

def iou(gt, props):
        gt = np.asarray(gt, dtype=np.float32).reshape(-1, 4)
        props = np.asarray(props, dtype=np.float32).reshape(-1, 4)

        gt = gt[:, None]
        props = props[None]

        iw = np.clip(
            np.minimum(gt[..., 2], props[..., 2])
            - np.maximum(gt[..., 0], props[..., 0]),
            0, None
        )

        ih = np.clip(
            np.minimum(gt[..., 3], props[..., 3])
            - np.maximum(gt[..., 1], props[..., 1]),
            0, None
        )

        inter = iw * ih

        a = (
            (gt[..., 2] - gt[..., 0])
            * (gt[..., 3] - gt[..., 1])
        )

        b = (
            (props[..., 2] - props[..., 0])
            * (props[..., 3] - props[..., 1])
        )

        union = a + b - inter

        return np.divide(
            inter,
            union,
            out=np.zeros_like(inter),
            where=union > 0
        )

def average_precision(predictions, ground_truth, iou_threshold=0.5):
    """
    Calculate dataset-level AP for one object class.

    predictions: list of dictionaries
        {
            "image_id": str,
            "box": [xmin, ymin, xmax, ymax],
            "score": float
        }

    ground_truth: dictionary
        {
            image_id: [[xmin, ymin, xmax, ymax], ...]
        }

    Returns:
        ap, precision, recall
    """

    # Sort predictions by confidence
    predictions = sorted(
        predictions,
        key=lambda p: p["score"],
        reverse=True
    )

    # Count ground-truth objects
    total_gt = sum(len(boxes) for boxes in ground_truth.values())

    if total_gt == 0:
        raise ValueError("No ground-truth boxes available.")

    # Keep track of matched ground-truth boxes
    matched = {
        image_id: np.zeros(len(boxes), dtype=bool)
        for image_id, boxes in ground_truth.items()
    }

    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))

    # Evaluate each prediction
    for i, pred in enumerate(predictions):
        image_id = pred["image_id"]
        pred_box = pred["box"]

        gt_boxes = ground_truth.get(image_id, [])

        if len(gt_boxes) == 0:
            fp[i] = 1
            continue

        overlaps = [iou(gt, pred_box) for gt in gt_boxes]

        best_idx = int(np.argmax(overlaps))
        best_iou = overlaps[best_idx]

        if best_iou >= iou_threshold and not matched[image_id][best_idx]:
            tp[i] = 1
            matched[image_id][best_idx] = True
        else:
            fp[i] = 1

    # Calculate cumulative TP and FP
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp, 1e-10
    )
    recall = cumulative_tp / total_gt

    # Add endpoints for AP calculation
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Interpolated precision
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Area under precision-recall curve
    indices = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum(
        (mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]
    )

    return float(ap), precision, recall

def mean_average_precision(predictions, ground_truth):
    """
    Average AP across IoU thresholds 0.50 to 0.95.
    Single-class dataset.
    """
    thresholds = np.arange(0.50, 1.00, 0.05)

    aps = [
        average_precision(predictions, ground_truth, t)[0]
        for t in thresholds
    ]

    return float(np.mean(aps))