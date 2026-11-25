import numpy as np


class RegionProposalLabeler:

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    @staticmethod
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

    def label(self, gt, props):
        gt = np.asarray(gt, dtype=np.float32).reshape(-1, 4)
        props = np.asarray(props, dtype=np.float32).reshape(-1, 4)

        if len(gt) == 0 or len(props) == 0:
            return np.zeros(len(props), dtype=np.int64)

        ious = self.iou(gt, props)

        max_ious = ious.max(axis=0)

        labels = (max_ious >= self.iou_threshold).astype(np.int64)

        return labels