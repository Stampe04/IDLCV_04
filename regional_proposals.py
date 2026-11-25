import cv2
import numpy as np


class RegionProposals:

    def __init__(
        self,
        top_n_props=200,
        alpha=0.65,
        beta=0.75
    ):
        self.top_n_props = top_n_props

        # EdgeBoxes
        self.eb = cv2.ximgproc.createEdgeBoxes()
        self.eb.setMaxBoxes(top_n_props)
        self.eb.setAlpha(alpha)
        self.eb.setBeta(beta)

        # Selective Search
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    @staticmethod
    def xywh_to_xyxy(boxes):
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

        return np.stack([
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 0] + boxes[:, 2],
            boxes[:, 1] + boxes[:, 3]
        ], axis=1)

    def selective_search(self, img_bgr):
        self.ss.setBaseImage(img_bgr)
        self.ss.switchToSelectiveSearchFast()

        rects = self.ss.process()
        boxes = self.xywh_to_xyxy(rects)

        return boxes[:self.top_n_props]