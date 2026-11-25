import cv2
import numpy as np


class RegionProposals:

    def __init__(
        self,
        model_path="model.yml.gz",
        top_n_props=200,
        alpha=0.65,
        beta=0.75
    ):
        self.top_n_props = top_n_props

        # Structured Edge Detection
        self.sed = cv2.ximgproc.createStructuredEdgeDetection(
            model_path
        )

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

    def edge_boxes(self, img_bgr):
        rgb = cv2.cvtColor(
            img_bgr, cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0

        edges = self.sed.detectEdges(rgb)
        orient = self.sed.computeOrientation(edges)
        edges = self.sed.edgesNms(edges, orient)

        boxes, scores = self.eb.getBoundingBoxes(edges, orient)

        boxes = self.xywh_to_xyxy(boxes)
        scores = np.asarray(scores, dtype=np.float32).ravel()

        order = np.argsort(-scores)

        return boxes[order][:self.top_n_props]

    def selective_search(self, img_bgr):
        self.ss.setBaseImage(img_bgr)
        self.ss.switchToSelectiveSearchFast()

        rects = self.ss.process()
        boxes = self.xywh_to_xyxy(rects)

        return boxes[:self.top_n_props]