import cv2
import numpy as np

sed = cv2.ximgproc.createStructuredEdgeDetection("model.yml.gz")

def edge_boxes(img_bgr, top_n=200, alpha=0.65, beta=0.75):

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    edges = sed.detectEdges(rgb)
    orient = sed.computeOrientation(edges)
    edges = sed.edgesNms(edges, orient)

    eb = cv2.ximgproc.createEdgeBoxes()
    eb.setMaxBoxes(top_n)
    eb.setAlpha(alpha)
    eb.setBeta(beta)
    boxes, scores = eb.getBoundingBoxes(edges, orient) 

    boxes = np.asarray(boxes, dtype=np.float32)
    boxes = np.stack([boxes[:, 0], boxes[:, 1],
                      boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]], 1)
    scores = np.asarray(scores, dtype=np.float32).ravel()
    return boxes, scores

def selective_search(img_bgr, top_n=200):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_bgr)
    ss.switchToSelectiveSearchFast()

    rects = ss.process()

    boxes = np.asarray(rects, dtype=np.float32)
    boxes = np.stack([boxes[:, 0], boxes[:, 1],
                      boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]], 1)
    return boxes[:top_n]

img = cv2.imread("/dtu/datasets1/02516/potholes/images/potholes1.png")
props, scores = edge_boxes(img, top_n=200)
props_2 = selective_search(img, top_n=200)