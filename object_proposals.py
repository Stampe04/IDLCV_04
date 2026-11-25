import cv2
import numpy as np
import glob
from dataset import *
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

sed = cv2.ximgproc.createStructuredEdgeDetection("model.yml.gz")

def edge_boxes(img_bgr, top_n_props=200, alpha=0.65, beta=0.75):

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    edges = sed.detectEdges(rgb)
    orient = sed.computeOrientation(edges)
    edges = sed.edgesNms(edges, orient)

    eb = cv2.ximgproc.createEdgeBoxes()
    eb.setMaxBoxes(top_n_props)
    eb.setAlpha(alpha)
    eb.setBeta(beta)
    boxes, scores = eb.getBoundingBoxes(edges, orient)
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = np.asarray(boxes, dtype=np.float32)
    boxes = np.stack([boxes[:, 0], boxes[:, 1],
                      boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]], 1)
    scores = np.asarray(scores, dtype=np.float32).ravel()
    order = np.argsort(-scores)
    return boxes[order][:top_n_props]

def selective_search(img_bgr, top_n_props=200):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_bgr)
    ss.switchToSelectiveSearchFast()

    rects = ss.process()
    if len(rects) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = np.asarray(rects, dtype=np.float32)
    boxes = np.stack([boxes[:, 0], boxes[:, 1],
                      boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]], 1)
    return boxes[:top_n_props]

def iou(gt, props):
    gt = gt[:, None]
    props =  props[None]
    iw = np.clip(np.minimum(gt[..., 2], props[..., 2]) - np.maximum(gt[..., 0], props[..., 0]), 0, None)
    ih = np.clip(np.minimum(gt[..., 3], props[..., 3]) - np.maximum(gt[..., 1], props[..., 1]), 0, None)
    inter = iw * ih
    a = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])
    b = (props[..., 2] - props[..., 0]) * (props[..., 3] - props[..., 1])
    return inter / (a + b - inter)

Ns = [10, 25, 50, 100, 200, 500, 750, 1000, 1250]
thr = 0.55
root = "/dtu/datasets1/02516/potholes"
potholes_dataset = potholes(transform=None)
train_imgs, test_imgs, train_xmls, test_xmls = train_test_split(potholes_dataset.images_path, potholes_dataset.annotations_path, test_size=0.1, random_state=42)

hits_edge = {n: 0 for n in Ns}
hits_ss = {n: 0 for n in Ns}
total = 0

for ip, xp in zip(train_imgs, train_xmls):
    img = cv2.imread(ip)
    filename, gt = read_content(xp)
    gt = np.array(gt, dtype=np.float32).reshape(-1, 4)

    if len(gt) == 0:
        print("Nincs GT box:", ip)
        continue

    total += len(gt)

    e = edge_boxes(img, 2000)
    s = selective_search(img, 2000)

    if len(e) == 0:
        print("Üres EdgeBoxes:", ip)

    for n in Ns:
        if len(e) > 0:
            hits_edge[n] += (iou(gt, e[:n]).max(axis=1) >= thr).sum()
        if len(s) > 0:
            hits_ss[n] += (iou(gt, s[:n]).max(axis=1) >= thr).sum()

for n in Ns:
    print(n, "EdgeBoxes:", hits_edge[n] / total, "SelSearch:", hits_ss[n] / total)