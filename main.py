import sys
sys.path.append("/zhome/99/3/215784/Desktop/archive/IDLCV/IDLCV_04")

from regional_proposal_labeler import RegionProposalLabeler
from regional_proposals import RegionProposals
from dataset import potholes, read_content
import cv2

img_path = "/dtu/datasets1/02516/potholes"
xml_path = "/dtu/datasets1/02516/potholes"

rp = RegionProposals(
    top_n_props=750
)

labeler = RegionProposalLabeler(iou_threshold=0.5)

img = cv2.imread(img_path)
filename, gt = read_content(xml_path)

# Generate region proposals
props = rp.selective_search(img)

# Assign labels to region proposals
labels = labeler.label(gt, props)

print(f"Ground truth boxes: {gt}")
print(f"Region proposals: {props}")
print(f"Labels: {labels}")