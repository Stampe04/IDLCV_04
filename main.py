

from IDLCV_04.regional_proposal_labeler import RegionProposalLabeler
from IDLCV_04.regional_proposals import RegionProposals
from IDLCV_04.dataset import potholes
import cv2

img_path = "/dtu/datasets1/02516/potholes"
xml_path = "/dtu/datasets1/02516/potholes"

rp = RegionProposals(
    top_n_props=850,
    alpha=0.65,
    beta=0.75
)

labeler = RegionProposalLabeler(iou_threshold=0.5)

img = cv2.imread(img_path)
filename, gt = read_content(xml_path)

# Generate region proposals
props = rp.edge_boxes(img)

# Assign labels to region proposals
labels = labeler.label(gt, props)