from glob import glob
import sys
sys.path.append("/zhome/99/3/215784/Desktop/archive/IDLCV/IDLCV_04")

from regional_proposal_labeler import RegionProposalLabeler
from regional_proposals import RegionProposals
from dataset import potholes, read_content
import cv2

root = "/dtu/datasets1/02516/potholes"
img_paths = sorted(glob(root + "/images/*.png"))
xml_paths = sorted(glob(root + "/annotations/*.xml"))

rp = RegionProposals(
    top_n_props=750
)

labeler = RegionProposalLabeler(iou_threshold=0.5)

for img_path, xml_path in zip(img_paths, xml_paths):
    img = cv2.imread(img_path)
    filename, gt = read_content(xml_path)

    # Generate region proposals
    props = rp.selective_search(img)

    xmin, ymin, xmax, ymax = props[0]

    # Assign labels to region proposals
    labels = labeler.label(gt, props)

    # extract images from region proposals convert
    cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]

    #resize the cropped image to 128x128 as a png
    cropped_img = cv2.resize(cropped_img, (128, 128))

    print(cropped_img.shape, labels[0])
    #save the cropped image to disk
    cv2.imwrite("Cropped Image.png", cropped_img)

    break