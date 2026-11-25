import torchvision.transforms as transforms
import torch
import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
from sklearn.model_selection import train_test_split

def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes

class potholes(torch.utils.data.Dataset):
    def __init__(self, transform):
        'Initialization'
        self.transform = transform
        self.DATA_PATH = '/dtu/datasets1/02516/potholes'
        # data_path = os.path.join(self.DATA_PATH)
        self.images_path = sorted(glob.glob(self.DATA_PATH + "/images/*.png"))
        self.annotations_path = sorted(glob.glob(self.DATA_PATH + "/annotations/*.xml"))

    def __len__(self):
        'Returns the total number of samples'
        return len(self.images_path)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_file = self.images_path[idx]
        annotation_file = self.annotations_path[idx]

        image = Image.open(image_file)
        annotation = read_content(annotation_file)

        filename, boxes = annotation

        if self.transform:
            X = self.transform(image)
        else:
            X = image
        Y = boxes

        return X, Y

if __name__ == "__main__":
    train_transform = transforms.Compose(transforms.ToTensor())
    potholes_dataset = potholes(transform=train_transform)
    train_imgs, test_imgs, train_xmls, test_xmls = train_test_split(potholes_dataset.images_path, potholes_dataset.annotations_path, test_size=0.1, random_state=42)
    print(f"Loaded {len(potholes_dataset)} images")
    print(f"First sample: {potholes_dataset[0]}")