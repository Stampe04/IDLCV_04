import cv2


from datasets import potholes


# Load image
image = cv2.imread('image.jpg')

# Initialize Selective Search
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)

# Choose method: 'fast' or 'quality'
ss.switchToSelectiveSearchFast()  # or switchToSelectiveSearchQuality()

# Generate region proposals
rects = ss.process()

# rects contains tuples of (x, y, w, h)
print(f"Total proposals: {len(rects)}")