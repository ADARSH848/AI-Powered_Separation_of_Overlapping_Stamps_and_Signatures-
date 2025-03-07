import os
import cv2
import numpy as np
import torch
from torchvision import transforms as T
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from PIL import Image

# Path to images and annotations
image_base_path = './batch1'
annotation_file = './batch1.json'

# Load COCO annotations
coco = COCO(annotation_file)

# Get all image IDs
image_ids = coco.getImgIds()

# Define transformation
transform = T.Compose([T.ToTensor()])

# Function to display image with masks
def display_image_with_masks(image_path, annotations, coco):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    
    # Draw masks on the image
    for ann in annotations:
        mask = coco.annToMask(ann)
        masked_image = np.ma.masked_where(mask == 0, mask)
        plt.imshow(masked_image, alpha=0.5, cmap='jet')
    
    plt.axis('off')
    plt.show()

# Iterate through images and display masks
for image_id in image_ids[:5]:  # Display first 5 images
    img_info = coco.loadImgs(image_id)[0]
    img_path = os.path.join(image_base_path, img_info['file_name'])
    
    # Get annotations for the current image
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    
    # Display image with masks
    display_image_with_masks(img_path, anns, coco)

