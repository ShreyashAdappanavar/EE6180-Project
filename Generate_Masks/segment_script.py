"""
This script loads a pre-trained SAM model to generate and save binary segmentation masks for all .png images in a specified input directory, 
storing the results in an output directory. 

This script was run on CeRAI Labs compute cluster since my Laptop couldn't efficiently run the Segment-Anything-Model model.
"""


import cv2
import numpy as np
import torch
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


#### Directoery names and model path. Enter the absolute path ovr here to ensure it works properly.

foreground_input_img_base_dir = "/home/gokul/Hier-Legal-Graph/mimic_dataset/mimiciv_dataset/2.2/Tempp/Generate_Masks/foreground"
foreground_output_img_base_dir = "/home/gokul/Hier-Legal-Graph/mimic_dataset/mimiciv_dataset/2.2/Tempp/Generate_Masks/foreground_mask"
sam_checkpoint_path = "/home/gokul/Hier-Legal-Graph/mimic_dataset/mimiciv_dataset/2.2/Tempp/Generate_Masks/SAM_Checkpoint/sam_vit_l_0b3195.pth"

######

os.makedirs(foreground_output_img_base_dir, exist_ok=True)

# Get all .png files in the directory
img_list = [f for f in os.listdir(foreground_input_img_base_dir) if f.endswith('.png')]
img_list.sort()

# Load SAM (ViT-L) with pretrained weights
sam = sam_model_registry["vit_l"](checkpoint=sam_checkpoint_path)
sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
mask_generator = SamAutomaticMaskGenerator(sam)

def process_image(input_path: str, output_path: str):
    image = cv2.imread(input_path)
    if image.shape[:2] != (512, 512):
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Generate all masks, pick the largest by area
    masks = mask_generator.generate(image_rgb)
    best_mask = max(masks, key=lambda m: m["area"])
    seg = best_mask["segmentation"]  

    mask_uint8 = (seg.astype(np.uint8)) * 255           
    binary_output = np.ascontiguousarray(mask_uint8)    

    contours, _ = cv2.findContours(binary_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(binary_output, contours, -1, color=255, thickness=cv2.FILLED)                                                   
    cv2.imwrite(output_path, binary_output)


for img_name in img_list:
    input_path = os.path.join(foreground_input_img_base_dir, img_name)
    output_path = os.path.join(foreground_output_img_base_dir, img_name)

    process_image(input_path, output_path)