# ObjectStitch-Image-Composition

**IMPORTANT NOTE:**  
The test.sh script has been updated to perform more experiments. It must now be executed using bash (i.e., run with "bash test.sh" instead of "sh test.sh") due to the automation of multiple experiments within the script.

```
# Mask Generation Scripts

This repository provides scripts to generate binary masks for foreground and background images to be used as model inputs.

## Requirements Before Running

1. **Image Preparation**:
   - All input images must be resized to **512×512 pixels**.
   - Foreground images should have the main object extending across the full height and width.

2. **Install Dependencies**:
```

pip install opencv-python numpy torch
pip install git+[https://github.com/facebookresearch/segment-anything.git](https://github.com/facebookresearch/segment-anything.git)

```

3. **Download SAM Model**:
- Download the `sam_vit_l_0b3195.pth` checkpoint from the official SAM repository: https://github.com/facebookresearch/segment-anything
- Update the path to this checkpoint in `segment_script.py`.

## Scripts

### Generate_Masks/segment_script.py
Automatically generates binary foreground masks using Meta's Segment Anything Model (SAM). For each image, it selects the largest mask region and saves it as a binary mask.

### Generate_Masks/background_mask.py
Allows manual drawing of a square bounding box on each background image. The selected region is saved as a binary mask with the square in white and the rest in black.

## Output

The scripts generate the following directories:
- `foreground_mask/`: Contains binary foreground masks.
- `mask_bbox/`: Contains binary background masks.

These folders serve as input to the model.
```
