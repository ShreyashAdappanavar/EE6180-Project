# ObjectStitch-Image-Composition

**IMPORTANT NOTE:**  
The test.sh script has been updated to perform more experiments. It must now be executed using bash (i.e., run with "bash test.sh" instead of "sh test.sh") due to the automation of multiple experiments within the script.


Sure, here's a **brief, clean, and general README** without emojis, following your instructions:

---

# Mask Generation Scripts

This repository provides scripts to generate binary masks for foreground and background images to be used as model inputs.

## Requirements Before Running

1. **Image Preparation**:

   * All input images must be resized to **512×512 pixels**.
   * Foreground images should have the main object extending across the full height and width.

2. **Install Dependencies**:

   ```bash
   pip install opencv-python numpy torch
   pip install git+https://github.com/facebookresearch/segment-anything.git
   ```

3. **Download SAM Model**:

   * Get the `sam_vit_l_0b3195.pth` checkpoint from the [official repo](https://github.com/facebookresearch/segment-anything) and update the path in `segment_script.py`.

## Scripts

### 1. `Generate_Masks/segment_script.py`

* Uses Meta's SAM model to generate **foreground** masks.
* Loads each image, runs segmentation, selects the largest region, and saves it as a binary mask.

### 2. `Generate_Masks/background_mask.py`

* Allows manual selection of a square region using the mouse to create **background** masks.
* The selected region is saved as a white square on a black background.

## Output

After running both scripts, you will get:

* `foreground_mask/`: Binary masks for foreground images.
* `mask_bbox/`: Binary masks for background images.

These outputs can be directly used as inputs for model training or evaluation.
