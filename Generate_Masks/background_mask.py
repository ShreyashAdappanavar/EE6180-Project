"""
This script loads .png images from a specified directory, displays each image one by one, and allows the user to draw a square bounding box using the mouse. 
Once the user confirms the selection (by pressing Enter), the script creates and saves a binary mask (with the box region in white and the rest in black) to 
an output directory. If the user presses 'q', the image is skipped. 

This script was run on my laptop since the compute cluster doesnt have an easy way to display the window for drawing the bounding boxes.
"""


import os
import cv2
import numpy as np

# Directories
background_input_img_base_dir = "C:/1_Shre_Core/IIT Madras/EE6180/Project/background_mask/background"
background_output_img_base_dir = "C:/1_Shre_Core/IIT Madras/EE6180/Project/background_mask/mask_bbox"

os.makedirs(background_output_img_base_dir, exist_ok=True)

# Get all .png files in the directory
img_list = [f for f in os.listdir(background_input_img_base_dir) if f.endswith('.png')]
img_list.sort()

# Globals for mouse callback
ix, iy = -1, -1
fx, fy = -1, -1
drawing = False

def on_mouse(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif (event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONUP) and drawing:
        # Compute delta
        dx, dy = x - ix, y - iy
        # Square side = max of dx, dy (one can also choose min, but generally max is chosen)
        side = max(abs(dx), abs(dy))
        # Recompute fx, fy so selection is always a square
        fx = ix + side * (1 if dx >= 0 else -1)
        fy = iy + side * (1 if dy >= 0 else -1)
        if event == cv2.EVENT_LBUTTONUP:
            drawing = False

def process_with_gui(input_path, output_path):
    global ix, iy, fx, fy, drawing

    img = cv2.imread(input_path)
    if img is None:
        print(f"Failed to load {input_path}")
        return
    if img.shape[:2] != (512, 512):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

    winname = "Draw Square Bounding Box (Square) and press Enter to Save, or Q to Skip"
    cv2.namedWindow(winname)
    cv2.setMouseCallback(winname, on_mouse, {'img': img, 'winname': winname})

    # Reset globals for each image
    ix = iy = fx = fy = -1
    drawing = False

    while True:
        display = img.copy()
        # Only draw if a selection has started
        if ix != -1 and fx != -1:
            # Draw the square
            cv2.rectangle(display, (ix, iy), (fx, fy), (0, 255, 0), 2)
            # Compute and show side length
            side = max(abs(fx - ix), abs(fy - iy))
            cv2.putText(display,
                        f"Size: {side}px",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)
        cv2.imshow(winname, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # "Enter" key
            if ix != -1 and fx != -1:
                # Make mask and save
                mask = np.zeros((512, 512), dtype=np.uint8)
                x0, x1 = sorted([ix, fx])
                y0, y1 = sorted([iy, fy])
                mask[y0:y1, x0:x1] = 255
                cv2.imwrite(output_path, mask)
            break
        elif key == ord('q'):
            break

    cv2.destroyWindow(winname)

# Loop over all images
for img_name in img_list:
    input_path  = os.path.join(background_input_img_base_dir, img_name)
    output_path = os.path.join(background_output_img_base_dir, img_name)
    process_with_gui(input_path, output_path)
