#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np
import fastdeploy as fd
from PIL import Image
import io
import gc

# ----------------------- Hardcoded Constants -----------------------

# Model file paths
DET_MODEL_DIR='./en_PP-OCRv3_det_infer'
REC_MODEL_DIR='./en_PP-OCRv3_rec_infer'
CLS_MODEL_DIR='./ch_ppocr_mobile_v2.0_cls_infer'
REC_LABEL_FILE='./en_dict.txt'

dataset_name = 'ViDoSeek' # dataset name
# Input image path
IMAGE_PATH = f'./data/{dataset_name}/img'  # Write your image path here

# Inference device configuration
BACKEND = "cpu"  # Options: "gpu" or "cpu"
DEVICE_ID = 0     # Set GPU device ID if using GPU

# Other parameters
MIN_SCORE = 0.6  # Recognition score threshold

# ----------------------- Function Definitions -----------------------

def decode_image(image_path):
    """
    Decode the image from the given path.
    """
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    elif isinstance(image_path, Image.Image):
        image =image_path
    else:
        raise ValueError("Invalid image path or image object.")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def calculate_spaces_and_newlines(current_box, previous_box, space_threshold=45, line_threshold=15):
    """Calculate the number of spaces and newlines between two text boxes."""
    spaces = 0
    newlines = 0
    
    # Check if the text boxes are on the same line
    if abs(current_box[1] - previous_box[1]) < line_threshold:
        spaces = max(1, int(abs(current_box[0] - previous_box[0]) / space_threshold))
    else:
        newlines = max(1, int(abs(current_box[1] - previous_box[1]) / line_threshold))
    
    return spaces, newlines

def tostr_layout_preserving(result):
    """Convert OCR results into a layout-preserving merged string."""
    text_boxes = []
    for box, text, score in zip(result.boxes, result.text, result.rec_scores):
        if score >= MIN_SCORE:  # Only include text boxes with score >= 0.6
            coords = [(box[i], box[i + 1]) for i in range(0, len(box), 2)]
            center_x = (coords[0][0] + coords[2][0]) / 2
            center_y = (coords[0][1] + coords[2][1]) / 2
            text_boxes.append((center_x, center_y, text, coords))

    # Sort text boxes from top to bottom and left to right
    text_boxes = sorted(text_boxes, key=lambda x: (x[1], x[0]))
    
    # Merge text boxes
    merged_text = []
    previous_box = None
    for box in text_boxes:
        if previous_box is not None:
            spaces, newlines = calculate_spaces_and_newlines(box, previous_box)
            merged_text.append('\n' * newlines + ' ' * spaces)
        merged_text.append(box[2])
        previous_box = box

    res_text = ''.join(merged_text)
    return res_text

def build_option():
    """Build FastDeploy runtime options based on backend and device."""
    det_option = fd.RuntimeOption()
    cls_option = fd.RuntimeOption()
    rec_option = fd.RuntimeOption()

    if BACKEND.lower() == "gpu":
        det_option.use_gpu(DEVICE_ID)
        cls_option.use_gpu(DEVICE_ID)
        rec_option.use_gpu(DEVICE_ID)
    else:
        det_option.use_cpu()
        cls_option.use_cpu()
        rec_option.use_cpu()

    return det_option, cls_option, rec_option


# Build model file paths
det_model_file = os.path.join(DET_MODEL_DIR, "inference.pdmodel")
det_params_file = os.path.join(DET_MODEL_DIR, "inference.pdiparams")

cls_model_file = os.path.join(CLS_MODEL_DIR, "inference.pdmodel")
cls_params_file = os.path.join(CLS_MODEL_DIR, "inference.pdiparams")

rec_model_file = os.path.join(REC_MODEL_DIR, "inference.pdmodel")
rec_params_file = os.path.join(REC_MODEL_DIR, "inference.pdiparams")

# Build runtime options
det_option, cls_option, rec_option = build_option()

# Initialize models
det_model = fd.vision.ocr.DBDetector(
    det_model_file, det_params_file, runtime_option=det_option
)

cls_model = fd.vision.ocr.Classifier(
    cls_model_file, cls_params_file, runtime_option=cls_option
)

rec_model = fd.vision.ocr.Recognizer(
    rec_model_file, rec_params_file, REC_LABEL_FILE, runtime_option=rec_option
)

# Set preprocessor and postprocessor parameters for the Det model
det_model.preprocessor.max_side_len = 960
det_model.postprocessor.det_db_thresh = 0.3
det_model.postprocessor.det_db_box_thresh = 0.6
det_model.postprocessor.det_db_unclip_ratio = 1.5
det_model.postprocessor.det_db_score_mode = "slow"
det_model.postprocessor.use_dilation = False

# Set postprocessor parameters for the Cls model
cls_model.postprocessor.cls_thresh = 0.9

# Create PP-OCRv3 instance
ppocr_v3 = fd.vision.ocr.PPOCRv3(
    det_model=det_model, cls_model=cls_model, rec_model=rec_model
)

# ----------------------- Main Function -----------------------
def main(img_name):
    image = decode_image(img_name)
    result = ppocr_v3.predict(image)
    text = tostr_layout_preserving(result)
    return text

if __name__ == "__main__":
    from tqdm import tqdm
    for img in tqdm(os.listdir(IMAGE_PATH)):
        txt_name = img.replace('.jpg', '.txt')
        txt_name = txt_name.replace('.png', '.txt')
        txt_path = os.path.join(f'./data/{dataset_name}/ppocr/{txt_name}')
        if os.path.exists(txt_path):
            continue
        text = main(os.path.join(IMAGE_PATH, img))
        # dump
        with open(os.path.join(f'./data/{dataset_name}/ppocr/{txt_name}'), 'w') as f:
            f.write(text)