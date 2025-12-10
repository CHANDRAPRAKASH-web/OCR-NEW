import os
import cv2
import numpy as np
import pytesseract
from .utils import load_config
import logging

cfg = load_config()

def decode_predictions(scores, geometry, score_thresh=0.5):
    # Standard EAST decode
    detections = []
    confidences = []

    (numRows, numCols) = scores.shape[2:4]
    for y in range(0, numRows):
        scoresData = scores[0,0,y]
        x0_data = geometry[0,0,y]
        x1_data = geometry[0,1,y]
        x2_data = geometry[0,2,y]
        x3_data = geometry[0,3,y]
        anglesData = geometry[0,4,y]
        for x in range(0, numCols):
            score = scoresData[x]
            if score < score_thresh:
                continue
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]
            endX = int(offsetX + (cos * x1_data[x]) + (sin * x2_data[x]))
            endY = int(offsetY - (sin * x1_data[x]) + (cos * x2_data[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            detections.append((startX, startY, endX, endY))
            confidences.append(float(score))
    return (detections, confidences)

def east_detect(image, east_path=None, min_confidence=0.5):
    if not east_path:
        east_path = cfg['detector'].get('east_model_path')
    if not east_path or not os.path.exists(east_path):
        raise FileNotFoundError("EAST model not found. Provide a valid path or use pytesseract fallback.")
    orig_h, orig_w = image.shape[:2]
    newW, newH = (320, 320)
    rW = orig_w / float(newW)
    rH = orig_h / float(newH)
    blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net = cv2.dnn.readNet(east_path)
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"])
    (rects, confidences) = decode_predictions(scores, geometry, score_thresh=min_confidence)
    boxes = []
    for (startX, startY, endX, endY) in rects:
        # scale back
        sX = int(startX * rW)
        sY = int(startY * rH)
        eX = int(endX * rW)
        eY = int(endY * rH)
        # sanitize box
        sX, sY = max(0, sX), max(0, sY)
        eX, eY = min(orig_w - 1, eX), min(orig_h - 1, eY)
        boxes.append((sX, sY, eX, eY))
    return boxes

def pytesseract_detect(image, lang='eng', conf_thresh=50):
    # returns list of boxes from pytesseract.image_to_data
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    boxes = []
    n = len(data['level'])
    for i in range(n):
        conf = float(data['conf'][i]) if data['conf'][i] != '-1' else -1
        if conf >= conf_thresh:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            boxes.append((x, y, x + w, y + h))
    return boxes

def detect_text_boxes(image, method='auto'):
    method = method.lower()
    if method == 'east':
        try:
            boxes = east_detect(image, east_path=cfg['detector'].get('east_model_path'))
            logging.info(f"EAST detected {len(boxes)} boxes")
            return boxes
        except Exception as e:
            logging.warning(f"EAST detection failed: {e}. Falling back to pytesseract.")
            return pytesseract_detect(image, lang=cfg['recognizer'].get('lang','eng'))
    elif method == 'pytesseract':
        return pytesseract_detect(image, lang=cfg['recognizer'].get('lang','eng'))
    else:  # auto
        # try EAST first if model exists
        east_path = cfg['detector'].get('east_model_path')
        if east_path and os.path.exists(east_path):
            try:
                boxes = east_detect(image, east_path=east_path)
                logging.info(f"EAST detected {len(boxes)} boxes")
                return boxes
            except Exception as e:
                logging.warning(f"EAST failed: {e}. Using pytesseract fallback.")
                return pytesseract_detect(image, lang=cfg['recognizer'].get('lang','eng'))
        else:
            logging.info("EAST model not found. Using pytesseract fallback.")
            return pytesseract_detect(image, lang=cfg['recognizer'].get('lang','eng'))
