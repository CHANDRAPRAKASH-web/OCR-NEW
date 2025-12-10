import pytesseract
from PIL import Image
import cv2
import numpy as np
from .utils import load_config
import logging

cfg = load_config()

def recognize_from_crop(crop, lang=None):
    if lang is None:
        lang = cfg['recognizer'].get('lang','eng')
    # ensure crop is in correct format for pytesseract
    if isinstance(crop, np.ndarray):
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    else:
        pil = crop
    custom_config = r'--oem 3 --psm 6'  # good general config
    text = pytesseract.image_to_string(pil, lang=lang, config=custom_config)
    conf_data = pytesseract.image_to_data(pil, lang=lang, output_type=pytesseract.Output.DICT)
    # compute mean confidence for non-empty words
    confs = [int(c) for c in conf_data['conf'] if c.isdigit()]
    mean_conf = int(sum(confs)/len(confs)) if confs else -1
    logging.debug(f"Recognized text: {text.strip()} (conf={mean_conf})")
    return text.strip(), mean_conf
