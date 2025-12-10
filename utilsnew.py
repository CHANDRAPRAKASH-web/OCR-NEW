import os
import cv2
import numpy as np
from pathlib import Path
import json
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def read_image(path, max_width=None, max_height=None):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    if max_width or max_height:
        h, w = img.shape[:2]
        scale = 1.0
        if max_width and w > max_width:
            scale = min(scale, max_width / w)
        if max_height and h > max_height:
            scale = min(scale, max_height / h)
        if scale != 1.0:
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img

def save_json(obj, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
