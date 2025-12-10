import os
import cv2
from .detector import detect_text_boxes
from .recognizer import recognize_from_crop
from .cleaner import preprocess_image, final_clean, expand_box
from .utils import ensure_dir, save_json, load_config
import logging

cfg = load_config()

def process_image(image_path, output_dir):
    ensure_dir(output_dir)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    # preprocessing (not destructive)
    pre = preprocess_image(img, cfg)
    # detection
    boxes = detect_text_boxes(img, method=cfg['detector'].get('method','auto'))
    results = []
    idx = 0
    for box in boxes:
        idx += 1
        x0, y0, x1, y1 = expand_box(box, img.shape, pad=6)
        crop = img[y0:y1, x0:x1]
        text, conf = recognize_from_crop(crop, lang=cfg['recognizer'].get('lang','eng'))
        clean_text = final_clean(text)
        # optionally save crop
        crop_path = None
        if cfg['output'].get('save_crops', True):
            crop_name = os.path.splitext(os.path.basename(image_path))[0] + f"_crop_{idx}.png"
            crop_path = os.path.join(output_dir, crop_name)
            cv2.imwrite(crop_path, crop)
        results.append({
            "box": [int(x0), int(y0), int(x1), int(y1)],
            "text_raw": text,
            "text_clean": clean_text,
            "confidence": conf,
            "crop_path": crop_path
        })
    # export json
    if cfg['output'].get('export_json', True):
        json_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".json")
        save_json({"image": os.path.basename(image_path), "results": results}, json_path)
    return results
