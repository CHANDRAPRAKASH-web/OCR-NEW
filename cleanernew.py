import cv2
import numpy as np
import re
import logging

def preprocess_image(img, cfg=None):
    # cfg can control steps
    img_proc = img.copy()
    if cfg and cfg['preprocess'].get('gray', True):
        img_proc = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
    # bilateral filter to preserve edges and reduce noise
    if cfg and cfg['preprocess'].get('bilateral_filter', True):
        img_proc = cv2.bilateralFilter(img_proc, d=9, sigmaColor=75, sigmaSpace=75)
    # adaptive thresholding optional
    if cfg and cfg['preprocess'].get('adaptive_thresh', False):
        img_proc = cv2.adaptiveThreshold(img_proc, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    return img_proc

def expand_box(box, image_shape, pad=5):
    h, w = image_shape[:2]
    x0, y0, x1, y1 = box
    x0 = max(0, x0-pad)
    y0 = max(0, y0-pad)
    x1 = min(w-1, x1+pad)
    y1 = min(h-1, y1+pad)
    return (x0, y0, x1, y1)

def final_clean(text):
    """
    Robust final cleaning pipeline for OCR text.
    IMPORTANT: this function avoids early returns and always runs consistent cleaning steps,
    fixing the "returned before final block" bug.
    Steps:
    1. normalize whitespace
    2. remove non-printable characters
    3. simple common OCR correction rules
    4. strip leading/trailing punctuation but keep valid punctuation inside
    """
    # Early validation
    if text is None:
        text = ""
    # Step 1: Normalize unicode and whitespace
    cleaned = text.replace('\r', '\n').replace('\t', ' ')
    # collapse multiple spaces and newlines (but keep single newline)
    cleaned = re.sub(r'[ \u00A0]+', ' ', cleaned)
    cleaned = re.sub(r'\n{2,}', '\n', cleaned)
    # Step 2: Remove non-printables
    cleaned = ''.join(ch for ch in cleaned if ch.isprintable())
    # Step 3: Basic OCR corrections (common mistakes)
    # Keep a list of deterministic replacements; do not return early
    replacements = {
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '“': '"', '”': '"',
    }
    for k, v in replacements.items():
        cleaned = cleaned.replace(k, v)
    # fix typical substitutions (like '0' vs 'O' when surrounded by digits/letters heuristics)
    cleaned = re.sub(r'(?<=\d)[O](?=\d)', '0', cleaned)  # O between digits -> 0
    cleaned = re.sub(r'(?<=\D)[0](?=\D)', 'O', cleaned)  # 0 between non-digits -> O
    # collapse spaces before punctuation
    cleaned = re.sub(r'\s+([,.;:!?%])', r'\1', cleaned)
    # trim
    cleaned = cleaned.strip()
    # final safety: if after all cleaning nothing meaningful, keep original minimal cleaned version
    if not cleaned:
        cleaned = text.strip() if text else ""
    return cleaned
