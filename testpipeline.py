import os
import tempfile
from src.pipeline import process_image
from src.utils import ensure_dir
import pytest
import cv2
import numpy as np

def make_sample_image(path):
    img = np.ones((200,400,3), dtype=np.uint8)*255
    cv2.putText(img, "TEST 123", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
    cv2.imwrite(path, img)

def test_process_image_basic(tmp_path):
    inp = tmp_path / "sample_images"
    out = tmp_path / "outputs"
    inp.mkdir()
    out.mkdir()
    img_p = inp / "t1.png"
    make_sample_image(str(img_p))
    results = process_image(str(img_p), str(out))
    # basic assertions: results is list
    assert isinstance(results, list)
    # at least one detected block (pytesseract fallback should find something)
    assert len(results) >= 1
    # each entry has required keys
    for r in results:
        assert "text_clean" in r
        assert "box" in r
