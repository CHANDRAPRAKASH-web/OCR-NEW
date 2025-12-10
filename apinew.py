from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import tempfile
import cv2
import numpy as np
from .pipeline import process_image
from .utils import ensure_dir, load_config
import logging
import requests

app = FastAPI(title="OCR Text Detection API", version="1.0")

# allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = load_config()

@app.get("/")
def root():
    return {"message": "OCR Text Detection API. See /docs for Swagger UI."}

def save_upload_tmp(file_obj: UploadFile) -> str:
    suffix = os.path.splitext(file_obj.filename)[1] if file_obj.filename else ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        with tmp as f:
            shutil.copyfileobj(file_obj.file, f)
    finally:
        file_obj.file.close()
    return tmp.name

def download_image_to_tmp(url: str) -> str:
    r = requests.get(url, stream=True, timeout=15)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {r.status_code}")
    content_type = r.headers.get("content-type","")
    # allow basic image types
    if "image" not in content_type:
        raise HTTPException(status_code=400, detail="URL does not appear to be an image.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".img")
    with tmp as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return tmp.name

@app.post("/ocr/file")
async def ocr_file(file: UploadFile = File(...), request: Request = None):
    """
    Upload an image file. Returns JSON with OCR results.
    """
    # save file to temp path
    tmp_path = save_upload_tmp(file)
    out_dir = tempfile.mkdtemp(prefix="ocr_out_")
    try:
        results = process_image(tmp_path, out_dir)
        return JSONResponse({"image": os.path.basename(tmp_path), "results": results})
    except Exception as e:
        logging.exception("Processing failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # cleanup uploaded tmp file (keep out_dir for debugging)
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.post("/ocr/url")
async def ocr_url(payload: dict):
    """
    POST JSON: {"url":"https://.../image.jpg"}
    """
    if "url" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'url' in payload")
    tmp_path = download_image_to_tmp(payload["url"])
    out_dir = tempfile.mkdtemp(prefix="ocr_out_")
    try:
        results = process_image(tmp_path, out_dir)
        return JSONResponse({"image": os.path.basename(tmp_path), "results": results})
    except Exception as e:
        logging.exception("Processing failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    # dev server
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
