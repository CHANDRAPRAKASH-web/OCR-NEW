# snippet to paste into src/api.py (replace existing /ocr/file and /ocr/url handlers)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile, shutil, os, logging, requests

from .pipeline import process_image
from .parser import parse_contact_fields

app = FastAPI(title="OCR Text Detection API", version="1.0")

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
    if "image" not in content_type:
        raise HTTPException(status_code=400, detail="URL does not appear to be an image.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".img")
    with tmp as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return tmp.name

@app.post("/ocr/file")
async def ocr_file(file: UploadFile = File(...)):
    tmp_path = save_upload_tmp(file)
    out_dir = tempfile.mkdtemp(prefix="ocr_out_")
    try:
        resp = process_image(tmp_path, out_dir)
        # If pipeline returned dict with parsed already, use it; else compute parsed
        if isinstance(resp, dict):
            results = resp.get("results", []) or []
            parsed = resp.get("parsed")
            if parsed is None:
                parsed = parse_contact_fields(results)
        else:
            results = resp or []
            parsed = parse_contact_fields(results)
        final = {
            "image": os.path.basename(tmp_path),
            "results": results,
            "parsed": parsed
        }
        return JSONResponse(final)
    except Exception as e:
        logging.exception("Processing failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.post("/ocr/url")
async def ocr_url(payload: dict):
    if "url" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'url' in payload")
    tmp_path = download_image_to_tmp(payload["url"])
    out_dir = tempfile.mkdtemp(prefix="ocr_out_")
    try:
        resp = process_image(tmp_path, out_dir)
        if isinstance(resp, dict):
            results = resp.get("results", []) or []
            parsed = resp.get("parsed")
            if parsed is None:
                parsed = parse_contact_fields(results)
        else:
            results = resp or []
            parsed = parse_contact_fields(results)
        final = {
            "image": os.path.basename(tmp_path),
            "results": results,
            "parsed": parsed
        }
        return JSONResponse(final)
    except Exception as e:
        logging.exception("Processing failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
