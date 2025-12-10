import argparse
import os
from .pipeline import process_image
from .utils import ensure_dir, load_config
import logging

cfg = load_config()

def main():
    parser = argparse.ArgumentParser(description="OCR Text Detection - pipeline runner")
    parser.add_argument("--input_dir", required=True, help="Directory with images to process")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    images = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    if not images:
        logging.error("No images found in input_dir")
        return

    for img_path in images:
        logging.info(f"Processing: {img_path}")
        try:
            res = process_image(img_path, args.output_dir)
            logging.info(f"Found {len(res)} text elements in {img_path}")
        except Exception as e:
            logging.exception(f"Failed to process {img_path}: {e}")

if __name__ == "__main__":
    main()
