"""
predict.py — Run inference with a trained Khmer OCR model
─────────────────────────────────────────────────────────
Usage (single image):
    python predict.py \
        --checkpoint outputs/best_model.pth \
        --image path/to/line_image.png

Usage (folder of images):
    python predict.py \
        --checkpoint outputs/best_model.pth \
        --image_dir  path/to/images/

Usage (from a parquet file, no labels needed):
    python predict.py \
        --checkpoint  outputs/best_model.pth \
        --parquet     data/unlabeled.parquet
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.vocab import CHAR2IDX, IDX2CHAR, NUM_CLASSES, ctc_decode
from data.dataset import get_transforms, resize_to_height
from models.crnn import KhmerOCR
from utils.improved_line_segmentation import segment_document


# ── Load model ────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved = ckpt.get("args", {})
    img_height = saved.get("img_height", 32)
    rnn_hidden  = saved.get("rnn_hidden", 256)
    rnn_layers  = saved.get("rnn_layers", 2)

    model = KhmerOCR(NUM_CLASSES, rnn_hidden, rnn_layers).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, img_height


# ── Predict one image ─────────────────────────────────────────────────────────

def predict_image(model, img: Image.Image, img_height: int, device: torch.device) -> str:
    transform = get_transforms(img_height, augment=False)
    img = resize_to_height(img.convert("RGB"), img_height)
    tensor = transform(img).unsqueeze(0).to(device)   # (1, 1, H, W)

    with torch.no_grad():
        logits = model(tensor)                        # (T, 1, C)
        indices = logits.argmax(dim=2).squeeze(1)     # (T,)
        text = ctc_decode(indices.tolist())

    return text


# ── Predict from parquet ──────────────────────────────────────────────────────

def predict_from_parquet(model, parquet_path: str, img_height: int, device: torch.device):
    import io
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    results = []

    for i, row in df.iterrows():
        img_data = row["image"]
        if isinstance(img_data, dict) and "bytes" in img_data:
            img = Image.open(io.BytesIO(img_data["bytes"]))
        elif isinstance(img_data, bytes):
            img = Image.open(io.BytesIO(img_data))
        else:
            img = img_data

        pred = predict_image(model, img, img_height, device)
        results.append(pred)
        print(f"[{i}] {pred}")

    return results


# ── Predict full document ────────────────────────────────────────────────────

def predict_document(
    model,
    img: Image.Image,
    img_height: int,
    device: torch.device,
    threshold: int = 127,
    min_gap: int = 3,
    min_height: int = 5,
    expand_margin: int = 2,
) -> str:
    """
    Predict text from a full document image by segmenting it into lines.
    
    Args:
        model: Trained KhmerOCR model
        img: Document image (PIL Image)
        img_height: Target height for line normalization
        device: Torch device (cpu or cuda)
        threshold: Binary threshold for line segmentation
        min_gap: Minimum gap between lines (rows)
        min_height: Minimum line height (pixels)
        expand_margin: Margin to expand around detected lines
    
    Returns:
        Full document text with lines separated by newlines
    """
    # Segment document into lines
    line_images, line_bounds = segment_document(
        img,
        threshold=threshold,
        min_gap=min_gap,
        min_height=min_height,
        expand_margin=expand_margin,
    )
    
    if not line_images:
        print("[Warning] No lines detected in document")
        return ""
    
    print(f"[Document] Detected {len(line_images)} lines")
    
    # Predict each line
    line_texts = []
    for i, line_img in enumerate(line_images):
        line_text = predict_image(model, line_img, img_height, device)
        line_texts.append(line_text)
        print(f"  Line {i+1}: {line_text}")
    
    # Combine lines with newlines
    document_text = "\n".join(line_texts)
    return document_text


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Khmer OCR Inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image",      help="Path to a single image file")
    p.add_argument("--document",   help="Path to document image (full page)")
    p.add_argument("--image_dir",  help="Directory of image files")
    p.add_argument("--parquet",    help="Parquet file with image column")
    p.add_argument("--threshold",  type=int, default=127, help="Binary threshold for document segmentation")
    p.add_argument("--min_gap",    type=int, default=3, help="Minimum gap between lines")
    p.add_argument("--min_height", type=int, default=5, help="Minimum line height")
    p.add_argument("--expand",     type=int, default=2, help="Expand lines by N pixels")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, img_height = load_model(args.checkpoint, device)
    print(f"Model loaded | img_height={img_height} | device={device}\n")

    if args.image:
        img  = Image.open(args.image)
        pred = predict_image(model, img, img_height, device)
        print(f"Prediction: {pred}")

    elif args.document:
        img = Image.open(args.document)
        pred = predict_document(
            model, img, img_height, device,
            threshold=args.threshold,
            min_gap=args.min_gap,
            min_height=args.min_height,
            expand_margin=args.expand,
        )
        print(f"\n[Document Prediction]\n{pred}")

    elif args.image_dir:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        paths = [p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in exts]
        paths.sort()
        for path in paths:
            img  = Image.open(path)
            pred = predict_image(model, img, img_height, device)
            print(f"{path.name}: {pred}")

    elif args.parquet:
        predict_from_parquet(model, args.parquet, img_height, device)

    else:
        print("Please provide --image, --document, --image_dir, or --parquet")


if __name__ == "__main__":
    main()
