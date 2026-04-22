"""
interactive_predict.py — Load the model once, predict as many images as you want
──────────────────────────────────────────────────────────────────────────────────
Usage:
    python interactive_predict.py
    python interactive_predict.py --checkpoint outputs/best_model.pth

The script will:
  1. Load the model from the checkpoint (defaults to outputs/best_model.pth)
  2. Ask you to enter image paths one by one
  3. Print the predicted Khmer text for each image
  4. Type 'quit' or press Ctrl+C to exit
"""

import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.vocab import NUM_CLASSES, ctc_decode
from data.dataset import get_transforms, resize_to_height
from models.crnn import KhmerOCR


# ── Load model ────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        print("  → Train the model first:  python train.py --train_parquet data/train.parquet --val_parquet data/val.parquet")
        sys.exit(1)

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

def predict(model, image_path: str, img_height: int, device: torch.device) -> str:
    if not os.path.exists(image_path):
        return f"[Error] File not found: {image_path}"

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"[Error] Could not open image: {e}"

    transform = get_transforms(img_height, augment=False)
    img = resize_to_height(img, img_height)
    tensor = transform(img).unsqueeze(0).to(device)   # (1, 1, H, W)

    with torch.no_grad():
        logits  = model(tensor)                        # (T, 1, C)
        indices = logits.argmax(dim=2).squeeze(1)      # (T,)
        text    = ctc_decode(indices.tolist())

    return text if text.strip() else "[empty prediction — image may be too short or blank]"


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Khmer OCR — interactive prediction")
    p.add_argument(
        "--checkpoint",
        default="outputs/best_model.pth",
        help="Path to trained model checkpoint (default: outputs/best_model.pth)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print("  Khmer OCR — Interactive Prediction")
    print("=" * 55)
    print(f"  Loading checkpoint: {args.checkpoint}")

    model, img_height = load_model(args.checkpoint, device)

    print(f"  Device     : {device}")
    print(f"  Image height: {img_height}px")
    print("=" * 55)
    print("  Enter an image path to predict.")
    print("  Type 'quit' or press Ctrl+C to exit.")
    print("=" * 55 + "\n")

    while True:
        try:
            image_path = input("Image path: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if image_path.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        if not image_path:
            continue

        # Strip accidental quotes (e.g. if user drags file into terminal)
        image_path = image_path.strip("'\"")

        result = predict(model, image_path, img_height, device)
        print(f"Prediction : {result}\n")


if __name__ == "__main__":
    main()