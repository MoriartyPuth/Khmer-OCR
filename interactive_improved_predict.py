"""
Interactive Document OCR with Improved Preprocessing
───────────────────────────────────────────────────────────────
Enhanced version with padding, deskewing, and confidence filtering.

Usage:
    python interactive_improved_predict.py --checkpoint outputs/best_model.pth

Parameters you can adjust for your document:
    --padding_tb 8              # Vertical padding (5-10 for Khmer, default 8)
    --deskew                    # Enable deskewing (recommended)
    --confidence_threshold 0.6  # Reject predictions below this (0.5-0.7)
    --threshold 127             # Segmentation binary threshold (100-150)
"""

import argparse
import os
import sys
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from improved_document_predict import (
    load_model,
    predict_document_improved,
)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Document OCR with improved preprocessing"
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device",
    )
    parser.add_argument(
        "--padding_tb",
        type=int,
        default=8,
        help="Vertical padding for lines (5-10 for Khmer)",
    )
    parser.add_argument(
        "--padding_lr",
        type=int,
        default=2,
        help="Horizontal padding for lines",
    )
    parser.add_argument(
        "--deskew",
        action="store_true",
        default=True,
        help="Enable line deskewing",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.60,
        help="Minimum confidence to accept prediction (0.0-1.0)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Binary threshold for line segmentation",
    )
    parser.add_argument(
        "--min_gap",
        type=int,
        default=3,
        help="Minimum gap between lines",
    )
    parser.add_argument(
        "--min_height",
        type=int,
        default=5,
        help="Minimum line height",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[Setup] Device: {device}")

    # Load model
    print(f"[Loading] Model from {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"[Error] Checkpoint not found: {args.checkpoint}")
        print("  → Train the model first: python train.py ...")
        sys.exit(1)

    model, img_height = load_model(args.checkpoint, device)
    print(f"✓ Model loaded (img_height={img_height})")

    # Print configuration
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"  Vertical padding:       {args.padding_tb}px (Khmer: 5-10px)")
    print(f"  Horizontal padding:     {args.padding_lr}px")
    print(f"  Deskewing:              {'✓ Enabled' if args.deskew else '✗ Disabled'}")
    print(f"  Confidence threshold:   {args.confidence_threshold:.2f}")
    print(f"  Segmentation threshold: {args.threshold} (0-255)")
    print("=" * 70)

    # Interactive loop
    print("\n[Ready] Enter document image paths (or 'quit' to exit):\n")

    while True:
        try:
            doc_path = input("📄 Document path: ").strip()

            if doc_path.lower() in ("quit", "exit", "q"):
                print("Exiting...")
                break

            if not doc_path:
                print("  (empty path, try again)")
                continue

            if not os.path.exists(doc_path):
                print(f"  [Error] File not found")
                continue

            # Predict
            print("\n[Processing...]")
            full_text, line_texts, diagnostics = predict_document_improved(
                model,
                doc_path,
                img_height,
                device,
                threshold=args.threshold,
                min_gap=args.min_gap,
                min_height=args.min_height,
                padding_top_bottom=args.padding_tb,
                padding_left_right=args.padding_lr,
                deskew=args.deskew,
                confidence_threshold=args.confidence_threshold,
                verbose=True,
                diagnostics=True,
            )

            # Show summary
            print("\n" + "-" * 70)
            print("TEXT OUTPUT:")
            print("-" * 70)
            print(full_text if full_text.strip() else "[No text detected]")

            # Save to file
            output_file = doc_path.replace(".png", "_output.txt").replace(".jpg", "_output.txt")
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(full_text)
                print(f"\n💾 Saved to: {output_file}")
            except:
                pass

            print()

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"  [Error] {e}")
            continue


if __name__ == "__main__":
    main()
