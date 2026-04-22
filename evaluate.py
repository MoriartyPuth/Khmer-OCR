"""
evaluate.py — Evaluate a trained Khmer OCR model
─────────────────────────────────────────────────────────
Usage:
    python evaluate.py \
        --checkpoint  outputs/best_model.pth \
        --val_parquet data/val.parquet

Prints:
  - Average Character Error Rate (CER)
  - Average Word Error Rate (WER)
  - Sample predictions vs ground truth
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.vocab import CHAR2IDX, IDX2CHAR, NUM_CLASSES, ctc_decode
from data.dataset import build_dataloader
from models.crnn import KhmerOCR


# ── Metrics ───────────────────────────────────────────────────────────────────

def levenshtein(a, b):
    """Edit distance between two sequences."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]


def cer(pred: str, gt: str) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gt) / len(gt)


def wer(pred: str, gt: str) -> float:
    p_words = pred.split()
    g_words = gt.split()
    if not g_words:
        return 0.0 if not p_words else 1.0
    return levenshtein(p_words, g_words) / len(g_words)


def greedy_decode(logits):
    """logits: (T, B, C) → list of strings"""
    indices = logits.argmax(dim=2).permute(1, 0)  # (B, T)
    return [ctc_decode(row.tolist()) for row in indices]


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Khmer OCR")
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--val_full_parquet",   required=True)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--num_samples",   type=int, default=10,
                   help="How many sample predictions to print")
    p.add_argument("--num_workers",   type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    img_height  = saved_args.get("img_height", 32)
    rnn_hidden  = saved_args.get("rnn_hidden", 256)
    rnn_layers  = saved_args.get("rnn_layers", 2)

    model = KhmerOCR(NUM_CLASSES, rnn_hidden, rnn_layers).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    print(f"Checkpoint val CER: {ckpt.get('val_cer', 'N/A'):.4f}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    loader = build_dataloader(
        args.val_full_parquet, CHAR2IDX,
        batch_size=args.batch_size, img_height=img_height,
        augment=False, shuffle=False, num_workers=args.num_workers,
    )

    # ── Evaluation loop ───────────────────────────────────────────────────────
    total_cer = 0.0
    total_wer = 0.0
    total_samples = 0
    printed = 0

    with torch.no_grad():
        for images, labels, label_lengths, texts in loader:
            images = images.to(device)
            logits = model(images)             # (T, B, C)
            preds  = greedy_decode(logits.cpu())

            for pred, gt in zip(preds, texts):
                total_cer += cer(pred, gt)
                total_wer += wer(pred, gt)
                total_samples += 1

                # Print sample predictions
                if printed < args.num_samples:
                    print(f"GT  : {gt}")
                    print(f"PRED: {pred}")
                    print(f"CER : {cer(pred, gt):.4f}")
                    print()
                    printed += 1

    avg_cer = total_cer / total_samples
    avg_wer = total_wer / total_samples

    print("=" * 40)
    print(f"Test samples : {total_samples}")
    print(f"Avg CER      : {avg_cer:.4f}  ({avg_cer*100:.2f}%)")
    print(f"Avg WER      : {avg_wer:.4f}  ({avg_wer*100:.2f}%)")


if __name__ == "__main__":
    main()
