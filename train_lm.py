"""
Build the Khmer character language model from OCR training labels.
─────────────────────────────────────────────────────────────────────────────
The LM is used at inference to re-rank the CRNN's beam-search hypotheses, fixing
look-alike glyph confusions (្ត↔្ទ, ន↔ណ) and dropped characters.

    python train_lm.py --parquet data/hf_lines/data/train-00001.parquet
    python train_lm.py --parquet a.parquet b.parquet --order 4 --out outputs/khmer_lm.pkl

Output: outputs/khmer_lm.pkl  (committed so deployment needs no rebuild)
"""

import os
import argparse

import pandas as pd

from utils.khmer_lm import KhmerCharLM


def iter_texts(parquets):
    for pq in parquets:
        df = pd.read_parquet(pq, columns=["text"])
        for t in df["text"].astype(str):
            if t and t != "nan":
                yield t


def main():
    ap = argparse.ArgumentParser(description="Build Khmer char n-gram LM")
    ap.add_argument("--parquet", nargs="+", required=True,
                    help="One or more label parquets (need a 'text' column)")
    ap.add_argument("--order", type=int, default=4, help="n-gram order")
    ap.add_argument("--out", default="outputs/khmer_lm.pkl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[LM] Building {args.order}-gram model from {len(args.parquet)} file(s)…")
    lm = KhmerCharLM(n=args.order)
    n_lines = 0
    for t in iter_texts(args.parquet):
        lm.add_text(t)
        n_lines += 1
        if n_lines % 10000 == 0:
            print(f"  {n_lines:,} lines…")

    print(f"[LM] Counted {n_lines:,} lines — {lm.size_info()}")
    lm.prune()
    print(f"[LM] After pruning — {lm.size_info()}")

    lm.save(args.out)
    mb = os.path.getsize(args.out) / 1e6
    print(f"[LM] Saved {args.out}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
