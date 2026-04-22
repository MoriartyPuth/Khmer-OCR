"""
train.py — Train the Khmer OCR model
─────────────────────────────────────────────────────────
Usage:
    python train.py \
        --train_parquet data/train.parquet \
        --val_parquet   data/val.parquet \
        --epochs        50 \
        --batch_size    32 \
        --output_dir    outputs/

The script will:
  1. Train for N epochs
  2. Evaluate on validation set after every epoch
  3. Save the best checkpoint (lowest val CER)
  4. Print a simple training log
"""

import argparse
import os
import sys

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.vocab import CHAR2IDX, IDX2CHAR, NUM_CLASSES, ctc_decode
from data.dataset import build_dataloader
from models.crnn import KhmerOCR, CTCLoss


# ── Metrics ───────────────────────────────────────────────────────────────────

def character_error_rate(pred: str, gt: str) -> float:
    """Simple CER using edit distance."""
    import Levenshtein
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return Levenshtein.distance(pred, gt) / len(gt)


def greedy_decode_batch(logits, idx2char):
    """
    logits : (T, B, C)
    Returns list of decoded strings, one per batch item.
    """
    # Argmax over classes at each time step
    indices = logits.argmax(dim=2).permute(1, 0)  # (B, T)
    results = []
    for row in indices:
        results.append(ctc_decode(row.tolist()))
    return results


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_cer  = 0.0
    n_batches  = 0

    with torch.set_grad_enabled(train):
        for images, labels, label_lengths, texts in loader:
            images        = images.to(device)
            labels        = labels.to(device)
            label_lengths = label_lengths.to(device)

            logits = model(images)                        # (T, B, C)
            loss   = criterion(logits, labels, label_lengths)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            # CER on this batch
            preds = greedy_decode_batch(logits.detach().cpu(), IDX2CHAR)
            batch_cer = sum(
                character_error_rate(p, g) for p, g in zip(preds, texts)
            ) / len(texts)

            total_loss += loss.item()
            total_cer  += batch_cer
            n_batches  += 1

    return total_loss / n_batches, total_cer / n_batches


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Khmer OCR")
    p.add_argument("--resume", action="store_true", help="Resume from best_model.pth")
    p.add_argument("--train_parquet", required=True)
    p.add_argument("--val_parquet",   required=True)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--img_height",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--rnn_hidden",   type=int,   default=256)
    p.add_argument("--rnn_layers",   type=int,   default=2)
    p.add_argument("--output_dir",   default="outputs")
    p.add_argument("--num_workers",  type=int,   default=2)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Vocab  : {NUM_CLASSES} classes\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader = build_dataloader(
        args.train_parquet, CHAR2IDX,
        batch_size=args.batch_size, img_height=args.img_height,
        augment=True, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        args.val_parquet, CHAR2IDX,
        batch_size=args.batch_size, img_height=args.img_height,
        augment=False, shuffle=False, num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = KhmerOCR(
        num_classes=NUM_CLASSES,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
    ).to(device)

    criterion = CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_cer  = float("inf")
    best_path = os.path.join(args.output_dir, "best_model.pth")

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train CER':>10} | {'Val Loss':>9} | {'Val CER':>8}")
    print("-" * 60)
    start_epoch = 1
    if args.resume:
        checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            
            # ADD THIS LINE:
            best_cer = checkpoint.get("val_cer", float("inf")) 
            
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming from epoch {start_epoch} (Best CER so far: {best_cer:.4f})")

    # Change your loop to use start_epoch
    for epoch in range(start_epoch, args.epochs + 1):
    #for epoch in range(1, args.epochs + 1):
        train_loss, train_cer = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_cer   = run_epoch(model, val_loader,   criterion, None,      device, train=False)

        scheduler.step(val_cer)

        # Save best checkpoint
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "val_cer":    val_cer,
                "args":       vars(args),
            }, best_path)
            marker = " ← best"
        else:
            marker = ""

        print(
            f"{epoch:>6} | {train_loss:>10.4f} | {train_cer:>9.4f} "
            f"| {val_loss:>9.4f} | {val_cer:>8.4f}{marker}"
        )

    print(f"\nTraining complete. Best val CER: {best_cer:.4f}")
    print(f"Best model saved to: {best_path}")

    # ── Final evaluation on val (used as test set) ────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL EVALUATION  (val.parquet as test set)")
    print("=" * 60)

    # Reload best checkpoint for final eval
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    total_cer = 0.0
    total_wer = 0.0
    total_samples = 0
    printed = 0
    NUM_SAMPLE_PRINTS = 5

    def _levenshtein(a, b):
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[:], i
            for j in range(1, n + 1):
                dp[j] = prev[j-1] if a[i-1] == b[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
        return dp[n]

    with torch.no_grad():
        for images, labels, label_lengths, texts in val_loader:
            images = images.to(device)
            logits = model(images)
            preds  = greedy_decode_batch(logits.cpu(), IDX2CHAR)

            for pred, gt in zip(preds, texts):
                c = _levenshtein(pred, gt) / max(len(gt), 1)
                w = _levenshtein(pred.split(), gt.split()) / max(len(gt.split()), 1)
                total_cer += c
                total_wer += w
                total_samples += 1

                if printed < NUM_SAMPLE_PRINTS:
                    print(f"  GT  : {gt}")
                    print(f"  PRED: {pred}")
                    print(f"  CER : {c:.4f}\n")
                    printed += 1

    print(f"Samples : {total_samples}")
    print(f"Avg CER : {total_cer/total_samples:.4f}  ({total_cer/total_samples*100:.2f}%)")
    print(f"Avg WER : {total_wer/total_samples:.4f}  ({total_wer/total_samples*100:.2f}%)")


if __name__ == "__main__":
    main()