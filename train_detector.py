"""
Train the Text-Line Detector (Model 2 — U-Net)
─────────────────────────────────────────────────────────────────────────────
Uses the SAME single-line parquet as Model 1. Documents are synthesized
on the fly, so every batch is unique — no annotation work, no disk dataset.

    python train_detector.py --parquet data/train.parquet --epochs 30

Outputs:
    outputs/line_detector.pth   (best validation Dice)

Typical timing: ~30 min on GPU, a few hours on CPU at the default sizes.
Reduce --epoch_size for a faster (rougher) model.
"""

import os
import time
import argparse

import torch
from torch.utils.data import DataLoader

from data.synthesize_documents import SyntheticDocumentDataset
from models.line_detector import LineDetectorUNet, BCEDiceLoss, save_detector


def dice_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Dice over the batch (higher = better, 1.0 = perfect overlap)."""
    prob = (torch.sigmoid(logits) > 0.5).float()
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + 1.0) / (union + 1.0)).mean().item()


def main():
    p = argparse.ArgumentParser(description="Train the U-Net line detector")
    p.add_argument("--parquet",     required=True, help="Single-line OCR parquet (same as Model 1)")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--epoch_size",  type=int,   default=1500, help="Synthetic docs per epoch")
    p.add_argument("--val_size",    type=int,   default=150,  help="Fixed validation set size")
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--canvas_size", type=int,   default=512)
    p.add_argument("--base",        type=int,   default=24, help="U-Net base channel count")
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--output_dir",  default="outputs")
    p.add_argument("--resume",      action="store_true", help="Resume from line_detector.pth")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "line_detector.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── Data — train is fully random, val is seeded (same 150 docs each epoch) ──
    train_ds = SyntheticDocumentDataset(
        args.parquet, epoch_size=args.epoch_size,
        canvas_size=args.canvas_size, seed=None,
    )
    val_ds = SyntheticDocumentDataset(
        args.parquet, epoch_size=args.val_size,
        canvas_size=args.canvas_size, seed=12345,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda")
    print(f"[Data] {len(train_ds)} synthetic docs/epoch (train), {len(val_ds)} fixed (val)")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = LineDetectorUNet(base=args.base).to(device)
    if args.resume and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Resume] Loaded {best_path}")

    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"[Model] U-Net base={args.base}  params={n_params:,}")

    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        t0, total_loss = time.time(), 0.0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        train_loss = total_loss / len(train_ds)
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_dice, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, masks).item() * imgs.size(0)
                val_dice += dice_score(logits, masks) * imgs.size(0)
                n_val    += imgs.size(0)
        val_loss /= n_val
        val_dice /= n_val

        marker = ""
        if val_dice > best_dice:
            best_dice = val_dice
            save_detector(model, best_path, base=args.base,
                          epoch=epoch, val_dice=val_dice,
                          canvas_size=args.canvas_size)
            marker = "  ← saved"

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train {train_loss:.4f} | val {val_loss:.4f} | "
              f"dice {val_dice:.4f} | {time.time() - t0:.0f}s{marker}")

    print(f"\n[Done] Best val Dice: {best_dice:.4f}")
    print(f"[Checkpoint] {best_path}")
    print("\nNext steps:")
    print("  1. Test locally:  the app auto-detects outputs/line_detector.pth")
    print("  2. Deploy: upload line_detector.pth to the Hugging Face repo "
          "(MoriartyPuth/khmer-ocr-model) so Streamlit Cloud downloads it")


if __name__ == "__main__":
    main()
