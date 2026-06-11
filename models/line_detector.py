"""
Text-Line Detector (Model 2) — lightweight U-Net
─────────────────────────────────────────────────────────────────────────────
Detect-then-recognize architecture:

    Full document ──► LineDetectorUNet ──► text mask ──► line boxes
                                                              │ crop each box
                                            KhmerOCR (CRNN) ◄─┘  Model 1

The U-Net predicts a per-pixel text/background mask. Unlike projection or
connected-component heuristics, it LEARNS what text looks like, so colored
certificate backgrounds, watermarks, and seals don't break it.

Size: ~1.9M params (≈7.5 MB fp32) — fits easily on Streamlit Cloud.

Inference flow (`detect_lines`):
  1. min-channel grayscale, resize longest side to model size, pad to square
  2. forward pass → sigmoid mask
  3. threshold mask, extract per-line boxes via connected components
  4. map boxes back to original image coordinates
"""

from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ── Building blocks ───────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2 — the standard U-Net block."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_c, out_c))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Bilinear upsample → concat skip → DoubleConv."""

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_c + skip_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if odd sizes produced a 1-px mismatch
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh or dw:
            x = nn.functional.pad(x, (0, dw, 0, dh))
        return self.conv(torch.cat([skip, x], dim=1))


# ── U-Net ─────────────────────────────────────────────────────────────────────

class LineDetectorUNet(nn.Module):
    """
    Input : (B, 1, H, W) grayscale in [0, 1]   (H, W multiples of 16)
    Output: (B, 1, H, W) raw logits — apply sigmoid for the text mask
    """

    def __init__(self, base: int = 24):
        super().__init__()
        self.inc   = DoubleConv(1, base)                 # 512
        self.down1 = Down(base,     base * 2)            # 256
        self.down2 = Down(base * 2, base * 4)            # 128
        self.down3 = Down(base * 4, base * 8)            #  64
        self.down4 = Down(base * 8, base * 8)            #  32 (bottleneck)

        self.up1 = Up(base * 8, base * 8, base * 4)
        self.up2 = Up(base * 4, base * 4, base * 2)
        self.up3 = Up(base * 2, base * 2, base)
        self.up4 = Up(base,     base,     base)
        self.out = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.out(x)


# ── Loss (training) ───────────────────────────────────────────────────────────

class BCEDiceLoss(nn.Module):
    """BCE handles per-pixel accuracy; Dice handles class imbalance
    (text pixels are a small fraction of the page)."""

    def __init__(self, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        bce  = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice  = 1 - ((2 * inter + 1.0) / (union + 1.0)).mean()
        return bce + self.dice_weight * dice


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def save_detector(model: LineDetectorUNet, path: str, base: int = 24, **extra):
    torch.save({"model_state_dict": model.state_dict(),
                "base": base, **extra}, path)


def load_detector(path: str, device: torch.device) -> LineDetectorUNet:
    ckpt  = torch.load(path, map_location=device)
    model = LineDetectorUNet(base=ckpt.get("base", 24)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── Inference: document → line boxes ─────────────────────────────────────────

def _to_min_channel(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB")).min(axis=2).astype(np.uint8)


def predict_mask(
    model: LineDetectorUNet,
    img: Image.Image,
    device: torch.device,
    input_size: int = 512,
) -> np.ndarray:
    """
    Run the U-Net on a document image.

    Returns the probability mask at ORIGINAL image resolution (H, W) float32.
    """
    gray = _to_min_channel(img)
    oh, ow = gray.shape

    scale  = input_size / max(oh, ow)
    nh, nw = max(16, int(oh * scale)), max(16, int(ow * scale))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

    # pad to input_size square (network needs /16-divisible dims)
    padded = np.zeros((input_size, input_size), dtype=np.uint8)
    padded[:nh, :nw] = resized

    x = torch.from_numpy(padded.astype(np.float32) / 255.0)
    x = x.unsqueeze(0).unsqueeze(0).to(device)           # (1, 1, S, S)

    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()

    prob = prob[:nh, :nw]                                # drop padding
    return cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_LINEAR)


def extract_line_boxes(
    mask: np.ndarray,
    prob_threshold: float = 0.5,
    min_height: int = 6,
    min_width_frac: float = 0.02,
) -> List[Tuple[int, int, int, int]]:
    """
    Convert a probability mask into text-line boxes (x0, y0, x1, y1),
    sorted top-to-bottom.

    The mask is clean (the model was trained on solid line rectangles), so a
    light horizontal close + connected components separates lines reliably.
    """
    h, w = mask.shape
    binary = (mask > prob_threshold).astype(np.uint8)

    # close small horizontal gaps (between words) without joining lines
    kw = max(10, w // 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []
    for i in range(1, num):
        bx, by, bw, bh, area = stats[i, :5]
        if bh < min_height or bw < w * min_width_frac:
            continue
        boxes.append((int(bx), int(by), int(bx + bw), int(by + bh)))

    boxes.sort(key=lambda b: b[1])

    # merge boxes whose vertical ranges overlap >60% (same line split by a gap)
    merged: List[List[int]] = []
    for b in boxes:
        if merged:
            m = merged[-1]
            ov = min(m[3], b[3]) - max(m[1], b[1])
            if ov > 0.6 * min(m[3] - m[1], b[3] - b[1]):
                m[0] = min(m[0], b[0]); m[1] = min(m[1], b[1])
                m[2] = max(m[2], b[2]); m[3] = max(m[3], b[3])
                continue
        merged.append(list(b))

    return [tuple(m) for m in merged]


def detect_lines(
    model: LineDetectorUNet,
    img: Image.Image,
    device: torch.device,
    input_size: int = 512,
    prob_threshold: float = 0.5,
) -> List[Tuple[int, int, int, int]]:
    """Full document → sorted list of (x0, y0, x1, y1) line boxes."""
    mask = predict_mask(model, img, device, input_size)
    return extract_line_boxes(mask, prob_threshold)


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = LineDetectorUNet()
    dummy = torch.randn(2, 1, 512, 512)
    out   = model(dummy)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Input : {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Trainable params: {n:,}  (~{n * 4 / 1e6:.1f} MB fp32)")
