"""
Dataset loader for Khmer OCR.
Expects a .parquet file with columns:
  - 'image'  : raw image bytes (or PIL image)
  - 'text'   : ground-truth Khmer string

Khmer-only mode (default):
  - Rows with NO Khmer characters at all are dropped
  - Non-Khmer characters in labels are stripped out
  - The model only learns to predict Khmer
"""

import io
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ── Khmer Unicode range ───────────────────────────────────────────────────────
# U+1780 to U+17FF covers all Khmer characters, digits, punctuation
KHMER_RANGE = (0x1780, 0x17FF)

def is_khmer(ch: str) -> bool:
    return KHMER_RANGE[0] <= ord(ch) <= KHMER_RANGE[1]

def strip_to_khmer(text: str) -> str:
    """Keep only Khmer characters and spaces."""
    return "".join(ch for ch in text if is_khmer(ch) or ch == " ").strip()

def has_khmer(text: str) -> bool:
    """Return True if the text contains at least one Khmer character."""
    return any(is_khmer(ch) for ch in text)


# ── Image transforms ──────────────────────────────────────────────────────────

def get_transforms(img_height: int = 32, augment: bool = False):
    ops = []
    if augment:
        ops += [
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ]
    ops += [
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ]
    return T.Compose(ops)


def resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
    w, h = img.size
    new_w = max(1, int(w * target_h / h))
    return img.resize((new_w, target_h), Image.BICUBIC)


# ── Dataset ───────────────────────────────────────────────────────────────────

class KhmerOCRDataset(Dataset):
    """
    Args:
        parquet_path  : path to the .parquet file
        char2idx      : character → index mapping (from vocab.py)
        img_height    : fixed image height after resize
        augment       : apply data augmentation (train only)
        khmer_only    : if True, strip non-Khmer chars and drop rows with
                        no Khmer content (default: True)
    """

    def __init__(
        self,
        parquet_path: str,
        char2idx: dict,
        img_height: int = 32,
        augment: bool = False,
        khmer_only: bool = True,
    ):
        self.df = pd.read_parquet(parquet_path)
        self.char2idx = char2idx
        self.img_height = img_height
        self.transform = get_transforms(img_height, augment)
        self.khmer_only = khmer_only

        original_len = len(self.df)

        if khmer_only:
            # Step 1: Drop rows with no Khmer characters at all
            mask = self.df["text"].astype(str).apply(has_khmer)
            dropped_no_khmer = (~mask).sum()
            self.df = self.df[mask].reset_index(drop=True)

            # Step 2: Strip non-Khmer characters from labels (keep spaces too)
            self.df["text"] = self.df["text"].astype(str).apply(strip_to_khmer)

            # Step 3: Drop any rows that became empty after stripping
            mask2 = self.df["text"].str.len() > 0
            dropped_empty = (~mask2).sum()
            self.df = self.df[mask2].reset_index(drop=True)

            total_dropped = original_len - len(self.df)
            print(f"[Dataset] {original_len} rows → kept {len(self.df)} Khmer rows "
                  f"(dropped {total_dropped} non-Khmer/empty rows)")
        else:
            # Original behavior: drop rows with any out-of-vocab characters
            known = set(char2idx.keys())
            mask = self.df["text"].apply(lambda t: all(ch in known for ch in str(t)))
            dropped = (~mask).sum()
            if dropped:
                print(f"[Dataset] Dropped {dropped} rows with out-of-vocab characters.")
            self.df = self.df[mask].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Load image ──────────────────────────────────────────────────────
        img_data = row["image"]
        if isinstance(img_data, dict) and "bytes" in img_data:
            img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        elif isinstance(img_data, bytes):
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
        elif isinstance(img_data, Image.Image):
            img = img_data.convert("RGB")
        else:
            raise ValueError(f"Unsupported image format at row {idx}: {type(img_data)}")

        img = resize_to_height(img, self.img_height)
        img_tensor = self.transform(img)

        # ── Encode label ────────────────────────────────────────────────────
        text = str(row["text"])
        label = torch.tensor(
            [self.char2idx[ch] for ch in text if ch in self.char2idx],
            dtype=torch.long,
        )

        return img_tensor, label, text


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch):
    images, labels, texts = zip(*batch)

    max_w = max(img.shape[-1] for img in images)
    padded = torch.zeros(len(images), 1, images[0].shape[-2], max_w)
    for i, img in enumerate(images):
        padded[i, :, :, : img.shape[-1]] = img

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_cat    = torch.cat(labels)

    return padded, labels_cat, label_lengths, texts


# ── Convenience factory ───────────────────────────────────────────────────────

def build_dataloader(
    parquet_path: str,
    char2idx: dict,
    batch_size: int = 32,
    img_height: int = 32,
    augment: bool = False,
    shuffle: bool = True,
    num_workers: int = 2,
    khmer_only: bool = True,
) -> DataLoader:
    dataset = KhmerOCRDataset(
        parquet_path, char2idx, img_height, augment, khmer_only
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,   # disabled — no GPU detected
    )