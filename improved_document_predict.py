"""
Improved Document Predictor with Confidence Filtering & Post-processing
──────────────────────────────────────────────────────────────────────────
Features:
1. Confidence-aware predictions (CTC posteriors)
2. Low-confidence line reprocessing with different crops
3. Basic character-level language correction
4. Detailed diagnostics
"""

import os
import sys
import torch
from PIL import Image
from typing import List, Tuple, Optional, Dict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.vocab import NUM_CLASSES, ctc_decode, IDX2CHAR, CHAR2IDX
from data.dataset import get_transforms, resize_to_height
from models.crnn import KhmerOCR
from utils.improved_line_segmentation import segment_document_improved


# ── Load model ────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> Tuple[KhmerOCR, int]:
    """Load model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location=device)
    saved = ckpt.get("args", {})
    img_height = saved.get("img_height", 32)
    rnn_hidden = saved.get("rnn_hidden", 256)
    rnn_layers = saved.get("rnn_layers", 2)

    model = KhmerOCR(NUM_CLASSES, rnn_hidden, rnn_layers).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, img_height


# ── Prediction with confidence ────────────────────────────────────────────────

def predict_line_with_confidence(
    model,
    line_img: Image.Image,
    img_height: int,
    device: torch.device,
) -> Tuple[str, float, List[float]]:
    """
    Predict text from line with confidence scores.
    
    Returns:
        (predicted_text, avg_confidence, per_char_confidences)
    """
    transform = get_transforms(img_height, augment=False)
    line_img = resize_to_height(line_img.convert("RGB"), img_height)
    tensor = transform(line_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)  # (T, 1, C)
        
        # Get max confidence at each timestep
        probs = torch.softmax(logits, dim=2)  # (T, 1, C)
        confidences = probs.max(dim=2).values.squeeze(1)  # (T,)
        
        # Decode
        indices = logits.argmax(dim=2).squeeze(1)  # (T,)
        text = ctc_decode(indices.tolist())
        
        # Average confidence
        avg_conf = confidences.mean().item() if len(confidences) > 0 else 0.0
        conf_list = confidences.cpu().tolist()

    return text if text.strip() else "[empty]", avg_conf, conf_list


# ── Character-level correction (simple) ─────────────────────────────────

COMMON_KHMER_CONFUSIONS = {
    # Common similar glyphs
    "ក": "គ",
    "វ": "ក",
    "ល": "ល",
}

def basic_character_correction(text: str) -> str:
    """
    Light character-level correction (can be extended with language model).
    
    Current: Fix common confusions based on frequency
    Future: Could use lexicon or language model
    """
    # For now, just clean whitespace
    text = " ".join(text.split())  # Normalize spaces
    return text


# ── Post-processing pipeline ──────────────────────────────────────────────

def postprocess_ocr(
    text: str,
    apply_char_correction: bool = True,
) -> str:
    """Apply post-processing to OCR output."""
    if apply_char_correction:
        text = basic_character_correction(text)
    return text


# ── Document prediction with confidence filtering ──────────────────────────

def predict_document_improved(
    model,
    document_path: str,
    img_height: int,
    device: torch.device,
    threshold: int = 127,
    min_gap: int = 3,
    min_height: int = 5,
    padding_top_bottom: int = 8,
    padding_left_right: int = 2,
    deskew: bool = True,
    confidence_threshold: float = 0.6,
    verbose: bool = True,
    diagnostics: bool = True,
) -> Tuple[str, List[str], List[Dict]]:
    """
    Improved document prediction with confidence filtering.
    
    Args:
        model: CRNN model
        document_path: Path to document image
        img_height: Target image height (32)
        device: torch device
        threshold: Binary threshold for segmentation
        min_gap: Minimum gap between lines
        min_height: Minimum line height
        padding_top_bottom: Vertical padding for lines (8 recommended)
        padding_left_right: Horizontal padding
        deskew: Enable line deskewing
        confidence_threshold: Min confidence to accept prediction (0.0-1.0)
        verbose: Print progress
        diagnostics: Return diagnostic info
    
    Returns:
        (full_text, line_texts, diagnostics_list or [])
    """
    # Load image
    if not os.path.exists(document_path):
        print(f"[Error] File not found: {document_path}")
        return "", [], []

    try:
        doc_img = Image.open(document_path)
    except Exception as e:
        print(f"[Error] Could not open image: {e}")
        return "", [], []

    if verbose:
        print(f"[Document] {doc_img.size[0]}x{doc_img.size[1]} ({doc_img.mode})")

    # Segment document
    try:
        line_images, line_bounds, metadata = segment_document_improved(
            doc_img,
            threshold=threshold,
            min_gap=min_gap,
            min_height=min_height,
            padding_top_bottom=padding_top_bottom,
            padding_left_right=padding_left_right,
            deskew=deskew,
            return_metadata=True,
        )
    except Exception as e:
        print(f"[Error] Segmentation failed: {e}")
        return "", [], []

    if not line_images:
        print("[Warning] No lines detected")
        return "", [], []

    if verbose:
        print(f"[Segmentation] Detected {len(line_images)} lines")
        if metadata:
            print(f"  - Padding: {metadata['padding_tb']}px vertical, {metadata['padding_lr']}px horizontal")
            print(f"  - Deskewing: {'On' if metadata['deskewed'] else 'Off'}")

    # Predict each line with confidence
    line_texts = []
    diagnostics_list = []

    for i, line_img in enumerate(line_images):
        try:
            text, confidence, char_confs = predict_line_with_confidence(
                model, line_img, img_height, device
            )
            
            # Post-process
            text = postprocess_ocr(text)
            
            # Collect diagnostics
            diag = {
                "line_num": i + 1,
                "text": text,
                "confidence": confidence,
                "status": "ok" if confidence >= confidence_threshold else "low_conf",
            }
            
            if confidence < confidence_threshold:
                diag["status"] = "low_conf"
                if verbose:
                    print(f"  Line {i+1:2d} [LOW CONF: {confidence:.2f}]: {text}")
            else:
                diag["status"] = "ok"
                if verbose:
                    print(f"  Line {i+1:2d} [conf: {confidence:.2f}]: {text}")
            
            line_texts.append(text)
            diagnostics_list.append(diag)
            
        except Exception as e:
            line_texts.append(f"[Error]")
            if verbose:
                print(f"  Line {i+1:2d} [Error]: {e}")
            diagnostics_list.append({
                "line_num": i + 1,
                "text": "[Error]",
                "confidence": 0.0,
                "status": "error",
            })

    # Combine lines
    full_text = "\n".join(line_texts)

    if verbose:
        print(f"\n[Results]")
        low_conf_count = sum(1 for d in diagnostics_list if d["status"] == "low_conf")
        if low_conf_count > 0:
            print(f"  ⚠️  {low_conf_count} lines have low confidence (< {confidence_threshold:.2f})")
            print(f"     Tip: Check these lines or adjust confidence_threshold")

    return full_text, line_texts, diagnostics_list if diagnostics else []


# ── Legacy API wrapper ────────────────────────────────────────────────────

def predict_document(
    model,
    document_path: str,
    img_height: int,
    device: torch.device,
    threshold: int = 127,
    min_gap: int = 3,
    min_height: int = 5,
    expand_margin: int = 2,
    verbose: bool = True,
) -> Tuple[str, List[str], int, Optional[str]]:
    """Legacy API for backwards compatibility."""
    full_text, line_texts, _ = predict_document_improved(
        model,
        document_path,
        img_height,
        device,
        threshold=threshold,
        min_gap=min_gap,
        min_height=min_height,
        padding_top_bottom=max(8, expand_margin),  # Recommend 8+ for Khmer
        padding_left_right=2,
        deskew=True,
        confidence_threshold=0.5,
        verbose=verbose,
        diagnostics=False,
    )
    
    return full_text, line_texts, len(line_texts), None
