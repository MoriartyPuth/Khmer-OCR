"""
Improved Document Predictor
────────────────────────────────────────────────────────────────────────────
Features:
  - Beam search CTC decoding (replaces greedy argmax)
  - Confidence scoring over non-blank timesteps only
  - Thin-line guard: skips crops that are too short to produce valid text
  - Low-confidence retry: re-crops with wider padding and keeps the better result
  - Full preprocessing pipeline forwarded from improved_line_segmentation
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.vocab import NUM_CLASSES, ctc_decode, ctc_beam_search, IDX2CHAR, CHAR2IDX
from data.dataset import get_transforms, resize_to_height
from models.crnn import KhmerOCR
from utils.improved_line_segmentation import segment_document_improved
from models.line_detector import detect_lines


# ── Load model ────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> Tuple[KhmerOCR, int]:
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
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


# ── Prediction with confidence ────────────────────────────────────────────────

MIN_LINE_HEIGHT = 8  # px — lines shorter than this produce garbage when stretched to 32px

def predict_line_with_confidence(
    model,
    line_img: Image.Image,
    img_height: int,
    device: torch.device,
    use_beam_search: bool = True,
    beam_width: int = 10,
) -> Tuple[str, float, List[float]]:
    """
    Predict text from a line image.

    Returns (text, avg_confidence, per_timestep_confidences).

    Confidence is averaged only over non-blank timesteps — blank positions have
    naturally high softmax probability and inflate the score otherwise.

    Decoding:
        use_beam_search=True  → ctc_beam_search (better accuracy, ~same speed at beam=10)
        use_beam_search=False → greedy argmax (faster, slightly less accurate)
    """
    # Thin-line guard — reject crops too short to contain real text
    if line_img.size[1] < MIN_LINE_HEIGHT:
        return "[skipped:thin]", 0.0, []

    transform = get_transforms(img_height, augment=False)
    line_img = resize_to_height(line_img.convert("RGB"), img_height)
    tensor = transform(line_img).unsqueeze(0).to(device)   # (1, 1, H, W)

    with torch.no_grad():
        logits = model(tensor)                              # (T, 1, C)
        log_probs = torch.log_softmax(logits, dim=2)       # (T, 1, C)
        probs     = torch.softmax(logits, dim=2).squeeze(1) # (T, C)

        max_probs, max_ids = probs.max(dim=1)              # (T,)

        # Confidence — exclude blank (index 0) so score reflects real characters
        non_blank = max_ids != 0
        avg_conf  = max_probs[non_blank].mean().item() if non_blank.sum() > 0 else 0.0
        conf_list = max_probs.cpu().tolist()

        if use_beam_search:
            lp_np = log_probs.squeeze(1).cpu().numpy()    # (T, C)
            text  = ctc_beam_search(lp_np, beam_width=beam_width)
        else:
            indices = logits.argmax(dim=2).squeeze(1)
            text    = ctc_decode(indices.tolist())

    return (text if text.strip() else "[empty]"), avg_conf, conf_list


# ── Post-processing ───────────────────────────────────────────────────────────

def postprocess_ocr(text: str) -> str:
    return " ".join(text.split())


# ── Neural segmentation (Model 2 — U-Net line detector) ──────────────────────

def segment_document_neural(
    img: Image.Image,
    detector_model,
    device: torch.device,
    padding: int = 4,
    input_size: int = 512,
) -> Tuple[List[Image.Image], List[Tuple[int, int]], Dict]:
    """
    Detect-then-crop using the trained U-Net line detector.

    Unlike the classical pipeline, boxes have real x-extents, so short
    centred lines are cropped tightly (no huge white margins) — this
    noticeably improves CRNN accuracy on certificate headings and dates.

    Returns the same (line_images, line_bounds, metadata) shape as
    segment_document_improved so the two paths are interchangeable.
    """
    boxes = detect_lines(detector_model, img, device, input_size=input_size)

    gray = np.array(img.convert("RGB")).min(axis=2).astype(np.uint8)
    h, w = gray.shape

    line_images:  List[Image.Image]     = []
    line_bounds:  List[Tuple[int, int]] = []
    line_xranges: List[Tuple[int, int]] = []

    for (x0, y0, x1, y1) in boxes:
        y0p, y1p = max(0, y0 - padding), min(h, y1 + padding)
        x0p, x1p = max(0, x0 - padding), min(w, x1 + padding)
        if y1p - y0p < 2 or x1p - x0p < 2:
            continue
        line_images.append(Image.fromarray(gray[y0p:y1p, x0p:x1p]))
        line_bounds.append((y0, y1))
        line_xranges.append((x0p, x1p))

    metadata: Dict = {
        "num_lines":         len(line_images),
        "image_size":        img.size,
        "line_bounds":       line_bounds,
        "line_xranges":      line_xranges,
        "preprocessed_gray": gray,
        "detector":          "unet",
    }
    return line_images, line_bounds, metadata


# ── Document prediction ───────────────────────────────────────────────────────

def predict_document_improved(
    model,
    document_path: str,
    img_height: int,
    device: torch.device,
    # Segmentation params
    threshold: int = 127,
    min_gap: int = 3,
    min_height: int = 5,
    padding_top_bottom: int = 8,
    padding_left_right: int = 2,
    deskew: bool = True,
    use_otsu: bool = True,
    use_adaptive: bool = False,
    use_clahe: bool = True,
    deskew_document_first: bool = True,
    adaptive_gap: bool = True,
    remove_borders: bool = True,
    color_mode: str = "min_channel",
    mask_graphics: bool = True,
    # Prediction params
    confidence_threshold: float = 0.6,
    use_beam_search: bool = True,
    beam_width: int = 10,
    # Neural line detector (Model 2) — optional
    detector_model=None,
    detector_input_size: int = 512,
    # I/O
    verbose: bool = True,
    diagnostics: bool = True,
) -> Tuple[str, List[str], List[Dict]]:
    """
    Full document OCR pipeline.

    New parameters:
        remove_borders       : Strip dark scanner borders before processing
        use_beam_search      : Use CTC beam search instead of greedy decoding
        beam_width           : Beam width for beam search (10 is a good default)
        detector_model       : trained LineDetectorUNet — when given, line
                               detection uses the neural detector (Model 2)
                               and falls back to the classical pipeline only
                               if the detector finds zero lines

    Low-confidence retry:
        Lines below `confidence_threshold` are automatically re-cropped with
        +6 px of extra vertical padding and re-predicted. The higher-confidence
        result is kept. This recovers lines where diacritics were clipped.

    Returns:
        (full_text, line_texts, diagnostics_list)
    """
    if not os.path.exists(document_path):
        print(f"[Error] File not found: {document_path}")
        return "", [], []

    try:
        doc_img = Image.open(document_path)
    except Exception as e:
        print(f"[Error] Could not open image: {e}")
        return "", [], []

    if verbose:
        print(f"[Document] {doc_img.size[0]}x{doc_img.size[1]}  mode={doc_img.mode}")

    # Segmentation — neural detector first (when provided), classical otherwise.
    # Always request metadata so we can retry from preprocessed_gray.
    line_images, line_bounds, metadata = [], [], None

    if detector_model is not None:
        try:
            line_images, line_bounds, metadata = segment_document_neural(
                doc_img, detector_model, device,
                padding=max(2, padding_top_bottom // 2),
                input_size=detector_input_size,
            )
            if verbose and line_images:
                print(f"[Detector] U-Net found {len(line_images)} lines")
        except Exception as e:
            print(f"[Warning] Neural detection failed ({e}) — falling back to classical")
            line_images = []

    if not line_images:
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
                use_otsu=use_otsu,
                use_adaptive=use_adaptive,
                use_clahe=use_clahe,
                deskew_document_first=deskew_document_first,
                adaptive_gap=adaptive_gap,
                remove_borders=remove_borders,
                color_mode=color_mode,
                mask_graphics=mask_graphics,
            )
        except Exception as e:
            print(f"[Error] Segmentation failed: {e}")
            return "", [], []

    if not line_images:
        print("[Warning] No lines detected")
        return "", [], []

    preprocessed_gray: Optional[np.ndarray] = (metadata or {}).get("preprocessed_gray")
    gray_h = preprocessed_gray.shape[0] if preprocessed_gray is not None else 0
    gray_w = preprocessed_gray.shape[1] if preprocessed_gray is not None else 0
    # Neural detector provides x-extents per line; classical crops full width
    line_xranges: Optional[List[Tuple[int, int]]] = (metadata or {}).get("line_xranges")

    if verbose:
        decoder = f"beam(w={beam_width})" if use_beam_search else "greedy"
        print(f"[Segmentation] {len(line_images)} lines | "
              f"threshold={'otsu' if use_otsu else 'adaptive' if use_adaptive else threshold} | "
              f"clahe={use_clahe} | borders={remove_borders} | decode={decoder}")

    line_texts:      List[str]  = []
    diagnostics_list: List[Dict] = []

    for i, line_img in enumerate(line_images):
        try:
            text, confidence, char_confs = predict_line_with_confidence(
                model, line_img, img_height, device,
                use_beam_search=use_beam_search,
                beam_width=beam_width,
            )
            retried = False

            # ── Low-confidence retry with wider padding ───────────────────────
            if confidence < confidence_threshold and preprocessed_gray is not None:
                y0_raw, y1_raw = line_bounds[i]
                wider = padding_top_bottom + 6
                y0 = max(0, y0_raw - wider)
                y1 = min(gray_h, y1_raw + wider + 1)
                if line_xranges and i < len(line_xranges):
                    x_lo = max(0, line_xranges[i][0] - 6)
                    x_hi = min(gray_w, line_xranges[i][1] + 6)
                else:
                    x_lo, x_hi = 0, gray_w
                retry_np  = preprocessed_gray[y0:y1, x_lo:x_hi]
                retry_img = Image.fromarray(retry_np)

                r_text, r_conf, r_confs = predict_line_with_confidence(
                    model, retry_img, img_height, device,
                    use_beam_search=use_beam_search,
                    beam_width=beam_width,
                )
                r_text = postprocess_ocr(r_text)

                if r_conf > confidence:
                    text, confidence, char_confs = r_text, r_conf, r_confs
                    retried = True

            text = postprocess_ocr(text)

            status = "ok" if confidence >= confidence_threshold else "low_conf"
            diag: Dict = {
                "line_num":   i + 1,
                "text":       text,
                "confidence": confidence,
                "status":     status,
                "retried":    retried,
            }

            if verbose:
                tag = f"LOW CONF {confidence:.2f}" if status == "low_conf" else f"conf {confidence:.2f}"
                retry_tag = " [retried]" if retried else ""
                print(f"  Line {i+1:2d} [{tag}]{retry_tag}: {text}")

            line_texts.append(text)
            diagnostics_list.append(diag)

        except Exception as e:
            line_texts.append("[Error]")
            diagnostics_list.append({
                "line_num": i + 1, "text": "[Error]",
                "confidence": 0.0, "status": "error", "retried": False,
            })
            if verbose:
                print(f"  Line {i+1:2d} [Error]: {e}")

    full_text = "\n".join(line_texts)

    if verbose:
        low = sum(1 for d in diagnostics_list if d["status"] == "low_conf")
        retried_count = sum(1 for d in diagnostics_list if d.get("retried"))
        if low:
            print(f"\n  ⚠  {low} line(s) below confidence {confidence_threshold:.2f}")
        if retried_count:
            print(f"  ↺  {retried_count} line(s) improved by retry")

    return full_text, line_texts, diagnostics_list if diagnostics else []


# ── Legacy API wrapper ────────────────────────────────────────────────────────

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
    """Backwards-compatible wrapper."""
    full_text, line_texts, _ = predict_document_improved(
        model, document_path, img_height, device,
        threshold=threshold,
        min_gap=min_gap,
        min_height=min_height,
        padding_top_bottom=max(8, expand_margin),
        deskew=True,
        confidence_threshold=0.5,
        verbose=verbose,
        diagnostics=False,
        use_otsu=True,
        use_clahe=True,
        deskew_document_first=True,
        adaptive_gap=True,
        remove_borders=True,
        use_beam_search=True,
        beam_width=10,
    )
    return full_text, line_texts, len(line_texts), None
