"""
Improved Line Segmentation — full preprocessing pipeline for Khmer documents
─────────────────────────────────────────────────────────────────────────────
Pipeline (in order):
  1. Grayscale conversion
  2. CLAHE contrast enhancement
  3. Document-level deskew (correct whole-page rotation before segmenting)
  4. Binary threshold: Otsu (auto) / adaptive / fixed
  5. Vertical morphological dilation — joins floating Khmer diacritics to their
     base consonant row so they are not split as separate lines
  6. Gaussian-smoothed horizontal projection — suppresses noise-spike boundaries
  7. Adaptive gap detection — threshold derived from the document itself so it
     works at any resolution / zoom level
  8. Crop with top/bottom padding + optional per-line deskew
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image


# ── Helpers ───────────────────────────────────────────────────────────────────

def image_to_array(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to uint8 grayscale numpy array."""
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr


# ── Step 0 — Scan border removal ─────────────────────────────────────────────

def remove_scan_borders(gray: np.ndarray, dark_threshold: int = 50) -> np.ndarray:
    """
    Crop away thick dark borders produced by flatbed scanners.

    A row or column is classified as "border" when more than 60 % of its pixels
    are darker than `dark_threshold`. Without this step, large black borders
    skew the Otsu threshold calculation — Otsu sees the border as a dominant
    dark class and sets the threshold too high, causing text to be missed.

    If no substantial border is found (< 10 rows or columns removed) the image
    is returned unchanged so the function is safe to call unconditionally.
    """
    h, w = gray.shape
    BORDER_FRAC = 0.60  # row/col is border when >60 % pixels are dark

    dark_row = (gray < dark_threshold).mean(axis=1)   # (H,)
    dark_col = (gray < dark_threshold).mean(axis=0)   # (W,)

    content_rows = np.where(dark_row < BORDER_FRAC)[0]
    content_cols = np.where(dark_col < BORDER_FRAC)[0]

    if len(content_rows) < 10 or len(content_cols) < 10:
        return gray

    y0 = max(0, int(content_rows[0])  - 3)
    y1 = min(h, int(content_rows[-1]) + 4)
    x0 = max(0, int(content_cols[0])  - 3)
    x1 = min(w, int(content_cols[-1]) + 4)

    # Only crop if the saved border is at least 10 px on any side
    if y0 < 10 and y1 > h - 10 and x0 < 10 and x1 > w - 10:
        return gray

    return gray[y0:y1, x0:x1]


# ── Step 1 — CLAHE contrast enhancement ──────────────────────────────────────

def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE — normalises local contrast for faded/shadowed scans."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ── Step 2 — Document-level deskew ───────────────────────────────────────────

def deskew_document(gray: np.ndarray) -> np.ndarray:
    """
    Correct whole-page rotation before segmenting.

    Doing this at the document level (instead of per-line) prevents the
    horizontal projection from smearing across rows when the page is rotated
    even 2–3 degrees. Per-line deskew still runs afterward as a refinement.
    Skips correction when angle < 0.5° to avoid unnecessary warping.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


# ── Step 3 — Binary thresholding ─────────────────────────────────────────────

def apply_binary_threshold(
    gray: np.ndarray,
    threshold: int = 127,
    use_otsu: bool = True,
    use_adaptive: bool = False,
) -> np.ndarray:
    """
    Threshold modes (applied in priority order):
      use_adaptive=True  → ADAPTIVE_THRESH_GAUSSIAN_C (photos with uneven lighting)
      use_otsu=True      → Otsu auto-threshold        (clean scans)
      else               → fixed `threshold` value
    """
    if use_adaptive:
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=31,
            C=10,
        )
    if use_otsu:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return binary
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


# ── Step 4 — Horizontal projection (with dilation + smoothing) ───────────────

def compute_horizontal_projection(
    binary: np.ndarray,
    dilate_for_khmer: bool = True,
    smooth: bool = True,
) -> np.ndarray:
    """
    Compute horizontal projection profile with two preprocessing steps:

    dilate_for_khmer:
        Vertically dilates the binary image (kernel 1×12) so above/below-base
        Khmer diacritics (e.g. ំ ់ ៉) merge with their base consonant row
        before the projection is computed. Without this, diacritics appear as
        isolated pixel rows that the gap detector treats as separate lines.
        The dilated image is used only for projection — crops still come from
        the original grayscale.

    smooth:
        Convolves the projection with a Gaussian (σ=1.5) to suppress single-row
        noise spikes that would otherwise create spurious line boundaries.
    """
    if dilate_for_khmer:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 12))
        binary = cv2.dilate(binary, kernel, iterations=1)

    projection = np.sum(binary, axis=1) // 255

    if smooth:
        sigma = 1.5
        half = max(3, int(3 * sigma))
        x = np.arange(-half, half + 1, dtype=float)
        gauss = np.exp(-x ** 2 / (2 * sigma ** 2))
        gauss /= gauss.sum()
        projection = np.convolve(projection.astype(float), gauss, mode='same')
        projection = (projection > 0.5).astype(int)

    return projection


# ── Step 5 — Line boundary detection ─────────────────────────────────────────

def detect_line_boundaries(
    projection: np.ndarray,
    min_gap: int = 3,
    min_height: int = 5,
) -> List[Tuple[int, int]]:
    """Fixed-gap line boundary detection (used by adaptive version internally)."""
    text_rows = np.where(projection > 0)[0]
    if len(text_rows) == 0:
        return []

    lines: List[Tuple[int, int]] = []
    start = prev = text_rows[0]

    for row in text_rows[1:]:
        if row - prev > min_gap:
            if prev - start + 1 >= min_height:
                lines.append((int(start), int(prev)))
            start = row
        prev = row

    if text_rows[-1] - start + 1 >= min_height:
        lines.append((int(start), int(text_rows[-1])))

    return lines


def detect_line_boundaries_adaptive(
    projection: np.ndarray,
    min_height: int = 5,
) -> List[Tuple[int, int]]:
    """
    Derive the gap threshold from the document itself.

    Measures the length of every zero-run (gap between text rows) in the
    projection, takes the median, and uses 40 % of that as the split
    threshold. This makes the detector work correctly at any scan resolution
    without needing the user to manually tune `min_gap`.
    """
    gap_lengths: List[int] = []
    in_gap = False
    run = 0
    for p in projection:
        if p == 0:
            run += 1
            in_gap = True
        else:
            if in_gap:
                gap_lengths.append(run)
            in_gap = False
            run = 0
    if in_gap:
        gap_lengths.append(run)

    min_gap = max(2, int(np.median(gap_lengths) * 0.4)) if len(gap_lengths) >= 2 else 2
    return detect_line_boundaries(projection, min_gap=min_gap, min_height=min_height)


# ── Step 6 — Per-line deskew ──────────────────────────────────────────────────

def deskew_line(line_img: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Deskew a single cropped line using minAreaRect on the largest contour."""
    _, binary = cv2.threshold(line_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return line_img
    largest = max(contours, key=cv2.contourArea)
    angle = cv2.minAreaRect(largest)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > max_angle:
        return line_img
    h, w = line_img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(line_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


# ── Main segmentation entry point ─────────────────────────────────────────────

def segment_document_improved(
    img: Image.Image,
    threshold: int = 127,
    min_gap: int = 3,
    min_height: int = 5,
    padding_top_bottom: int = 8,
    padding_left_right: int = 2,
    deskew: bool = True,
    return_metadata: bool = False,
    use_otsu: bool = True,
    use_adaptive: bool = False,
    use_clahe: bool = True,
    deskew_document_first: bool = True,
    adaptive_gap: bool = True,
    remove_borders: bool = True,
) -> Tuple[List[Image.Image], List[Tuple[int, int]], Optional[dict]]:
    """
    Full segmentation pipeline for Khmer documents.

    Pipeline order:
      0. remove_scan_borders  — crop dark flatbed scanner borders before CLAHE
         (borders skew Otsu's histogram if not removed first)
      1. CLAHE contrast enhancement
      2. Document-level deskew
      3. Binary threshold (Otsu / adaptive / fixed)
      4. Morphological dilation + Gaussian-smoothed horizontal projection
      5. Adaptive or fixed gap-based line detection
      6. Crop with padding + optional per-line deskew

    Args:
        remove_borders       : Remove thick dark scanner borders before processing
        (all other args same as before)

    Returns:
        (line_images, line_bounds, metadata)
        metadata is None unless return_metadata=True.
        When return_metadata=True, metadata["preprocessed_gray"] holds the
        processed grayscale array (after border removal, CLAHE, deskew) which
        callers can use to re-crop lines for retry logic.
    """
    gray = image_to_array(img)

    if remove_borders:
        gray = remove_scan_borders(gray)

    if use_clahe:
        gray = enhance_contrast(gray)

    if deskew_document_first:
        gray = deskew_document(gray)

    img_h, img_w = gray.shape

    binary = apply_binary_threshold(
        gray,
        threshold=threshold,
        use_otsu=use_otsu,
        use_adaptive=use_adaptive,
    )

    projection = compute_horizontal_projection(binary, dilate_for_khmer=True, smooth=True)

    if adaptive_gap:
        line_bounds = detect_line_boundaries_adaptive(projection, min_height=min_height)
    else:
        line_bounds = detect_line_boundaries(projection, min_gap=min_gap, min_height=min_height)

    if not line_bounds:
        empty_meta = {"error": "No lines detected"} if return_metadata else None
        return [], [], empty_meta

    line_images: List[Image.Image] = []
    original_bounds: List[Tuple[int, int]] = []

    for start, end in line_bounds:
        y0 = max(0, start - padding_top_bottom)
        y1 = min(img_h, end + padding_top_bottom + 1)

        line_np = gray[y0:y1, :]  # full width, from deskewed gray

        if deskew:
            line_np = deskew_line(line_np, max_angle=15.0)

        line_images.append(Image.fromarray(line_np))
        original_bounds.append((start, end))

    metadata: Optional[dict] = None
    if return_metadata:
        metadata = {
            "num_lines": len(line_images),
            "image_size": img.size,
            "line_bounds": original_bounds,
            "padding_tb": padding_top_bottom,
            "padding_lr": padding_left_right,
            "deskewed": deskew,
            "use_otsu": use_otsu,
            "use_adaptive": use_adaptive,
            "use_clahe": use_clahe,
            "deskew_document_first": deskew_document_first,
            "adaptive_gap": adaptive_gap,
            # Callers (e.g. retry logic) can re-crop from this without
            # re-running the expensive preprocessing pipeline
            "preprocessed_gray": gray,
        }

    return line_images, original_bounds, metadata


# ── Legacy API ────────────────────────────────────────────────────────────────

def segment_document(
    img: Image.Image,
    threshold: int = 127,
    min_gap: int = 3,
    min_height: int = 5,
    expand_margin: int = 0,
) -> Tuple[List[Image.Image], List[Tuple[int, int]]]:
    """Backwards-compatible wrapper — calls the full improved pipeline."""
    line_images, line_bounds, _ = segment_document_improved(
        img,
        threshold=threshold,
        min_gap=min_gap,
        min_height=min_height,
        padding_top_bottom=max(8, expand_margin),
        deskew=True,
        use_otsu=True,
        use_clahe=True,
        deskew_document_first=True,
        adaptive_gap=True,
        remove_borders=True,
    )
    return line_images, line_bounds


# ── Diagnostics ───────────────────────────────────────────────────────────────

def get_line_stats(projection: np.ndarray) -> dict:
    return {
        'total_rows': len(projection),
        'text_rows': int(np.sum(projection > 0)),
        'max_projection': int(np.max(projection)),
        'min_projection': int(np.min(projection)),
        'mean_projection': float(np.mean(projection)),
    }
