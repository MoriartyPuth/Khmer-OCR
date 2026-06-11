"""
Improved Line Segmentation — full preprocessing pipeline for Khmer documents
─────────────────────────────────────────────────────────────────────────────
Pipeline (in order):
  0a. extract_text_channel — color-aware grayscale: "min_channel" makes any
      dark-colored text (orange, blue, navy) appear dark; solves colored certs
  0b. remove_scan_borders  — crop near-black flatbed scanner borders
  0c. crop_to_content_region — crop colored design borders (certificates, forms)
  1.  CLAHE contrast enhancement
  2.  Document-level deskew
  3.  Binary threshold (Otsu / adaptive / fixed)
  4.  PRIMARY — segment_lines_by_components: connected-component text-line
       grouping (what modern OCR engines including EasyOCR/PaddleOCR do).
       Groups text blobs by vertical proximity — no reliance on "zero-ink rows"
       so it works on certificates, forms, and any colored-background document.
  4b. FALLBACK — horizontal projection gap detection (kept for plain scans
       where CC might over-split sparse text).
  5.  Crop with padding + aspect-ratio filter + per-line deskew
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image


# ── helpers ───────────────────────────────────────────────────────────────────

def image_to_array(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to uint8 grayscale numpy array (standard luminance)."""
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr


# ── Step 0a — Color-aware grayscale extraction ────────────────────────────────

def extract_text_channel(img: Image.Image, color_mode: str = "min_channel") -> np.ndarray:
    """
    Convert PIL Image to a grayscale array that makes dark-colored text visible.

    color_mode options:
      "min_channel" (default) — minimum of R, G, B per pixel.
          Dark text of ANY color (orange, blue, navy, red) has at least one
          channel near 0, so its minimum is near 0 → appears very dark.
          White paper (255,255,255) → min = 255 → stays bright.
          Colored background (e.g. blue 70,120,200) → min = 70 → medium gray.
          Best for certificates and forms with colored text/borders.

      "gray" — standard luminance grayscale (0.299R + 0.587G + 0.114B).
          Orange text (230,100,0) → gray ≈ 127 → blends into background.
          Best for plain black-on-white scanned documents.

      "value_inv" — 255 minus HSV Value channel.
          Value = max(R,G,B). Inverted so dark pixels become bright.
          Alternative for colored documents.
    """
    arr = np.array(img.convert("RGB"))
    if color_mode == "min_channel":
        return arr.min(axis=2).astype(np.uint8)
    if color_mode == "value_inv":
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        return (255 - hsv[:, :, 2]).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


# ── Step 0b — Scanner border removal ─────────────────────────────────────────

def remove_scan_borders(gray: np.ndarray, dark_threshold: int = 50) -> np.ndarray:
    """
    Crop away thick near-black borders from flatbed scanner output.
    A row/col is classified as border when >60% of its pixels are below
    `dark_threshold`. Returns image unchanged if no substantial border found.
    """
    h, w = gray.shape
    BORDER_FRAC = 0.60

    dark_row = (gray < dark_threshold).mean(axis=1)
    dark_col = (gray < dark_threshold).mean(axis=0)

    content_rows = np.where(dark_row < BORDER_FRAC)[0]
    content_cols = np.where(dark_col < BORDER_FRAC)[0]

    if len(content_rows) < 10 or len(content_cols) < 10:
        return gray

    y0 = max(0, int(content_rows[0])  - 3)
    y1 = min(h, int(content_rows[-1]) + 4)
    x0 = max(0, int(content_cols[0])  - 3)
    x1 = min(w, int(content_cols[-1]) + 4)

    if y0 < 10 and y1 > h - 10 and x0 < 10 and x1 > w - 10:
        return gray  # no substantial border

    return gray[y0:y1, x0:x1]


# ── Step 0c — Colored border / content region crop ───────────────────────────

def crop_to_content_region(
    gray: np.ndarray,
    bright_mean_threshold: int = 160,
) -> np.ndarray:
    """
    Crop away colored decorative borders by finding where row/column mean
    brightness drops below `bright_mean_threshold`.

    WHY THIS IS CRITICAL for certificates:
        A blue gradient border (min_channel value ~70-100) makes every row
        have projection > 0 even between text lines, so no gaps are detected
        and the algorithm returns zero lines. Cropping the border first removes
        all horizontal spanning "noise columns" so real inter-line gaps appear.

    Works in concert with `extract_text_channel("min_channel")`:
        Blue gradient columns: mean ~70-100  → below 160 → cropped ✓
        White content columns: mean ~200-240 → above 160 → kept   ✓
        Plain scanner doc (no colored border): no columns cropped  ✓

    bright_mean_threshold: lower = more aggressive cropping.
        160 is safe for most certificates (blue border ≈ 70-130, paper ≈ 200+).
    """
    h, w = gray.shape

    row_mean = gray.mean(axis=1)
    col_mean = gray.mean(axis=0)

    content_rows = np.where(row_mean > bright_mean_threshold)[0]
    content_cols = np.where(col_mean > bright_mean_threshold)[0]

    if len(content_rows) < 20 or len(content_cols) < 20:
        return gray

    y0 = max(0, int(content_rows[0])  - 5)
    y1 = min(h, int(content_rows[-1]) + 6)
    x0 = max(0, int(content_cols[0])  - 5)
    x1 = min(w, int(content_cols[-1]) + 6)

    # Only apply if we're actually removing a meaningful border
    if y0 < h * 0.01 and y1 > h * 0.99 and x0 < w * 0.01 and x1 > w * 0.99:
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
    Skips correction when detected angle < 0.5° to avoid unnecessary warping.
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
    Threshold modes (priority order):
      use_adaptive=True → ADAPTIVE_THRESH_GAUSSIAN_C (phone photos, uneven light)
      use_otsu=True     → Otsu auto-threshold        (clean scans, most documents)
      else              → fixed `threshold` value
    """
    if use_adaptive:
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10,
        )
    if use_otsu:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return binary
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


# ── Step 4 — Graphical region masking ────────────────────────────────────────

def mask_graphical_regions(binary: np.ndarray) -> np.ndarray:
    """
    Remove logos, seals, and decorative images from the binary image before
    computing the horizontal projection.

    WHY NOT morphological 2D closing:
        A 2D close kernel large enough to fill a logo (≥ 30 px) also bridges
        the 20–40 px inter-line gaps of typical certificate text, merging all
        text lines into one giant blob whose AR drops below the threshold —
        the ENTIRE text gets masked and 0 lines are detected.

    CORRECT APPROACH — horizontal-only dilation:
        Dilation in the X direction only connects characters that are on the
        SAME row (within the same text line) without bridging the VERTICAL
        gaps between lines. After this dilation:
          • A text line → one wide rectangle  (bw ≈ image width)
          • A logo / seal → one compact blob   (bw ≈ logo diameter)

        We keep only the wide text-line components; the compact graphic
        components (where bw < 20% of image width AND aspect ratio < 2.5)
        are masked out.
    """
    h, w = binary.shape

    # Horizontal dilation: connects characters within a line, leaves vertical
    # inter-line gaps intact.  Two iterations to bridge typical character spacing.
    h_kw = max(15, w // 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kw, 1))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    mask = np.ones_like(binary, dtype=np.uint8)
    for i in range(1, num_labels):
        bx, by, bw, bh, area = stats[i, :5]
        if bw == 0 or bh == 0:
            continue
        # Text lines span most of the page width → bw / w is large
        # Graphics (logos, seals) are compact → bw / w is small AND roughly square
        if bw / w < 0.20 and bw / bh < 2.5:
            mask[by:by + bh, bx:bx + bw] = 0

    return (binary * mask).astype(np.uint8)


# ── Step 4b — CC-based text-line grouping (PRIMARY detector) ─────────────────

def segment_lines_by_components(
    binary: np.ndarray,
    min_char_area: int = 100,
    min_line_width_frac: float = 0.03,
    min_line_ar: float = 2.5,
) -> List[Tuple[int, int]]:
    """
    Detect text-line bounding boxes using connected-component grouping.

    This is what modern OCR engines (EasyOCR, PaddleOCR, Tesseract) do at their
    core: find character blobs, group nearby ones into text lines, return the
    bounding box of each group as a line crop.

    Unlike horizontal-projection gap detection this does NOT require rows of
    zero ink between lines, so it works correctly on certificates, coloured
    forms, and any document whose background prevents clean zero-projection gaps.

    Algorithm:
      1. Small vertical dilation (1×5) to connect Khmer diacritics to their
         base consonants without bridging the much-larger inter-line gaps.
      2. Find all connected components.
      3. Discard noise (area < min_char_area).
      4. Sort remaining components by vertical centre.
      5. Greedily merge into line groups: a component joins the current group
         when its Y-range overlaps or is within (30% of group's median height)
         pixels of the group's current Y extent.
      6. Discard groups that are:
           • narrower than min_line_width_frac × image width  (isolated dots)
           • have group_width / group_height < min_line_ar    (logos, seals)
      7. Merge any overlapping group bounds and return sorted (y0, y1) list.

    min_line_ar=2.5 filters compact graphics naturally — a logo/seal is
    roughly square (AR ≈ 1) while even the shortest centred text heading
    has AR > 3 at typical certificate resolution.
    """
    h, w = binary.shape

    # Thin vertical dilation: joins stacked Khmer diacritics to base glyph
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    dilated  = cv2.dilate(binary, v_kernel, iterations=1)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    comps = []
    for i in range(1, num_labels):
        bx, by, bw, bh, area = stats[i, :5]
        if area < min_char_area:
            continue
        comps.append((bx, by, bw, bh))

    if not comps:
        return []

    comps.sort(key=lambda c: c[1] + c[3] / 2)   # sort by vertical centre

    # Adaptive merge distance: 30% of the median component height so it
    # bridges within-line diacritic gaps but not the larger inter-line gaps
    median_h = float(np.median([c[3] for c in comps]))
    merge_px = max(6, int(median_h * 0.30))

    groups: List[List] = [[comps[0]]]
    for comp in comps[1:]:
        bx, by, bw, bh = comp
        last = groups[-1]
        g_y0 = min(c[1]        for c in last)
        g_y1 = max(c[1] + c[3] for c in last)
        if by <= g_y1 + merge_px and (by + bh) >= g_y0 - merge_px:
            last.append(comp)
        else:
            groups.append([comp])

    bounds = []
    for grp in groups:
        lx0 = min(c[0]        for c in grp)
        lx1 = max(c[0] + c[2] for c in grp)
        ly0 = min(c[1]        for c in grp)
        ly1 = max(c[1] + c[3] for c in grp)

        grp_w = lx1 - lx0
        grp_h = max(ly1 - ly0, 1)

        if grp_w / w < min_line_width_frac:
            continue          # too narrow — isolated punctuation / decorative dot
        if grp_w / grp_h < min_line_ar:
            continue          # roughly square — logo, seal, watermark

        bounds.append((ly0, ly1))

    bounds.sort(key=lambda b: b[0])

    # Merge any overlapping bounds (edge case after the grouping above)
    merged: List[List[int]] = []
    for y0, y1 in bounds:
        if merged and y0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], y1)
        else:
            merged.append([y0, y1])

    return [(y0, y1) for y0, y1 in merged]


# ── Step 5 — Horizontal projection (dilation + smoothing) ────────────────────

def compute_horizontal_projection(
    binary: np.ndarray,
    dilate_for_khmer: bool = True,
    smooth: bool = True,
) -> np.ndarray:
    """
    dilate_for_khmer: vertical dilation (1×12) joins above/below-base Khmer
        diacritics to their base consonant row so they are not split as lines.
    smooth: Gaussian convolution suppresses isolated noise-spike boundaries.
    """
    if dilate_for_khmer:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 12))
        binary = cv2.dilate(binary, kernel, iterations=1)

    projection = np.sum(binary, axis=1) // 255

    if smooth:
        sigma = 1.5
        half  = max(3, int(3 * sigma))
        x     = np.arange(-half, half + 1, dtype=float)
        gauss = np.exp(-x ** 2 / (2 * sigma ** 2))
        gauss /= gauss.sum()
        projection = np.convolve(projection.astype(float), gauss, mode='same')
        projection = (projection > 0.5).astype(int)

    return projection


# ── Step 6 — Line boundary detection ─────────────────────────────────────────

def detect_line_boundaries(
    projection: np.ndarray,
    min_gap: int = 3,
    min_height: int = 5,
) -> List[Tuple[int, int]]:
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
    """Derive the gap threshold from the document's own inter-line spacing."""
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


# ── Step 7 — Per-line deskew ──────────────────────────────────────────────────

def deskew_line(line_img: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Deskew a single cropped line using minAreaRect on the largest contour."""
    _, binary = cv2.threshold(line_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return line_img
    largest = max(contours, key=cv2.contourArea)
    angle   = cv2.minAreaRect(largest)[-1]
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
    color_mode: str = "min_channel",
    mask_graphics: bool = True,
    min_line_aspect: float = 2.0,
) -> Tuple[List[Image.Image], List[Tuple[int, int]], Optional[dict]]:
    """
    Full segmentation pipeline for Khmer documents.

    Key new parameters vs the previous version:

    color_mode (str, default "min_channel"):
        How to convert the image to grayscale before processing.
        "min_channel" — best for colored certificates/forms (orange/blue text).
        "gray"        — standard luminance, best for plain scanned documents.
        "value_inv"   — inverted HSV value, alternative for colored documents.

    mask_graphics (bool, default True):
        Remove logos, seals, and large decorative images before projection.
        Without this, a 300×300 logo creates ~300 false "text rows".

    min_line_aspect (float, default 2.0):
        Minimum width/height ratio to accept a detected region as a text line.
        Logos/seals after border removal typically have AR ≈ 1.
        Real text lines have AR >> 3 (even short centered headings).

    Returns:
        (line_images, line_bounds, metadata)
        metadata["preprocessed_gray"] is available for retry logic when
        return_metadata=True.
    """
    # 0a. Color-aware grayscale
    gray = extract_text_channel(img, color_mode)

    # 0b/c. Border removal — scanner borders first, then colored design borders
    if remove_borders:
        gray = remove_scan_borders(gray)
        gray = crop_to_content_region(gray)

    # 1. Contrast enhancement
    if use_clahe:
        gray = enhance_contrast(gray)

    # 2. Document-level deskew
    if deskew_document_first:
        gray = deskew_document(gray)

    img_h, img_w = gray.shape

    # 3. Binary threshold
    binary = apply_binary_threshold(
        gray, threshold=threshold, use_otsu=use_otsu, use_adaptive=use_adaptive,
    )

    # 4. PRIMARY — connected-component text-line grouping
    #    Works on ANY background (colored certs, forms, uneven lighting)
    #    because it doesn't require zero-projection rows between lines.
    line_bounds = segment_lines_by_components(binary)

    # 4b. FALLBACK — horizontal projection (for plain scans with very sparse text
    #     where CC grouping may over-fragment into many tiny groups)
    if not line_bounds:
        binary_proj = mask_graphical_regions(binary) if mask_graphics else binary
        projection  = compute_horizontal_projection(binary_proj, dilate_for_khmer=True, smooth=True)
        if adaptive_gap:
            line_bounds = detect_line_boundaries_adaptive(projection, min_height=min_height)
        else:
            line_bounds = detect_line_boundaries(projection, min_gap=min_gap, min_height=min_height)

    if not line_bounds:
        empty_meta = {"error": "No lines detected"} if return_metadata else None
        return [], [], empty_meta

    # 7. Crop, filter, deskew
    line_images:    List[Image.Image]   = []
    original_bounds: List[Tuple[int, int]] = []

    for start, end in line_bounds:
        y0 = max(0, start - padding_top_bottom)
        y1 = min(img_h, end + padding_top_bottom + 1)
        line_h = y1 - y0

        if line_h < 1:
            continue

        # Aspect ratio filter: skip blobs that are too square to be text lines
        ar = img_w / line_h
        if ar < min_line_aspect:
            continue

        line_np = gray[y0:y1, :]

        if deskew:
            line_np = deskew_line(line_np, max_angle=15.0)

        line_images.append(Image.fromarray(line_np))
        original_bounds.append((start, end))

    metadata: Optional[dict] = None
    if return_metadata:
        metadata = {
            "num_lines":        len(line_images),
            "image_size":       img.size,
            "line_bounds":      original_bounds,
            "padding_tb":       padding_top_bottom,
            "padding_lr":       padding_left_right,
            "deskewed":         deskew,
            "color_mode":       color_mode,
            "mask_graphics":    mask_graphics,
            "use_otsu":         use_otsu,
            "use_adaptive":     use_adaptive,
            "use_clahe":        use_clahe,
            "deskew_document_first": deskew_document_first,
            "adaptive_gap":     adaptive_gap,
            "preprocessed_gray": gray,   # used by retry logic in document predictor
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
        color_mode="min_channel",
        mask_graphics=True,
    )
    return line_images, line_bounds


# ── Diagnostics ───────────────────────────────────────────────────────────────

def get_line_stats(projection: np.ndarray) -> dict:
    return {
        'total_rows':       len(projection),
        'text_rows':        int(np.sum(projection > 0)),
        'max_projection':   int(np.max(projection)),
        'min_projection':   int(np.min(projection)),
        'mean_projection':  float(np.mean(projection)),
    }
