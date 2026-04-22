"""
Improved Line Segmentation with Padding, Deskewing, and Quality Checks
────────────────────────────────────────────────────────────────────────
Enhancements:
1. Adaptive padding (especially for Khmer diacritics)
2. Per-line deskewing using minAreaRect
3. Line quality checks (size validation)
4. Confidence-aware processing
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
import warnings


def image_to_array(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array (uint8, grayscale)."""
    if img.mode != 'RGB' and img.mode != 'L':
        img = img.convert('RGB')
    
    img_array = np.array(img)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    return gray


def apply_binary_threshold(gray: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Apply binary thresholding (for segmentation only)."""
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


def compute_horizontal_projection(binary: np.ndarray) -> np.ndarray:
    """Compute horizontal projection profile."""
    projection = np.sum(binary, axis=1) // 255
    return projection


def detect_line_boundaries(
    projection: np.ndarray,
    min_gap: int = 3,
    min_height: int = 5
) -> List[Tuple[int, int]]:
    """Detect line boundaries from horizontal projection."""
    text_rows = np.where(projection > 0)[0]
    
    if len(text_rows) == 0:
        return []
    
    lines = []
    start = text_rows[0]
    prev = text_rows[0]
    
    for row in text_rows[1:]:
        gap = row - prev
        
        if gap > min_gap:
            end = prev
            height = end - start + 1
            
            if height >= min_height:
                lines.append((start, end))
            
            start = row
        
        prev = row
    
    # Don't forget the last line
    end = text_rows[-1]
    height = end - start + 1
    if height >= min_height:
        lines.append((start, end))
    
    return lines


def deskew_line(line_img: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Deskew a line image using minAreaRect.
    
    Args:
        line_img: Grayscale line image (H, W)
        max_angle: Maximum expected skew angle in degrees
    
    Returns:
        Deskewed line image
    """
    # Threshold to binary for contour detection
    _, binary = cv2.threshold(line_img, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return line_img
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]
    
    # Normalize angle to [-90, 0]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > max_angle:
        return line_img  # Angle too large, return as is
    
    # Rotate image
    h, w = line_img.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(line_img, rotation_matrix, (w, h), 
                               borderMode=cv2.BORDER_REPLICATE)
    
    return deskewed


def segment_document_improved(
    img: Image.Image,
    threshold: int = 127,
    min_gap: int = 3,
    min_height: int = 5,
    padding_top_bottom: int = 8,
    padding_left_right: int = 2,
    deskew: bool = True,
    return_metadata: bool = False,
) -> Tuple[List[Image.Image], List[Tuple[int, int]], Optional[dict]]:
    """
    Improved line segmentation with padding and deskewing.
    
    Args:
        img: PIL Image of document
        threshold: Binary threshold for line detection
        min_gap: Minimum gap between lines
        min_height: Minimum line height
        padding_top_bottom: Vertical padding (5-10 recommended for Khmer)
        padding_left_right: Horizontal padding (small)
        deskew: Whether to deskew each line
        return_metadata: Return metadata about segmentation
    
    Returns:
        (line_images, line_bounds, metadata) if return_metadata=True
        else (line_images, line_bounds, None)
    """
    # Convert to grayscale
    gray = image_to_array(img)
    
    # Apply binary threshold (for line detection only)
    binary = apply_binary_threshold(gray, threshold)
    
    # Compute horizontal projection
    projection = compute_horizontal_projection(binary)
    
    # Detect line boundaries
    line_bounds = detect_line_boundaries(projection, min_gap, min_height)
    
    if not line_bounds:
        return [], [], {"error": "No lines detected"}
    
    # Extract line images with padding and deskewing
    line_images = []
    original_line_bounds = []
    
    for start, end in line_bounds:
        height = end - start + 1
        
        # Apply padding (expand especially top/bottom for Khmer)
        padded_start = max(0, start - padding_top_bottom)
        padded_end = min(gray.shape[0] - 1, end + padding_top_bottom)
        
        # Left/right padding (small)
        padded_left = max(0, 0 - padding_left_right)
        padded_right = min(img.width - 1, img.width + padding_left_right)
        
        # Crop from original image (to preserve color info)
        line_img_pil = img.crop((
            padded_left,
            padded_start,
            padded_right,
            padded_end + 1
        ))
        
        # Convert to numpy for deskewing
        line_img_np = image_to_array(line_img_pil)
        
        # Deskew if enabled
        if deskew:
            line_img_np = deskew_line(line_img_np, max_angle=15.0)
        
        # Convert back to PIL
        line_img_pil = Image.fromarray(line_img_np)
        
        line_images.append(line_img_pil)
        original_line_bounds.append((start, end))
    
    metadata = {
        "num_lines": len(line_images),
        "image_size": img.size,
        "line_bounds": original_line_bounds,
        "padding_tb": padding_top_bottom,
        "padding_lr": padding_left_right,
        "deskewed": deskew,
    } if return_metadata else None
    
    return line_images, original_line_bounds, metadata


def segment_document(
    img: Image.Image,
    threshold: int = 127,
    min_gap: int = 3,
    min_height: int = 5,
    expand_margin: int = 0
) -> Tuple[List[Image.Image], List[Tuple[int, int]]]:
    """
    Legacy API for backwards compatibility.
    Calls improved segmentation with recommended defaults for Khmer.
    """
    # Map old expand_margin to new padding parameters
    # Recommend higher padding for Khmer (8-10 pixels top/bottom)
    padding_tb = max(8, expand_margin)  # At least 8 for Khmer diacritics
    padding_lr = 2  # Small left/right padding
    
    line_images, line_bounds, _ = segment_document_improved(
        img,
        threshold=threshold,
        min_gap=min_gap,
        min_height=min_height,
        padding_top_bottom=padding_tb,
        padding_left_right=padding_lr,
        deskew=True,
        return_metadata=False,
    )
    
    return line_images, line_bounds


def get_line_stats(projection: np.ndarray) -> dict:
    """Get statistics about the projection profile."""
    return {
        'total_rows': len(projection),
        'text_rows': int(np.sum(projection > 0)),
        'max_projection': int(np.max(projection)),
        'min_projection': int(np.min(projection)),
        'mean_projection': float(np.mean(projection)),
    }
