"""
Error Level Analysis (ELA) Forensic Module
Detects compression discrepancies, splices, and synthetic generation artifacts.
"""

import io
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance
from typing import Tuple, Dict, Any


def compute_ela(image: Image.Image, quality: int = 90, scale: float = 15.0) -> Tuple[np.ndarray, float]:
    """
    Computes Error Level Analysis by resaving the image at a known JPEG quality
    and calculating the amplified pixel-level difference.
    Returns:
        ela_visual: RGB image array of amplified compression differences
        ela_score: Normalized anomaly score [0.0 - 1.0]
    """
    # Ensure image is in RGB
    orig_rgb = image.convert("RGB")

    # Save to in-memory JPEG buffer at specified quality
    buffer = io.BytesIO()
    orig_rgb.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    # Calculate absolute difference
    diff = ImageChops.difference(orig_rgb, recompressed)

    # Scale the difference for visual inspection
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 1
    if max_diff == 0:
        max_diff = 1

    # Enhance visual contrast
    scale_factor = scale if scale > 0 else (255.0 / max_diff)
    enhancer = ImageEnhance.Brightness(diff)
    diff_enhanced = enhancer.enhance(scale_factor)

    ela_visual = np.array(diff_enhanced)

    # Quantitative analysis: Standard deviation and local variance of difference map
    diff_gray = cv2.cvtColor(np.array(diff), cv2.COLOR_RGB2GRAY)
    ela_mean = np.mean(diff_gray)
    ela_std = np.std(diff_gray)

    # Natural uncompressed or uniform images have steady low error variance.
    # Inconsistent synthetic media or spliced sections produce high local variance.
    norm_score = float(np.clip((ela_std * 0.15) + (ela_mean * 0.05), 0.0, 1.0))

    return ela_visual, norm_score


def analyze_ela_artifacts(image: Image.Image, quality: int = 90) -> Dict[str, Any]:
    """
    Full ELA artifact inspection pipeline.
    """
    ela_vis, anomaly_score = compute_ela(image, quality=quality)

    return {
        "ela_anomaly_score": anomaly_score,
        "ela_visualization": ela_vis,
        "compression_quality_checked": quality,
        "is_anomalous": anomaly_score > 0.50
    }
