"""
High-Frequency Sensor Noise & PRNU Residual Forensic Analyzer
Detects lack of authentic camera sensor noise or artificial patchy noise textures.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, Any


def extract_noise_residuals(image: Image.Image) -> Tuple[np.ndarray, float]:
    """
    Extracts high-frequency noise residuals by subtracting a median-filtered denoised image
    from the original grayscale image, then computing local variance distribution.
    Returns:
        noise_visual: Amplified noise residual image [0-255 uint8]
        anomaly_score: Score [0.0 - 1.0] indicating artificial noise inconsistency
    """
    gray = np.array(image.convert("L"), dtype=np.float32)

    # Apply 3x3 median filter to estimate the denoised structural component
    denoised = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)

    # Noise residual: R = I - Denoised(I)
    residual = gray - denoised

    # Amplified residual for visualization
    res_amplified = np.clip((residual + 128.0), 0, 255).astype(np.uint8)

    # Compute Laplacian variance for high-frequency sharpness distribution
    laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
    lap_var = float(laplacian.var())

    # Spatial noise homogeneity test: compute local variance across 16x16 blocks
    h, w = gray.shape
    block_size = 16
    block_vars = []

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            patch = residual[y:y+block_size, x:x+block_size]
            block_vars.append(np.var(patch))

    if len(block_vars) > 0:
        var_of_vars = float(np.std(block_vars))
        mean_var = float(np.mean(block_vars) + 1e-6)
        noise_inhomogeneity = var_of_vars / mean_var
    else:
        noise_inhomogeneity = 0.0

    # In natural photos, sensor noise is evenly distributed across similar luminance.
    # In AI media, noise varies wildly or is completely missing in synthesized patches.
    anomaly_score = float(np.clip(noise_inhomogeneity * 0.35 + (0.1 if lap_var < 50 else 0.0), 0.0, 1.0))

    return res_amplified, anomaly_score


def analyze_noise_artifacts(image: Image.Image) -> Dict[str, Any]:
    """
    Full noise forensic pipeline.
    """
    res_vis, anomaly_score = extract_noise_residuals(image)

    return {
        "noise_anomaly_score": anomaly_score,
        "noise_visualization": res_vis,
        "is_anomalous": anomaly_score > 0.45
    }
