"""
Color & Chrominance Distribution Forensic Analyzer
Examines YCbCr and HSV color space statistics for generative artifacts.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any


def analyze_chrominance_distribution(image: Image.Image) -> Dict[str, Any]:
    """
    Analyzes chrominance channel correlations (Cb, Cr) and saturation dispersion.
    AI generators often exhibit subtle imbalances between luminescence and chrominance.
    """
    rgb = np.array(image.convert("RGB"))
    ycbcr = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y_channel = ycbcr[:, :, 0].astype(np.float32)
    cr_channel = ycbcr[:, :, 1].astype(np.float32)
    cb_channel = ycbcr[:, :, 2].astype(np.float32)

    # Correlation between luminance and chrominance
    corr_y_cr = float(np.corrcoef(y_channel.flatten(), cr_channel.flatten())[0, 1])
    corr_y_cb = float(np.corrcoef(y_channel.flatten(), cb_channel.flatten())[0, 1])

    # HSV saturation analysis
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    sat_std = float(np.std(sat))
    sat_mean = float(np.mean(sat))

    # Anomaly calculation
    # Extreme oversaturation or unnatural chrominance decoupling from luminance indicates synthetic origin
    color_decoupling = abs(corr_y_cr) + abs(corr_y_cb)
    anomaly_score = float(np.clip((sat_std / 128.0) * 0.4 + (1.0 - (color_decoupling / 2.0)) * 0.3, 0.0, 1.0))

    return {
        "color_anomaly_score": anomaly_score,
        "chroma_correlation": (corr_y_cr + corr_y_cb) / 2.0,
        "saturation_dispersion": sat_std,
        "is_anomalous": anomaly_score > 0.50
    }
