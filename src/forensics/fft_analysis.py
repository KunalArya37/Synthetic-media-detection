"""
2D Fast Fourier Transform (FFT) & Azimuthal Spectral Anomaly Analyzer
Detects periodic high-frequency grid artifacts and upsampling signatures in synthetic media.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, Any


def compute_fft_spectrum(image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 2D Fast Fourier Transform and centered log-magnitude spectrum.
    Returns:
        magnitude_spectrum: Log-scaled 2D power spectrum [0 - 255 uint8] for visualization
        raw_spectrum: Float magnitude spectrum array
    """
    # Convert to grayscale
    gray = np.array(image.convert("L"), dtype=np.float32)
    h, w = gray.shape

    # Apply 2D FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)

    # Magnitude spectrum
    raw_magnitude = np.abs(f_shift)
    # Log scale to compress dynamic range: S = 20 * log(1 + |F|)
    log_magnitude = 20 * np.log(1.0 + raw_magnitude)

    # Normalize to 0-255 uint8 for visualization
    min_val, max_val = log_magnitude.min(), log_magnitude.max()
    if max_val - min_val > 1e-6:
        norm_spectrum = ((log_magnitude - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        norm_spectrum = np.zeros((h, w), dtype=np.uint8)

    return norm_spectrum, log_magnitude


def azimuthal_average(image_spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 1D radially averaged power spectrum profile (Azimuthal Integration).
    """
    h, w = image_spectrum.shape
    center_y, center_x = h // 2, w // 2

    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).astype(np.int32)

    # Compute radial bins
    max_radius = int(np.min([center_x, center_y]))
    radial_profile = np.zeros(max_radius)
    for radius in range(max_radius):
        mask = (r == radius)
        if np.any(mask):
            radial_profile[radius] = np.mean(image_spectrum[mask])

    frequencies = np.arange(max_radius) / float(max_radius)
    return frequencies, radial_profile


def analyze_spectral_anomalies(image: Image.Image) -> Dict[str, Any]:
    """
    Performs full frequency domain forensic analysis.
    Returns:
        Dictionary with anomaly score, frequency peaks, spectrum image, and radial profile.
    """
    vis_spectrum, log_mag = compute_fft_spectrum(image)
    freqs, radial_profile = azimuthal_average(log_mag)

    # Detect high-frequency spikes and deviation from natural 1/f power law
    # In natural images, radial profile decays smoothly. In GANs/Diffusion, discrete peaks occur at high frequencies.
    high_freq_cutoff = int(len(radial_profile) * 0.5)
    high_freq_region = radial_profile[high_freq_cutoff:]

    if len(high_freq_region) > 5:
        # Measure high-frequency variance & peak-to-average ratio
        diffs = np.abs(np.diff(high_freq_region))
        high_freq_roughness = float(np.std(diffs))
        peak_ratio = float(np.max(high_freq_region) / (np.mean(high_freq_region) + 1e-6))
        
        # Calibrate anomaly score to [0.0, 1.0]
        anomaly_score = float(np.clip((high_freq_roughness * 0.08) + (peak_ratio - 1.0) * 0.35, 0.0, 1.0))
    else:
        anomaly_score = 0.0
        high_freq_roughness = 0.0

    return {
        "spectral_anomaly_score": anomaly_score,
        "high_freq_roughness": high_freq_roughness,
        "spectrum_visualization": vis_spectrum,
        "frequencies": freqs.tolist(),
        "radial_profile": radial_profile.tolist(),
        "is_anomalous": anomaly_score > 0.55
    }
