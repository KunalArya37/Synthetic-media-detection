"""
Composite Authenticity Engine & Multi-Evidence Fusion Core
Combines Deep Learning Probabilities, FFT Spectral Anomaly, ELA, Chrominance, and Noise Signatures
with Adaptive Non-Linear Risk Fusion.
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional

from src.config import (
    DEVICE, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    WEIGHT_DL, WEIGHT_FFT, WEIGHT_ELA, WEIGHT_NOISE
)
from src.models.detector import SyntheticMediaDetector
from src.models.gradcam import GradCAM, generate_cam_overlay
from src.forensics.fft_analysis import analyze_spectral_anomalies
from src.forensics.ela_analysis import analyze_ela_artifacts
from src.forensics.noise_analysis import analyze_noise_artifacts
from src.forensics.color_analysis import analyze_chrominance_distribution


class MediaAuthenticator:
    """
    Forensic engine combining multiple orthogonal detection signals
    into a unified authenticity decision with adaptive non-linear risk fusion.
    """
    def __init__(self, model: SyntheticMediaDetector):
        self.model = model
        self.device = DEVICE
        self.model.to(self.device)
        self.model.eval()

        # Initialize Grad-CAM engine
        self.grad_cam = GradCAM(self.model, self.model.target_cam_layer)

        # Standard inference transform
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Transforms PIL image to normalized tensor (1, C, H, W)."""
        rgb_img = image.convert("RGB")
        tensor = self.transform(rgb_img).unsqueeze(0).to(self.device)
        return tensor

    def authenticate(
        self,
        image: Image.Image,
        run_gradcam: bool = True,
        threshold: float = 0.50,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Executes complete forensic pipeline and evidence fusion.
        Args:
            image: Input PIL Image
            run_gradcam: Whether to compute Grad-CAM heatmaps
            threshold: Synthetic decision boundary (default 0.50)
            custom_weights: Optional custom signal weights dictionary
        Returns:
            Structured forensic analysis dictionary
        """
        rgb_image = image.convert("RGB")
        input_tensor = self.preprocess_image(rgb_image)

        # 1. Deep Learning Model Inference + Uncertainty Estimation
        dl_prob_synthetic, uncertainty_std, dl_confidence = self.model.predict_with_uncertainty(input_tensor, num_samples=8)

        # 2. Physics & Forensic Artifact Signals
        fft_results = analyze_spectral_anomalies(rgb_image)
        ela_results = analyze_ela_artifacts(rgb_image)
        noise_results = analyze_noise_artifacts(rgb_image)
        color_results = analyze_chrominance_distribution(rgb_image)

        s_dl = float(dl_prob_synthetic)
        s_fft = float(fft_results["spectral_anomaly_score"])
        s_ela = float(ela_results["ela_anomaly_score"])
        s_noise = float(noise_results["noise_anomaly_score"])
        s_color = float(color_results["color_anomaly_score"])

        # 3. Adaptive Multi-Evidence Risk Fusion
        # If custom weights are provided from the UI sliders, apply them; otherwise use adaptive non-linear fusion
        if custom_weights:
            w_dl = custom_weights.get("dl", WEIGHT_DL)
            w_fft = custom_weights.get("fft", WEIGHT_FFT)
            w_ela = custom_weights.get("ela", WEIGHT_ELA)
            w_noise = custom_weights.get("noise", WEIGHT_NOISE)
            total_w = w_dl + w_fft + w_ela + w_noise
            composite_synthetic_score = ((w_dl * s_dl) + (w_fft * s_fft) + (w_ela * s_ela) + (w_noise * s_noise)) / max(1e-5, total_w)
        else:
            # Adaptive Fusion: In modern diffusion media (Midjourney, DALL-E, SD), web recompression often lowers ELA/Noise.
            # We use an adaptive combination where strong DL or spectral signals are not penalized by clean single-save compression.
            primary_ai_signal = max(s_dl, (0.65 * s_dl + 0.35 * s_fft))
            secondary_signals = (0.4 * s_fft) + (0.3 * s_color) + (0.3 * max(s_ela, s_noise))
            
            # Non-linear fusion
            composite_synthetic_score = (0.70 * primary_ai_signal) + (0.30 * secondary_signals)

        composite_synthetic_score = float(np.clip(composite_synthetic_score, 0.0, 1.0))
        composite_authentic_score = 1.0 - composite_synthetic_score

        # 4. Determine Verdict & Severity Ratings
        if composite_synthetic_score >= threshold:
            verdict = "Synthetic Media (AI-Generated)"
            verdict_code = "SYNTHETIC"
            badge_color = "red"
            verdict_summary = "High probability of generative AI synthesis, upsampling grid patterns, or deep visual manipulation."
        elif composite_synthetic_score <= (threshold - 0.15):
            verdict = "Authentic Media (Camera Capture)"
            verdict_code = "AUTHENTIC"
            badge_color = "green"
            verdict_summary = "Image exhibits natural sensor noise, consistent spectral decay, and genuine optical characteristics."
        else:
            verdict = "Suspicious / Inconclusive Manipulation"
            verdict_code = "SUSPICIOUS"
            badge_color = "orange"
            verdict_summary = "Ambiguous characteristics detected; possible partial editing, heavy post-processing, or web compression masking."

        # Severity levels for breakdown table
        def get_severity(score: float) -> str:
            if score >= 0.70:
                return "HIGH ANOMALY"
            elif score >= 0.40:
                return "MEDIUM"
            return "LOW / NATURAL"

        # 5. Grad-CAM Explainability Heatmap
        cam_overlay_rgb = None
        cam_heatmap = None
        if run_gradcam:
            try:
                target_cls = 1 if dl_prob_synthetic >= 0.40 else 0
                cam_heatmap = self.grad_cam.generate_heatmap(input_tensor, target_class=target_cls)
                cam_overlay_rgb, _ = generate_cam_overlay(rgb_image, cam_heatmap, alpha=0.45)
            except Exception as e:
                print(f"Grad-CAM notice: {e}")

        # Final composite confidence calculation
        overall_confidence = float(np.clip((dl_confidence * 0.7) + ((1.0 - abs(composite_synthetic_score - 0.5) * 2) * 0.3), 0.6, 0.99))

        return {
            "verdict": verdict,
            "verdict_code": verdict_code,
            "verdict_summary": verdict_summary,
            "badge_color": badge_color,
            "synthetic_probability": composite_synthetic_score * 100.0,
            "authenticity_score": composite_authentic_score * 100.0,
            "confidence": overall_confidence * 100.0,
            "uncertainty_margin": uncertainty_std * 100.0,
            "signals": {
                "deep_learning": {
                    "score": s_dl * 100.0,
                    "severity": get_severity(s_dl),
                    "weight": "55%"
                },
                "fft_spectral": {
                    "score": s_fft * 100.0,
                    "severity": get_severity(s_fft),
                    "weight": "20%",
                    "high_freq_roughness": fft_results.get("high_freq_roughness", 0.0),
                    "radial_frequencies": fft_results.get("frequencies", []),
                    "radial_power": fft_results.get("radial_profile", []),
                    "radial_profile": fft_results.get("radial_profile", []),
                    "visualization": fft_results.get("spectrum_visualization")
                },
                "ela_compression": {
                    "score": s_ela * 100.0,
                    "severity": get_severity(s_ela),
                    "weight": "15%",
                    "visualization": ela_results["ela_visualization"]
                },
                "sensor_noise": {
                    "score": s_noise * 100.0,
                    "severity": get_severity(s_noise),
                    "weight": "10%",
                    "visualization": noise_results["noise_visualization"]
                },
                "color_chrominance": {
                    "score": s_color * 100.0,
                    "severity": get_severity(s_color)
                }
            },
            "gradcam": {
                "heatmap": cam_heatmap,
                "overlay_rgb": cam_overlay_rgb
            }
        }
