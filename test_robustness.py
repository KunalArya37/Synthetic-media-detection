"""
Robustness & Generalization Stress-Testing Suite
Tests detector performance against real-world social media degradations:
JPEG compression, Gaussian blur, resizing/downsampling, and sensor noise perturbations.
"""

import sys
import io
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.config import (
    TEST_DIR, BEST_MODEL_PATH, REPORTS_DIR, DEVICE,
    DEFAULT_MODEL_NAME, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
)
from src.models.detector import build_detector


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))


def apply_resize_down_up(image: Image.Image, scale_factor: float) -> Image.Image:
    w, h = image.size
    new_w, new_h = max(16, int(w * scale_factor)), max(16, int(h * scale_factor))
    down = image.convert("RGB").resize((new_w, new_h), Image.Resampling.BILINEAR)
    up = down.resize((w, h), Image.Resampling.BICUBIC)
    return up


def apply_additive_noise(image: Image.Image, std: float) -> Image.Image:
    img_np = np.array(image.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0, std, img_np.shape)
    noisy_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_np)


def evaluate_perturbation(model, samples: List[Tuple[Path, int]], perturb_fn, transform) -> float:
    correct = 0
    total = len(samples)

    with torch.no_grad():
        for img_path, label in samples:
            try:
                img = Image.open(img_path).convert("RGB")
                img_pert = perturb_fn(img)
                tensor = transform(img_pert).unsqueeze(0).to(DEVICE)
                with torch.amp.autocast(device_type=DEVICE if DEVICE == 'cuda' else 'cpu', enabled=(DEVICE == 'cuda')):
                    logits = model(tensor)
                pred = torch.argmax(logits, dim=-1).item()
                if pred == label:
                    correct += 1
            except Exception:
                pass

    return (correct / max(1, total)) * 100.0


def run_robustness_suite(model_path: Path, model_name: str = DEFAULT_MODEL_NAME):
    print("=" * 65)
    print(" SYNTHETIC MEDIA DETECTOR - ROBUSTNESS BENCHMARK SUITE")
    print("=" * 65)
    print(f"[*] Loading Model Checkpoint : {model_path}")
    print(f"[*] Evaluation Device        : {DEVICE.upper()}")
    print("-" * 65)

    # 1. Collect test samples
    samples = []
    for cls_name, label in [("real", 0), ("synthetic", 1)]:
        cls_dir = TEST_DIR / cls_name
        if cls_dir.exists():
            for p in cls_dir.glob("*.jpg"):
                samples.append((p, label))

    if not samples:
        print("[!] No test samples found. Run dataset_downloader.py first.")
        return

    # 2. Load model
    model = build_detector(backbone_name=model_name, pretrained=False, checkpoint_path=str(model_path), device=DEVICE)
    model.eval()

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    results = {}

    # Experiment 1: JPEG Compression Resistance
    print("\n>>> Testing JPEG Compression Resistance...")
    jpeg_qualities = [95, 80, 65, 50, 35, 20]
    jpeg_accs = []
    for q in tqdm(jpeg_qualities, desc="JPEG Qualities"):
        acc = evaluate_perturbation(model, samples, lambda img, q=q: apply_jpeg_compression(img, q), eval_transform)
        jpeg_accs.append(acc)
    results["jpeg_compression"] = {"qualities": jpeg_qualities, "accuracies": jpeg_accs}

    # Experiment 2: Gaussian Blur Resistance
    print("\n>>> Testing Gaussian Blur Resistance...")
    blur_radii = [0.0, 1.0, 2.0, 3.0, 4.0]
    blur_accs = []
    for r in tqdm(blur_radii, desc="Blur Radii"):
        acc = evaluate_perturbation(model, samples, lambda img, r=r: apply_gaussian_blur(img, r), eval_transform)
        blur_accs.append(acc)
    results["gaussian_blur"] = {"radii": blur_radii, "accuracies": blur_accs}

    # Experiment 3: Resizing / Downsampling Resistance
    print("\n>>> Testing Downsampling/Upsampling Degradation...")
    scales = [1.0, 0.75, 0.50, 0.35, 0.25]
    scale_accs = []
    for s in tqdm(scales, desc="Scale Factors"):
        acc = evaluate_perturbation(model, samples, lambda img, s=s: apply_resize_down_up(img, s), eval_transform)
        scale_accs.append(acc)
    results["downsampling"] = {"scales": scales, "accuracies": scale_accs}

    # Experiment 4: Additive Noise Resistance
    print("\n>>> Testing Additive Noise Resistance...")
    noise_stds = [0, 5, 10, 20, 30]
    noise_accs = []
    for n in tqdm(noise_stds, desc="Noise Stds"):
        acc = evaluate_perturbation(model, samples, lambda img, n=n: apply_additive_noise(img, n), eval_transform)
        noise_accs.append(acc)
    results["additive_noise"] = {"noise_stds": noise_stds, "accuracies": noise_accs}

    # 3. Plot Publication-Quality Robustness Degradation Chart
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=300)

    # Plot 1: JPEG
    axes[0, 0].plot(jpeg_qualities, jpeg_accs, 'o-', color='#0284C7', lw=2.5)
    axes[0, 0].set_title('JPEG Compression Resistance', fontweight='bold')
    axes[0, 0].set_xlabel('JPEG Quality Factor (Higher is Better)')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_ylim([40, 105])
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)

    # Plot 2: Blur
    axes[0, 1].plot(blur_radii, blur_accs, 's-', color='#E11D48', lw=2.5)
    axes[0, 1].set_title('Gaussian Blur Resistance', fontweight='bold')
    axes[0, 1].set_xlabel('Blur Radius (pixels)')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_ylim([40, 105])
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)

    # Plot 3: Downsampling
    axes[1, 0].plot(scales, scale_accs, '^-', color='#16A34A', lw=2.5)
    axes[1, 0].set_title('Downsampling & Resizing Degradation', fontweight='bold')
    axes[1, 0].set_xlabel('Resolution Scale Factor')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_ylim([40, 105])
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)

    # Plot 4: Noise
    axes[1, 1].plot(noise_stds, noise_accs, 'd-', color='#7C3AED', lw=2.5)
    axes[1, 1].set_title('Additive Sensor Noise Resistance', fontweight='bold')
    axes[1, 1].set_xlabel('Gaussian Noise Std Dev')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_ylim([40, 105])
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)

    plt.suptitle('Synthetic Media Detector - Robustness Stress Test Profiles', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plot_path = REPORTS_DIR / "robustness_benchmarks.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    json_path = REPORTS_DIR / "robustness_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 65)
    print(f"[+] Robustness Benchmark Completed.")
    print(f"[+] Saved Robustness Degradation Chart -> {plot_path}")
    print(f"[+] Saved Robustness Summary JSON     -> {json_path}")
    print("=" * 65)


if __name__ == "__main__":
    from typing import Tuple
    parser = argparse.ArgumentParser(description="Test Synthetic Media Detector Robustness")
    parser.add_argument("--model-path", type=str, default=str(BEST_MODEL_PATH), help="Path to model checkpoint")
    args = parser.parse_args()
    run_robustness_suite(Path(args.model_path))
