"""
Comprehensive Evaluation & Benchmarking Script
Evaluates model on unseen test partition, calculates forensic metrics, and saves academic report figures.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.config import (
    TEST_DIR, BEST_MODEL_PATH, REPORTS_DIR, DEVICE,
    DEFAULT_MODEL_NAME, IMG_SIZE
)
from src.models.detector import build_detector
from src.utils.dataset_loader import create_dataloaders
from src.utils.metrics import (
    calculate_metrics, plot_confusion_matrix, plot_roc_curve
)


def evaluate_test_set(model_path: Path, model_name: str = DEFAULT_MODEL_NAME, test_dir: Path = TEST_DIR, split_name: str = "internal_test", batch_size: int = 16):
    print("=" * 65)
    print(f" SYNTHETIC MEDIA DETECTOR - BENCHMARK ({split_name.upper()})")
    print("=" * 65)
    print(f"[*] Checkpoint Model : {model_path}")
    print(f"[*] Evaluation Device: {DEVICE.upper()}")
    print(f"[*] Test Partition   : {test_dir}")
    print("-" * 65)

    # 1. Load Data
    _, _, test_loader = create_dataloaders(test_dir, test_dir, test_dir=test_dir, batch_size=batch_size)
    if test_loader is None or len(test_loader.dataset) == 0:
        print(f"[!] No test images found in {test_dir}.")
        return None

    print(f"[+] Loaded {len(test_loader.dataset)} test samples.")

    # 2. Build & Load Model
    model = build_detector(backbone_name=model_name, pretrained=False, checkpoint_path=str(model_path), device=DEVICE)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    # 3. Inference
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"[{split_name.upper()} Evaluation]"):
            images = images.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type=DEVICE if DEVICE == 'cuda' else 'cpu', enabled=(DEVICE == 'cuda')):
                logits = model(images)
            
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)

    # 4. Calculate Academic Metrics
    metrics = calculate_metrics(y_true, y_prob)

    print("\n" + "=" * 65)
    print(f" {split_name.upper()} PERFORMANCE RESULTS")
    print("=" * 65)
    print(f"  • Overall Accuracy      : {metrics['accuracy'] * 100:.2f}%")
    print(f"  • Precision (Synthetic) : {metrics['precision'] * 100:.2f}%")
    print(f"  • Recall (Sensitivity)  : {metrics['recall'] * 100:.2f}%")
    print(f"  • F1-Score              : {metrics['f1_score']:.4f}")
    print(f"  • ROC-AUC Score         : {metrics['roc_auc']:.4f}")
    print(f"  • Equal Error Rate (EER): {metrics['eer'] * 100:.2f}%")
    print(f"  • Average Precision (AP): {metrics['average_precision']:.4f}")
    print(f"  • Probability Calibration Brier Score: {metrics['brier_score']:.4f}")
    print("-" * 65)
    print("\nDetailed Classification Breakdown:\n")
    print(metrics['classification_report_str'])
    print("-" * 65)

    # 5. Save Visual Figures & Reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{split_name}_" if split_name != "internal_test" else ""
    cm_path = REPORTS_DIR / f"{prefix}confusion_matrix.png"
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), cm_path)
    print(f"[+] Saved Confusion Matrix -> {cm_path}")

    roc_path = REPORTS_DIR / f"{prefix}roc_curve.png"
    plot_roc_curve(np.array(metrics["fpr"]), np.array(metrics["tpr"]), metrics["roc_auc"], metrics["eer"], roc_path)
    print(f"[+] Saved ROC-AUC Curve -> {roc_path}")

    json_path = REPORTS_DIR / f"{prefix}evaluation_summary.json"
    with open(json_path, "w") as f:
        json.dump({k: v for k, v in metrics.items() if not isinstance(v, (np.ndarray,))}, f, indent=4)
    print(f"[+] Saved Metrics Summary -> {json_path}")
    print("=" * 65)
    return metrics


if __name__ == "__main__":
    from src.config import EXTERNAL_TEST_DIR
    parser = argparse.ArgumentParser(description="Evaluate Synthetic Media Detection Model")
    parser.add_argument("--model-path", type=str, default=str(BEST_MODEL_PATH), help="Path to saved model checkpoint")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Model backbone name")
    parser.add_argument("--test-dir", type=str, default=None, help="Optional custom test directory")
    parser.add_argument("--eval-all", action="store_true", default=True, help="Evaluate both internal GenImage and external Synthbuster test sets")
    args = parser.parse_args()

    model_p = Path(args.model_path)
    if args.test_dir:
        evaluate_test_set(model_p, args.model_name, Path(args.test_dir), split_name="custom_test")
    else:
        print("\n[1/2] Evaluating Internal Test Set (GenImage)...")
        evaluate_test_set(model_p, args.model_name, TEST_DIR, split_name="internal_test")
        
        if EXTERNAL_TEST_DIR.exists() and len(list(EXTERNAL_TEST_DIR.glob("*/*.*"))) > 0:
            print("\n[2/2] Evaluating External Test Set (Synthbuster + RAISE)...")
            evaluate_test_set(model_p, args.model_name, EXTERNAL_TEST_DIR, split_name="external_synthbuster_raise")

