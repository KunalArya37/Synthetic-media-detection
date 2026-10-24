"""
Comprehensive Evaluation Metrics & Visualization Module
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, EER, and plots publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss
)
from typing import Dict, Any, Tuple, Optional
from pathlib import Path


def calculate_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes all standard and forensic classification metrics.
    Args:
        y_true: Ground truth binary labels (0: Real, 1: Synthetic)
        y_pred_prob: Predicted probability for class 1 (Synthetic)
        threshold: Decision threshold (default 0.5)
    """
    y_pred = (y_pred_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_pred_prob)
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_prob)
        # Calculate Equal Error Rate (EER)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    except Exception:
        auc = 0.5
        fpr, tpr, roc_thresholds = np.array([0, 1]), np.array([0, 1]), np.array([0, 1])
        eer = 0.5

    try:
        prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_prob)
        ap = average_precision_score(y_true, y_pred_prob)
    except Exception:
        prec_curve, rec_curve, pr_thresholds = np.array([1, 0]), np.array([0, 1]), np.array([0, 1])
        ap = 0.5

    cm = confusion_matrix(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred_prob)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "eer": float(eer),
        "average_precision": float(ap),
        "brier_score": float(brier),
        "confusion_matrix": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision_curve": prec_curve.tolist(),
        "recall_curve": rec_curve.tolist(),
        "classification_report_str": classification_report(y_true, y_pred, target_names=["Authentic", "Synthetic"])
    }


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, title: str = "Confusion Matrix - Synthetic Media Detector"):
    """Saves formatted confusion matrix heatmap."""
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['Real', 'Synthetic'],
        yticklabels=['Real', 'Synthetic'],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(title, fontsize=12, pad=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11, labelpad=8)
    plt.ylabel('Ground Truth Label', fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, eer: float, save_path: Path):
    """Saves publication-quality ROC-AUC curve."""
    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(fpr, tpr, color='#1E88E5', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#9E9E9E', linestyle='--', lw=1.5, label='Random Chance (AUC = 0.50)')
    plt.plot([eer], [1 - eer], marker='o', markersize=7, color='#D81B60', label=f'EER Point ({eer:.2%})')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_training_history(history: Dict[str, list], save_path: Path):
    """Plots training and validation loss/accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    # Loss plot
    ax1.plot(epochs, history["train_loss"], 'o-', color='#1E88E5', label='Train Loss', lw=2)
    ax1.plot(epochs, history["val_loss"], 's-', color='#E53935', label='Val Loss', lw=2)
    ax1.set_title('Cross-Entropy Loss vs Epochs', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], 'o-', color='#43A047', label='Train Acc', lw=2)
    ax2.plot(epochs, history["val_acc"], 's-', color='#FB8C00', label='Val Acc', lw=2)
    ax2.set_title('Classification Accuracy vs Epochs', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
