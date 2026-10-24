"""
GPU-Accelerated Training Engine for Synthetic Media Detector
Implements Transfer Learning, Mixed Precision, Cosine Annealing, and Validation Checkpointing.
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.config import (
    TRAIN_DIR, VAL_DIR, BEST_MODEL_PATH, LATEST_MODEL_PATH,
    REPORTS_DIR, DEVICE, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    EPOCHS, EARLY_STOPPING_PATIENCE, DEFAULT_MODEL_NAME
)
from src.models.detector import build_detector, SyntheticMediaDetector
from src.utils.dataset_loader import create_dataloaders
from src.utils.metrics import calculate_metrics, plot_training_history


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="[Training]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with torch.amp.autocast(device_type=device if device == 'cuda' else 'cpu', enabled=(device == 'cuda')):
            logits = model(images)
            loss = criterion(logits, labels)

        # Scaled backward pass
        if device == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.2%}"})

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float, Dict[str, Any]]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device if device == 'cuda' else 'cpu', enabled=(device == 'cuda')):
                logits = model(images)
                loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=-1)[:, 1]

            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    total = len(all_targets)
    val_loss = running_loss / total
    val_acc = (np.array(all_preds) == np.array(all_targets)).mean() * 100.0

    metrics = calculate_metrics(np.array(all_targets), np.array(all_probs))
    return val_loss, val_acc, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Synthetic Media Detection Model")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Model backbone (efficientnet_b4, resnet50, convnext_tiny)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--fresh", action="store_true", default=False, help="Force fresh training ignoring existing checkpoint")
    args = parser.parse_args()

    # Resume from checkpoint if one exists (fine-tune with lower LR)
    should_resume = BEST_MODEL_PATH.exists() and not args.fresh
    if should_resume and args.lr == LEARNING_RATE:
        args.lr = 2e-5  # lower LR for fine-tuning

    print("=" * 65)
    print(" AI-DRIVEN SYNTHETIC MEDIA DETECTION ENGINE - MODEL TRAINING")
    print("=" * 65)
    print(f"[*] Target Architecture : {args.model}")
    print(f"[*] Hardware Device     : {DEVICE.upper()} ({torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'})")
    print(f"[*] Batch Size          : {args.batch_size}")
    print(f"[*] Learning Rate       : {args.lr} {'(fine-tune)' if should_resume else '(pretrain)'}")
    print(f"[*] Total Epochs        : {args.epochs}")
    print(f"[*] Mode                : {'Resume fine-tune from checkpoint' if should_resume else 'Fresh training from ImageNet'}")
    print("-" * 65)

    # 1. Prepare DataLoaders
    train_loader, val_loader, _ = create_dataloaders(TRAIN_DIR, VAL_DIR, batch_size=args.batch_size)
    print(f"[+] Loaded {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples.")

    # 2. Build Model
    model = build_detector(backbone_name=args.model,
                           pretrained=(not should_resume),
                           checkpoint_path=str(BEST_MODEL_PATH) if should_resume else None,
                           device=DEVICE)

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda'))

    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    start_time = time.time()

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n>>> Epoch {epoch}/{args.epochs} [LR: {scheduler.get_last_lr()[0]:.6f}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_loss, val_acc, val_metrics = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        epoch_duration = time.time() - epoch_start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"    Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.2f}% | Val F1: {val_metrics['f1_score']:.4f} | Val AUC: {val_metrics['roc_auc']:.4f} ({epoch_duration:.1f}s)")

        # Save Latest Checkpoint
        torch.save(model.state_dict(), LATEST_MODEL_PATH)

        # Check for Best Model
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_metrics['f1_score'] > best_val_f1):
            best_val_acc = val_acc
            best_val_f1 = val_metrics['f1_score']
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"    [*] Saved new BEST model checkpoint -> {BEST_MODEL_PATH.name} (Acc: {best_val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n[!] Early stopping triggered after {epoch} epochs (Patience: {EARLY_STOPPING_PATIENCE})")
                break

    total_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"[+] Training Completed in {total_time/60:.2f} minutes.")
    print(f"[+] Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"[+] Model saved to: {BEST_MODEL_PATH}")

    # Plot & Save Training Curves
    history_plot_path = REPORTS_DIR / "training_history.png"
    plot_training_history(history, history_plot_path)
    print(f"[+] Saved training history curves -> {history_plot_path}")
    print("=" * 65)


if __name__ == "__main__":
    from typing import Tuple, Dict, Any
    main()
