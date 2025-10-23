"""
Training script for mouse action recognition model.
Includes class weighting, checkpointing, and comprehensive metrics.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import json
from datetime import datetime

from data_loader import create_data_loaders
from model import create_model
from evaluation import evaluate, compute_metrics


class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device: str,
    scaler=None
) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns:
        Dictionary with loss and accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (clips, labels) in enumerate(train_loader):
        clips = clips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(clips)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Metrics
        total_loss += loss.item() * clips.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 20 == 0:
            print(
                f"  Batch {batch_idx}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}, "
                f"Acc: {100 * correct / total:.2f}%"
            )

    avg_loss = total_loss / total
    avg_acc = 100 * correct / total

    return {"loss": avg_loss, "accuracy": avg_acc}


def train_model(
    video_dir: str,
    annotation_dir: str,
    output_dir: str = "./checkpoints",
    num_epochs: int = 100,
    batch_size: int = 32,
    clip_length: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    model_type: str = "standard",
    use_amp: bool = True,
    random_seed: int = 42
):
    """
    Train action recognition model.

    Args:
        video_dir: Directory with video files
        annotation_dir: Directory with annotation CSVs
        output_dir: Where to save checkpoints
        num_epochs: Max training epochs
        batch_size: Batch size
        clip_length: Temporal clip length (must be odd)
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        model_type: "standard" or "light"
        use_amp: Use automatic mixed precision
        random_seed: Random seed
    """
    # Setup
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print("Loading data...")
    train_loader, val_loader, class_weights = create_data_loaders(
        video_dir, annotation_dir,
        batch_size=batch_size,
        clip_length=clip_length,
        stride=1,
        test_size=0.2,
        random_seed=random_seed
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Model
    print("Creating model...")
    model = create_model(
        num_classes=7,
        clip_length=clip_length,
        model_type=model_type,
        device=device
    )

    # Loss function with class weights
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Tensorboard
    writer = SummaryWriter(str(run_dir))

    # Training loop
    early_stopping = EarlyStopping(patience=15, min_delta=1e-4)
    best_val_f1 = 0.0

    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")

        # Validate
        print("Validating...")
        val_loss, val_acc, val_metrics = evaluate(
            model, val_loader, criterion, device
        )
        print(
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
            f"F1 (macro): {val_metrics['f1_macro']:.4f}"
        )

        # Log to tensorboard
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("F1/macro", val_metrics["f1_macro"], epoch)
        writer.add_scalar("F1/weighted", val_metrics["f1_weighted"], epoch)

        # Save checkpoint if best F1
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            checkpoint_path = run_dir / f"best_model_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": {
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "clip_length": clip_length,
                    "learning_rate": learning_rate,
                    "model_type": model_type
                }
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        # Learning rate scheduler
        scheduler.step()

        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    writer.close()

    # Save config
    config = {
        "video_dir": video_dir,
        "annotation_dir": annotation_dir,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "clip_length": clip_length,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "model_type": model_type,
        "use_amp": use_amp,
        "random_seed": random_seed,
        "best_val_f1": float(best_val_f1)
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete. Results saved to {run_dir}")
    return run_dir


if __name__ == "__main__":
    video_dir = "/Users/anagara8/Documents/prj_mouse_pain/Videos"
    annotation_dir = "/Users/anagara8/Documents/prj_mouse_pain/Annotations"

    train_model(
        video_dir,
        annotation_dir,
        output_dir="./checkpoints",
        num_epochs=100,
        batch_size=32,
        clip_length=16,
        learning_rate=1e-3,
        weight_decay=1e-5,
        model_type="standard",
        use_amp=True
    )
