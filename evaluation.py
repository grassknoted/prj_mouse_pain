"""
Evaluation metrics for action recognition.
Includes F1 scores, confusion matrix, per-class metrics, and visualizations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Class labels for reference
CLASS_NAMES = [
    "rest",
    "paw_withdraw",
    "paw_lick",
    "paw_shake",
    "walk",
    "active"
]

NUM_CLASSES = len(CLASS_NAMES)


def evaluate(
    model: nn.Module,
    val_loader,
    criterion,
    device: str
) -> Tuple[float, float, Dict]:
    """
    Evaluate model on validation set.

    Returns:
        val_loss, val_accuracy, detailed_metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for clips, labels in val_loader:
            clips = clips.to(device)
            labels = labels.to(device)

            logits = model(clips)
            loss = criterion(logits, labels)

            total_loss += loss.item() * clips.size(0)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    val_loss = total_loss / len(all_labels)
    val_acc = 100 * np.mean(all_preds == all_labels)

    metrics = compute_metrics(all_preds, all_labels)

    return val_loss, val_acc, metrics


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        preds: Predicted labels (N,)
        labels: Ground truth labels (N,)

    Returns:
        Dictionary with various metrics
    """
    metrics = {}

    # F1 scores
    metrics["f1_macro"] = f1_score(labels, preds, average="macro", zero_division=0)
    metrics["f1_weighted"] = f1_score(labels, preds, average="weighted", zero_division=0)
    metrics["f1_micro"] = f1_score(labels, preds, average="micro", zero_division=0)

    # Precision and recall
    metrics["precision_macro"] = precision_score(labels, preds, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(labels, preds, average="macro", zero_division=0)

    # Per-class F1
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    for i, name in enumerate(CLASS_NAMES):
        metrics[f"f1_{name}"] = f1_per_class[i]

    # Per-class precision and recall
    precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0)

    for i, name in enumerate(CLASS_NAMES):
        metrics[f"precision_{name}"] = precision_per_class[i]
        metrics[f"recall_{name}"] = recall_per_class[i]

    return metrics


def print_detailed_report(preds: np.ndarray, labels: np.ndarray):
    """Print detailed classification report."""
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))
    print("="*80)


def plot_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    save_path: str = None,
    title: str = "Confusion Matrix"
):
    """
    Plot and optionally save confusion matrix.
    """
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Count"}
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")

    plt.show()


def plot_per_class_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    save_path: str = None
):
    """Plot per-class precision, recall, and F1 scores."""
    metrics = compute_metrics(preds, labels)

    f1_scores = [metrics[f"f1_{name}"] for name in CLASS_NAMES]
    precision_scores = [metrics[f"precision_{name}"] for name in CLASS_NAMES]
    recall_scores = [metrics[f"recall_{name}"] for name in CLASS_NAMES]

    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, precision_scores, width, label="Precision", alpha=0.8)
    bars2 = ax.bar(x, recall_scores, width, label="Recall", alpha=0.8)
    bars3 = ax.bar(x + width, f1_scores, width, label="F1", alpha=0.8)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim([0, 1.1])

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, height,
                    f"{height:.2f}",
                    ha="center", va="bottom", fontsize=9
                )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved per-class metrics plot to {save_path}")

    plt.show()


def plot_class_distribution(
    labels: np.ndarray,
    save_path: str = None,
    title: str = "Class Distribution"
):
    """Plot class distribution."""
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(CLASS_NAMES)), [counts[i] if i in unique else 0 for i in range(len(CLASS_NAMES))], alpha=0.7)
    plt.xlabel("Action Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")

    # Add count labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height,
                    f"{int(height)}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved class distribution plot to {save_path}")

    plt.show()


def evaluate_and_save_results(
    model: nn.Module,
    val_loader,
    device: str,
    output_dir: str
):
    """
    Complete evaluation with visualizations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for clips, labels in val_loader:
            clips = clips.to(device)
            logits = model(clips)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Print detailed report
    print_detailed_report(all_preds, all_labels)

    # Save metrics
    metrics = compute_metrics(all_preds, all_labels)
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        for key, value in sorted(metrics.items()):
            f.write(f"{key}: {value:.4f}\n")

    # Plot confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(all_preds, all_labels, save_path=str(cm_path))

    # Plot per-class metrics
    metrics_path = output_dir / "per_class_metrics.png"
    plot_per_class_metrics(all_preds, all_labels, save_path=str(metrics_path))

    # Plot class distribution
    dist_path = output_dir / "class_distribution.png"
    plot_class_distribution(all_labels, save_path=str(dist_path), title="Validation Set Distribution")

    print(f"\nResults saved to {output_dir}")

    return metrics


if __name__ == "__main__":
    # Test metrics computation
    preds = np.array([0, 1, 1, 2, 0, 1, 2, 0, 1, 1])
    labels = np.array([0, 1, 0, 2, 0, 1, 1, 0, 1, 2])

    metrics = compute_metrics(preds, labels)
    print("Metrics:", metrics)
    print_detailed_report(preds, labels)
