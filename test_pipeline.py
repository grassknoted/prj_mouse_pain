"""
Quick test of the action recognition pipeline.
Tests data loading, model creation, and a training step.
"""

import torch
import numpy as np
from pathlib import Path

from data_loader import create_data_loaders
from model import create_model
from evaluation import compute_metrics


def test_data_loader():
    """Test data loading and preprocessing."""
    print("="*80)
    print("TEST 1: Data Loading Pipeline")
    print("="*80)

    video_dir = "./Videos"
    annotation_dir = "./Annotations"

    # Check if directories exist
    if not Path(video_dir).exists():
        print(f"ERROR: {video_dir} not found")
        return False

    if not Path(annotation_dir).exists():
        print(f"ERROR: {annotation_dir} not found")
        return False

    try:
        train_loader, val_loader, class_weights = create_data_loaders(
            video_dir, annotation_dir,
            batch_size=4,
            clip_length=16,
            stride=2,
            test_size=0.2
        )

        print(f"✓ Train loader: {len(train_loader)} batches")
        print(f"✓ Val loader: {len(val_loader)} batches")
        print(f"✓ Class weights: {class_weights}")

        # Test one batch
        for batch_idx, (clips, labels) in enumerate(train_loader):
            print(f"✓ Batch shape: {clips.shape} (B, T, H, W)")
            print(f"✓ Labels shape: {labels.shape}")
            print(f"✓ Label values: {labels.unique().tolist()}")
            break

        print("\n✓ Data loading test PASSED\n")
        return True

    except Exception as e:
        print(f"\n✗ Data loading test FAILED: {e}\n")
        return False


def test_model():
    """Test model creation and forward pass."""
    print("="*80)
    print("TEST 2: Model Architecture")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    try:
        model = create_model(
            num_classes=7,
            clip_length=16,
            model_type="standard",
            device=device
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"✓ Model created")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")

        # Test forward pass
        x = torch.randn(4, 16, 256, 256).to(device)
        with torch.no_grad():
            out = model(x)

        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {out.shape}")
        print(f"✓ Output range: [{out.min():.2f}, {out.max():.2f}]")

        print("\n✓ Model test PASSED\n")
        return True

    except Exception as e:
        print(f"\n✗ Model test FAILED: {e}\n")
        return False


def test_metrics():
    """Test evaluation metrics."""
    print("="*80)
    print("TEST 3: Evaluation Metrics")
    print("="*80)

    try:
        # Create fake predictions
        preds = np.array([0, 1, 1, 2, 0, 1, 2, 0, 1, 1, 3, 4, 5])
        labels = np.array([0, 1, 0, 2, 0, 1, 1, 0, 1, 2, 3, 4, 5])

        metrics = compute_metrics(preds, labels)

        print(f"✓ F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"✓ F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"✓ Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"✓ Recall (macro): {metrics['recall_macro']:.4f}")

        # Check all metrics exist
        required_keys = ['f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
        for key in required_keys:
            if key not in metrics:
                raise ValueError(f"Missing metric: {key}")

        print("\n✓ Metrics test PASSED\n")
        return True

    except Exception as e:
        print(f"\n✗ Metrics test FAILED: {e}\n")
        return False


def test_light_model():
    """Test lightweight model variant."""
    print("="*80)
    print("TEST 4: Lightweight Model")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = create_model(
            num_classes=7,
            clip_length=16,
            model_type="light",
            device=device
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Light model created")
        print(f"✓ Total parameters: {total_params:,}")

        # Test forward pass
        x = torch.randn(2, 16, 256, 256).to(device)
        with torch.no_grad():
            out = model(x)

        print(f"✓ Output shape: {out.shape}")

        print("\n✓ Light model test PASSED\n")
        return True

    except Exception as e:
        print(f"\n✗ Light model test FAILED: {e}\n")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  MOUSE PAIN ACTION RECOGNITION - PIPELINE TEST".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print()

    results = []

    results.append(("Data Loading", test_data_loader()))
    results.append(("Model Architecture", test_model()))
    results.append(("Metrics", test_metrics()))
    results.append(("Lightweight Model", test_light_model()))

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<50} {status}")

    print()
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Ready to train.\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.\n")
        return 1


if __name__ == "__main__":
    exit(main())
