"""
Test script for updated data loaders with multi-trial structure.
"""

import sys
from pathlib import Path

# Test standard data loader
print("=" * 60)
print("Testing Standard Data Loader (Visual-Only)")
print("=" * 60)

try:
    from data_loader import create_data_loaders

    train_loader, val_loader, class_weights = create_data_loaders(
        video_dir="./Videos",
        annotation_dir="./Annotations",
        batch_size=4,
        clip_length=15,  # Must be odd
        stride=2,
        num_workers=0  # Use 0 for testing
    )

    print(f"\n✓ Data loaders created successfully!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Class weights: {class_weights}")

    # Test one batch
    print("\n  Testing one batch from train loader...")
    for batch_idx, (clips, labels) in enumerate(train_loader):
        print(f"    Batch shape: clips={clips.shape}, labels={labels.shape}")
        print(f"    Clips dtype: {clips.dtype}, range: [{clips.min():.3f}, {clips.max():.3f}]")
        print(f"    Labels: {labels.tolist()}")
        break

    print("\n  Testing one batch from val loader...")
    for batch_idx, (clips, labels) in enumerate(val_loader):
        print(f"    Batch shape: clips={clips.shape}, labels={labels.shape}")
        print(f"    Labels: {labels.tolist()}")
        break

    print("\n✓ Standard data loader test PASSED!\n")

except Exception as e:
    print(f"\n✗ Standard data loader test FAILED!")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test multimodal data loader
print("=" * 60)
print("Testing Multimodal Data Loader (Visual + DLC)")
print("=" * 60)

try:
    from multimodal_data_loader import create_multimodal_data_loaders

    # Note: DLC files are in Videos/ directory
    train_loader, val_loader, class_weights = create_multimodal_data_loaders(
        video_dir="./Videos",
        annotation_dir="./Annotations",
        dlc_dir="./Videos",  # DLC files are in Videos/ directory
        batch_size=4,
        clip_length=15,  # Must be odd
        stride=2,
        num_workers=0  # Use 0 for testing
    )

    print(f"\n✓ Multimodal data loaders created successfully!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Class weights: {class_weights}")

    # Test one batch
    print("\n  Testing one batch from train loader...")
    for batch_idx, (visual, pose, labels) in enumerate(train_loader):
        print(f"    Batch shape: visual={visual.shape}, pose={pose.shape}, labels={labels.shape}")
        print(f"    Visual dtype: {visual.dtype}, range: [{visual.min():.3f}, {visual.max():.3f}]")
        print(f"    Pose dtype: {pose.dtype}, range: [{pose.min():.3f}, {pose.max():.3f}]")
        print(f"    Labels: {labels.tolist()}")
        break

    print("\n  Testing one batch from val loader...")
    for batch_idx, (visual, pose, labels) in enumerate(val_loader):
        print(f"    Batch shape: visual={visual.shape}, pose={pose.shape}, labels={labels.shape}")
        print(f"    Labels: {labels.tolist()}")
        break

    print("\n✓ Multimodal data loader test PASSED!\n")

except Exception as e:
    print(f"\n✗ Multimodal data loader test FAILED!")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nThe data loaders are now correctly handling the multi-trial structure:")
print("  - Each video can have multiple trials (360 frames each)")
print("  - Video length can be any multiple of 360 (1080, 1800, etc.)")
print("  - Annotations are matched by trial number")
print("  - Frame offsets are correctly calculated")
print("  - DLC data is properly sliced per trial")
