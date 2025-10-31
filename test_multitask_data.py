#!/usr/bin/env python3
"""Quick test script to verify data loading works correctly."""

import sys
sys.path.insert(0, '/media/data_cifs/projects/prj_mouse_pain/Dec24/prj_mouse_pain')

from train_multitask import MiceActionDatasetFromAnnotations, normalize_keypoint_name

# Test dataset creation
print("="*80)
print("Testing Dataset Creation")
print("="*80)

try:
    kp_names = [normalize_keypoint_name(name) for name in
                'mouth,tail_base,L_frontpaw,R_frontpaw,L_hindpaw,R_hindpaw'.split(',')]

    print(f"\nKeypoint names: {kp_names}")

    dataset = MiceActionDatasetFromAnnotations(
        annotations_dir='/media/data_cifs/projects/prj_mouse_pain/Dec24/prj_mouse_pain/Annotations',
        videos_dir='/media/data_cifs/projects/prj_mouse_pain/Dec24/prj_mouse_pain/Videos',
        kp_names=kp_names,
        split='train',
        train_T=180,
        val_T=240,
        img_size=224,
        rare_boost_cap=1.0,  # No boosting for test
        seed=123
    )

    print(f"\n✓ Dataset created successfully!")
    print(f"✓ Total samples: {len(dataset)}")
    print(f"✓ Merged classes: {dataset.merged_classes}")
    print(f"✓ Column mapping: {dataset.col_map}")

    if len(dataset) > 0:
        print("\n" + "="*80)
        print("Testing Sample Loading")
        print("="*80)

        sample = dataset[0]
        print(f"\n✓ Sample loaded successfully!")
        print(f"  - Video shape: {sample['video'].shape}")
        print(f"  - Actions shape: {sample['actions'].shape}")
        print(f"  - Action mask: {sample['action_mask']}")
        print(f"  - Keypoints shape: {sample['keypoints'].shape}")
        print(f"  - Visibility shape: {sample['visibility'].shape}")
        print(f"  - Time feats shape: {sample['time_feats'].shape}")

        # Check action distribution in this sample
        action_labels = sample['actions'].argmax(dim=-1)
        unique_actions = action_labels.unique()
        print(f"\n  - Unique action classes in sample: {unique_actions.tolist()}")
        for action_idx in unique_actions:
            class_name = dataset.merged_classes[action_idx]
            count = (action_labels == action_idx).sum().item()
            print(f"    - {class_name} ({action_idx}): {count} frames")

        print("\n" + "="*80)
        print("✓ All tests passed! Ready to train.")
        print("="*80)
    else:
        print("\n✗ Error: No samples found in dataset!")
        print("  Check that:")
        print("  1. Videos exist in Videos/ directory")
        print("  2. Action CSVs exist in Annotations/ with pattern 'prediction_<video>_<trial>.csv'")
        print("  3. Videos and CSVs have >= 360 frames/rows")

except Exception as e:
    print(f"\n✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
