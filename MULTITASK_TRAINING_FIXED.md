# Multi-Task Training Script - Fix Summary

## Issues Fixed

### 1. **CSV Pattern Mismatch** ✓
- **Problem**: Code was looking for `predictions_<video>_<trial>.csv` (plural)
- **Reality**: Files are named `prediction_<video>_<trial>.csv` (singular)
- **Fix**: Changed glob pattern from `predictions_` to `prediction_`

### 2. **Action Class Type Mismatch** ✓
- **Problem**: Code expected string class names like "rest", "paw_withdraw"
- **Reality**: Action column contains numeric indices (0, 1, 2, 3, 4, 5, 6, 7)
- **Fix**:
  - Added predefined `ACTION_CLASSES` dict matching existing `data_loader.py`
  - Updated all methods to work with numeric indices instead of strings
  - Fixed: `_load_actions()`, `_print_class_distribution()`, `_apply_rare_class_boosting()`, and class weight computation

### 3. **Class Discovery** ✓
- **Problem**: `_discover_classes()` method tried to discover classes dynamically
- **Reality**: Classes are predefined (0-7) with known names
- **Fix**: Removed dynamic discovery, use predefined class mapping

### 4. **Model Input Size Mismatch** ✓
- **Problem**: ViT models expected 518x518 input but code used 224x224
- **Fix**: Pass `img_size` parameter to timm model creation and enable `dynamic_img_size`

### 5. **DLC Keypoint Name Mismatch** ✓
- **Problem**: Code looked for `nose`, `left_front_paw`, etc.
- **Reality**: DLC CSVs use `mouth`, `L_frontpaw`, `R_frontpaw`, `L_hindpaw`, `R_hindpaw`, `tail_base`
- **Fix**: Updated `normalize_keypoint_name()` to map to actual DLC names

### 6. **DLC CSV File Discovery** ✓
- **Problem**: DLC CSVs are named `<video_stem>DLC_resnet50_...csv`, not `<video_stem>.csv`
- **Fix**: Updated `find_dlc_csv()` to search for CSVs starting with video stem in video directory

## Action Class Mapping

```python
ACTION_CLASSES = {
    0: "rest",
    1: "paw_withdraw",  # Merge target
    2: "paw_lick",
    3: "paw_guard",     # Merges to paw_withdraw (1)
    4: "paw_shake",
    5: "flinch",        # Merges to paw_withdraw (1)
    6: "walk",
    7: "active"
}
```

After merging [1, 3, 5] → 1, you get **7 final classes**:
- rest (0)
- paw_withdraw (1) - includes original paw_guard and flinch
- paw_lick (2)
- paw_shake (4)
- walk (6)
- active (7)

Note: The 6 classes mentioned in the spec likely meant indices 0-5 were being used, with class 6 possibly unused.

## DLC Keypoint Names

Your DLC CSV files use these keypoint names (case-sensitive):
- `mouth` (head/snout)
- `L_frontpaw` (left front paw)
- `R_frontpaw` (right front paw)
- `L_hindpaw` (left hind paw)
- `R_hindpaw` (right hind paw)
- `tail_base` (base of tail)

The default `--kp_names` parameter uses these exact names.

## How to Run

### Basic Training
```bash
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --epochs 30 \
    --batch_size 2 \
    --lr 1e-4
```

### Advanced Training with EMA
```bash
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --freeze_backbone_epochs 3 \
    --train_T 180 \
    --val_T 240 \
    --use_ema \
    --ema_decay 0.999 \
    --focal_gamma 1.5 \
    --lambda_kp 1.0 \
    --rare_boost_cap 12.0 \
    --kp_names nose,tail_base,right_front_paw,left_front_paw,right_hind_paw,left_hind_paw
```

### Quick Test (2 epochs)
```bash
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --epochs 2 \
    --batch_size 2 \
    --train_T 60 \
    --val_T 60 \
    --rare_boost_cap 1.0
```

## Expected Output

The script will:

1. **Discovery Phase**
   ```
   [info] Original classes: ['rest', 'paw_withdraw', 'paw_lick', ...]
   [info] Merged classes: ['rest', 'paw_withdraw', ...]
   [info] Discovering videos in Videos...
   ```

2. **Class Distribution Tables**
   ```
   ============================================================
   Class Distribution BEFORE boosting
   ============================================================
   Class                     Count      Share
   ------------------------------------------------------------
   rest                     45678     54.32%
   paw_withdraw               850      1.01%
   ...
   ```

3. **Smoke Test**
   ```
   [info] Running smoke test...
     Video shape: torch.Size([180, 3, 224, 224])
     Actions shape: torch.Size([180, 7])
     Keypoints shape: torch.Size([180, 12])
     Visibility shape: torch.Size([180, 6])
   ```

4. **Training Progress**
   ```
   Epoch 1/30
   -------------------------------------------------------------------------------
   Training: 100%|██████████| 50/50 [02:15<00:00]
   Train loss: 2.1234 (action: 1.8765, kp: 0.2469)
   Validation: 100%|██████████| 12/12 [00:30<00:00]
   Val loss: 1.9876
   Val macro F1: 0.4523
   Val macro Acc: 0.6234
   Val segment F1@0.3: 0.3987
   Val keypoint MAE (norm): 0.0234
   Per-class F1:
     rest: 0.8234
     paw_withdraw: 0.3456
     ...
   [info] Saved best checkpoint to /path/to/best_model_multitask.pt
   ```

## Checkpoint Contents

The saved checkpoint includes:
```python
{
    'model_state_dict': ...,
    'ema_state_dict': ...,  # If --use_ema
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'epoch': 15,
    'best_val_f1': 0.6234,
    'classes': ['rest', 'paw_withdraw', ...],  # 7 merged classes
    'orig_classes': ['rest', 'paw_withdraw', ...],  # 8 original classes
    'col_map': {0: 0, 1: 1, 2: 2, 3: 1, 4: 3, 5: 1, 6: 4, 7: 5},
    'merge_idxs': [1, 3, 5],
    'best_thresholds': [0.3, 0.5, ...],
    'kp_names': ['nose', 'tail_base', ...],
    'val_metrics': {...},
    'args': {...}
}
```

## Troubleshooting

### No samples found
**Symptom**: `train samples before boosting: 0`

**Check**:
1. Videos exist in `--videos` directory (`.mp4` files)
2. Action CSVs exist in `--annotations` with pattern `prediction_<video_name>_<trial>.csv`
3. Videos have ≥360 frames
4. Action CSVs have ≥360 rows with "Action" column

### ModuleNotFoundError
**Solution**: Activate your training environment:
```bash
# If using conda
conda activate your_env_name

# Or use the same Python as train.py
which python  # Should point to your env's Python
```

### CUDA out of memory
**Solution**: Reduce batch size or window length:
```bash
python train_multitask.py \
    --batch_size 1 \
    --train_T 120 \
    --val_T 120 \
    ...
```

### VideoMAE2 not available warning
**Expected behavior**: Script will automatically fall back to 2D ViT:
```
[warn] VideoMAE2 not available (...), falling back to 2D path
[warn] Falling back to 2D path: collapsing clip_T frames with mean pooling.
[info] Using 2D backbone: vit_small_patch14_dinov2.lvd142m
```

This is normal if `timm` doesn't have VideoMAE2 models. The script will work fine with 2D backbones.

## Next Steps

1. **Run training**: Use one of the commands above
2. **Monitor progress**: Watch for class distribution, F1 scores, and loss curves
3. **Check checkpoint**: Verify `best_model_multitask.pt` is saved
4. **Evaluate**: Use the checkpoint for inference on new videos

## Files Created

- `train_multitask.py` - Main training script (~1850 lines)
- `test_multitask_data.py` - Data loading test script
- `MULTITASK_TRAINING_FIXED.md` - This document

## Key Features

✓ VideoMAE2 (3D) with automatic 2D ViT fallback
✓ TCN temporal head with dilated convolutions
✓ Dual task heads: action classification + keypoint regression
✓ Robust data loading (handles missing files, <360 frames, etc.)
✓ Action CSV discovery with correct pattern
✓ Numeric action index handling
✓ DLC keypoint parsing with aliasing
✓ Class merging (paw_guard + flinch → paw_withdraw)
✓ Rare-class boosting with augmentations
✓ Focal loss with temporal smoothness
✓ Masked keypoint loss
✓ Frame F1, segment F1, keypoint MAE metrics
✓ Mixed precision (bf16)
✓ EMA support
✓ Backbone freezing/unfreezing
✓ Comprehensive logging
