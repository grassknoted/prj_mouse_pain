# Multi-Task Training - All Issues Fixed ✓

## Quick Summary

All 6 major issues have been fixed:

1. ✅ **CSV pattern** - Changed `predictions_` → `prediction_`
2. ✅ **Action indices** - Using numeric indices (0-7) instead of string names
3. ✅ **Model input size** - Pass `img_size` to timm models with `dynamic_img_size=True`
4. ✅ **Keypoint names** - Updated to match DLC format: `mouth`, `L_frontpaw`, etc.
5. ✅ **DLC CSV discovery** - Search for files starting with video stem
6. ✅ **DLC parsing** - Robust multi-level header parsing with error handling

## Your Data Format

### Action CSVs (in Annotations/)
- Pattern: `prediction_<video_name>_<trial_number>.csv`
- Example: `prediction_shortened_2023-10-25_CFA_010_267M_tracking.mp4_1.csv`
- Format: Two columns (Frame, Action) where Action is 0-7

### DLC CSVs (in Videos/)
- Pattern: `<video_name>DLC_resnet50_pawtracking_comparisionMay7shuffle1_500000.csv`
- Format: Multi-level header (scorer, bodyparts, coords)
- Keypoints: `mouth`, `L_frontpaw`, `R_frontpaw`, `L_hindpaw`, `R_hindpaw`, `tail_base`

### Videos (in Videos/)
- Format: `.mp4` files
- Must have ≥360 frames

## Ready to Run

```bash
# Activate your environment (same one you use for train.py)
# conda activate your_env OR use your Python

# Basic training
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --epochs 30 \
    --batch_size 2

# Quick 2-epoch test
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --epochs 2 \
    --batch_size 2 \
    --train_T 60 \
    --val_T 60
```

## What to Expect

### 1. Discovery Phase
```
[info] Original classes: ['rest', 'paw_withdraw', 'paw_lick', 'paw_guard', ...]
[info] Merged classes: ['rest', 'paw_withdraw', 'paw_lick', 'paw_shake', ...]
[info] Discovering videos in Videos...
Discovering videos: 100%|████████| 10/10
[info] train samples before boosting: 25
```

### 2. Class Distribution
```
============================================================
Class Distribution BEFORE boosting
============================================================
Class                     Count      Share
------------------------------------------------------------
rest                     45678     54.32%
paw_withdraw               850      1.01%
paw_lick                  1234      1.47%
...
```

### 3. Backbone Loading
Either:
```
[info] Using VideoMAE2 (3D) backbone
```
Or (more likely):
```
[warn] VideoMAE2 not available (...), falling back to 2D path
[warn] Falling back to 2D path: collapsing clip_T frames with mean pooling.
[info] Using 2D backbone: vit_small_patch14_dinov2.lvd142m
```

### 4. Smoke Test
```
[info] Running smoke test...
  Video shape: torch.Size([180, 3, 224, 224])
  Actions shape: torch.Size([180, 7])
  Keypoints shape: torch.Size([180, 12])
  Visibility shape: torch.Size([180, 6])
  Action mask: 1.0
```

### 5. Training
```
Epoch 1/30
------------------------------------------------------------------------
Training: 100%|████████| 12/12 [01:23<00:00]
Train loss: 2.3456 (action: 2.1000, kp: 0.2456)
Validation: 100%|████████| 3/3 [00:18<00:00]
Val loss: 2.1234
Val macro F1: 0.3456
Val macro Acc: 0.5234
Val segment F1@0.3: 0.2987
Val keypoint MAE (norm): 0.0345
Per-class F1:
  rest: 0.7234
  paw_withdraw: 0.2456
  paw_lick: 0.1987
  paw_shake: 0.3123
  walk: 0.4567
  active: 0.3890
[info] Saved best checkpoint to ../Annotations/../best_model_multitask.pt
```

## Checkpoint Location

The best model is saved to:
```
<parent directory of --annotations>/best_model_multitask.pt
```

For example, if you run:
```bash
python train_multitask.py --annotations ./Annotations --videos ./Videos
```

The checkpoint will be saved to:
```
./best_model_multitask.pt
```

## Checkpoint Contents

```python
checkpoint = torch.load('best_model_multitask.pt')

# Available keys:
checkpoint['model_state_dict']       # Model weights
checkpoint['ema_state_dict']         # EMA weights (if --use_ema)
checkpoint['classes']                # 7 merged class names
checkpoint['orig_classes']           # 8 original class names
checkpoint['col_map']                # {0:0, 1:1, 2:2, 3:1, 4:3, 5:1, 6:4, 7:5}
checkpoint['merge_idxs']             # [1, 3, 5]
checkpoint['best_thresholds']        # Per-class optimal thresholds
checkpoint['kp_names']               # ['mouth', 'tail_base', ...]
checkpoint['val_metrics']            # All validation metrics
checkpoint['args']                   # All hyperparameters
```

## Common Issues & Solutions

### "No samples found"
**Cause**: Videos or CSVs don't meet requirements

**Check**:
1. Videos are `.mp4` files in `--videos` directory
2. Action CSVs match pattern `prediction_<video>_<trial>.csv` in `--annotations`
3. Videos have ≥360 frames
4. Action CSVs have ≥360 rows

**Debug**:
```bash
# Check videos
ls -lh ./Videos/*.mp4

# Check action CSVs
ls -lh ./Annotations/prediction_*.csv

# Count frames in a video
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 ./Videos/your_video.mp4

# Count rows in action CSV
wc -l ./Annotations/prediction_your_video.mp4_1.csv
```

### "Keypoint not found" warnings
**Cause**: DLC CSV keypoint names don't match

**Solution**: Check your DLC CSV header:
```bash
head -3 ./Videos/your_videoDLC_resnet50_....csv
```

If your keypoints have different names, update `--kp_names`:
```bash
python train_multitask.py \
    --kp_names your_keypoint1,your_keypoint2,... \
    ...
```

### "Input height doesn't match model"
**Cause**: Model expects different resolution

**Solution**: This should now be fixed with `dynamic_img_size=True`. If it still occurs, try:
```bash
python train_multitask.py \
    --img_size 384 \  # Try 384 instead of 224
    ...
```

### CUDA Out of Memory
**Solution**: Reduce batch size or window length:
```bash
python train_multitask.py \
    --batch_size 1 \
    --train_T 120 \
    --val_T 120 \
    ...
```

### ModuleNotFoundError
**Solution**: Activate the same environment you use for `train.py`:
```bash
# Find which Python you use for train.py
which python

# Make sure it's the one with all packages installed
python -c "import torch, cv2, timm, pandas; print('OK')"
```

## Advanced Options

### Use EMA for Better Generalization
```bash
python train_multitask.py \
    --use_ema \
    --ema_decay 0.999 \
    ...
```

### Adjust Class Balancing
```bash
python train_multitask.py \
    --rare_threshold_share 0.05 \  # Classify more classes as rare
    --rare_boost_cap 20.0 \         # Allow higher replication
    ...
```

### Tune Loss Weights
```bash
python train_multitask.py \
    --lambda_kp 2.0 \          # Keypoint loss weight
    --lambda_smooth 1e-3 \     # Temporal smoothness weight
    --focal_gamma 2.0 \        # Focal loss gamma
    ...
```

### Longer Training
```bash
python train_multitask.py \
    --epochs 100 \
    --warmup_epochs 5 \
    --freeze_backbone_epochs 5 \
    ...
```

## Files Created

1. **train_multitask.py** - Main training script (~1850 lines)
2. **test_multitask_data.py** - Data loading test
3. **debug_dlc_csv.py** - DLC CSV inspection tool
4. **MULTITASK_TRAINING_FIXED.md** - Detailed documentation
5. **FIXES_SUMMARY.md** - This file

## Next Steps

1. **Run training**: Start with a 2-epoch test to verify everything works
2. **Monitor progress**: Watch loss, F1 scores, and keypoint MAE
3. **Tune hyperparameters**: Adjust learning rate, loss weights, etc. based on results
4. **Full training**: Run for 30-50 epochs once you're satisfied with the setup
5. **Evaluate**: Use the checkpoint for inference on new videos

## Questions?

If you encounter any issues:

1. Check this file and MULTITASK_TRAINING_FIXED.md
2. Run the debug scripts:
   - `python test_multitask_data.py` (requires packages)
   - `python debug_dlc_csv.py <path_to_dlc_csv>` (requires pandas)
3. Verify your data format matches the examples above
4. Check that you're using the correct Python environment

Good luck with your training! 🚀
