# Pose Graph Features and Model Types - Complete Guide

## Summary of Latest Changes

The multi-task training script now uses **pose graph features** instead of raw keypoint coordinates for action prediction. Two separate model types are available:

1. **Action-only model**: Predicts actions only (faster, focused)
2. **Multi-task model**: Predicts actions + keypoints (joint learning)

---

## What are Pose Graph Features?

### Problem with Raw Coordinates

Raw (x, y) keypoint coordinates have issues:
- **Not translation invariant**: Same pose at different positions = different coordinates
- **Not scale invariant**: Same pose at different scales = different coordinates
- **Not rotation invariant**: Same pose rotated = different coordinates

### Solution: Geometric Features

Instead of raw coordinates, we compute 18 geometric features:

#### 8 Edge Lengths (distances between keypoint pairs):
1. mouth ↔ tail_base
2. mouth ↔ L_frontpaw
3. mouth ↔ R_frontpaw
4. tail_base ↔ L_hindpaw
5. tail_base ↔ R_hindpaw
6. L_frontpaw ↔ L_hindpaw
7. R_frontpaw ↔ R_hindpaw
8. L_frontpaw ↔ R_frontpaw

#### 10 Angles (angles formed by keypoint triplets):
1. tail_base - mouth - L_frontpaw
2. tail_base - mouth - R_frontpaw
3. mouth - tail_base - L_hindpaw
4. mouth - tail_base - R_hindpaw
5. mouth - L_frontpaw - L_hindpaw
6. mouth - R_frontpaw - R_hindpaw
7. tail_base - L_frontpaw - L_hindpaw
8. tail_base - R_frontpaw - R_hindpaw
9. L_frontpaw - mouth - R_frontpaw
10. L_hindpaw - tail_base - R_hindpaw

### Benefits

These features are:
- ✅ Translation invariant (same regardless of position)
- ✅ Scale invariant (normalized by body size)
- ✅ Rotation invariant (angles and relative distances)
- ✅ More informative for action recognition

---

## Two Model Types

### 1. Action-Only Model (`--model_type action_only`)

**What it does:**
- Takes video frames + pose graph features
- Predicts only action labels (no keypoint coordinates)
- Uses pose features concatenated with visual features

**Architecture:**
```
Video frames (B, T, 3, H, W)
    ↓
Visual backbone (VideoMAE2 or ViT)
    ↓
Visual features (B, T, D)
    ↓
    + Pose graph features (B, T, 18) → Project to 128D
    ↓
Concatenate (B, T, D+128)
    ↓
TCN (temporal modeling)
    ↓
Action head
    ↓
Action logits (B, T, num_classes)
```

**When to use:**
- Focus only on action recognition
- Don't need keypoint predictions
- Faster training (no keypoint loss)
- Simpler model (fewer parameters)

**Training command:**
```bash
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --model_type action_only \
    --epochs 200 \
    --batch_size 2
```

### 2. Multi-Task Model (`--model_type multitask`)

**What it does:**
- Takes video frames + pose graph features
- Predicts action labels + keypoint coordinates
- Uses pose features for actions, predicts coords as auxiliary task

**Architecture:**
```
Video frames (B, T, 3, H, W)
    ↓
Visual backbone (VideoMAE2 or ViT)
    ↓
Visual features (B, T, D)
    ↓
    + Pose graph features (B, T, 18) → Project to 128D
    ↓
Concatenate (B, T, D+128)
    ↓
TCN (temporal modeling)
    ↓
Shared features (B, T, hidden_dim)
    ↓
    ├─→ Action head → Action logits (B, T, num_classes)
    └─→ Keypoint head → Keypoint coords (B, T, 2K)
```

**When to use:**
- Want both actions and keypoint predictions
- Joint training for better feature learning
- Keypoint prediction as auxiliary task helps action recognition

**Training command:**
```bash
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --model_type multitask \
    --epochs 200 \
    --batch_size 2 \
    --lambda_kp 1.0  # Keypoint loss weight
```

---

## Data Pipeline with Pose Graph

### Before (Old):
```python
1. Load keypoints: (T, 12) raw coords [x1, y1, x2, y2, ...]
2. Pass to model directly
3. Model uses raw coords for action prediction
```

### After (New):
```python
1. Load keypoints: (T, K, 2) as 2D coordinates
2. Apply jittering (train only): Add Gaussian noise
3. Compute pose graph features: (T, 18)
   - 8 edge lengths
   - 10 angles
4. Pass to model:
   - pose_features (T, 18) → for action prediction
   - keypoints_flat (T, 2K) → for coordinate prediction (multitask only)
```

---

## Keypoint Coordinate Jittering

### What it does

During training, adds small Gaussian noise to keypoint coordinates **before** computing pose graph features:

```python
# Original keypoint: (x, y) = (0.5, 0.3)
# Add noise: N(0, σ) where σ = 0.02 (2% of image size)
keypoints_jittered = keypoints + noise
keypoints_jittered = clip(keypoints_jittered, 0, 1)

# Then compute pose graph from jittered keypoints
pose_features = compute_pose_graph(keypoints_jittered)
```

### Why it helps

- More robust to slight pose variations
- Data augmentation for keypoint-dependent features
- Better generalization to new poses

### Usage

```bash
# Enable jittering (default: 0.02)
python train_multitask.py --kp_jitter_std 0.02 ...

# Disable jittering
python train_multitask.py --kp_jitter_std 0.0 ...

# Stronger jittering (more augmentation)
python train_multitask.py --kp_jitter_std 0.05 ...
```

---

## Metric Reporting

### Action-Only Model

```
Epoch 50/200
------------------------------------------------------------------------
Train loss: 0.9234 (action: 0.9234, kp: 0.0000)
Val loss: 1.0456

============================================================
ACTION METRICS
============================================================
Val macro F1: 0.5234
Val macro Acc: 0.6789
Val segment F1@0.3: 0.3456
Per-class F1:
  rest: 0.7834
  paw_withdraw: 0.4567
  paw_lick: 0.5123
  paw_shake: 0.4789
  walk: 0.3456
  active: 0.3567
============================================================
```

### Multi-Task Model

```
Epoch 50/200
------------------------------------------------------------------------
Train loss: 1.0234 (action: 0.9234, kp: 0.1000)
Val loss: 1.1456

============================================================
ACTION METRICS
============================================================
Val macro F1: 0.5234
Val macro Acc: 0.6789
Val segment F1@0.3: 0.3456
Per-class F1:
  rest: 0.7834
  paw_withdraw: 0.4567
  paw_lick: 0.5123
  paw_shake: 0.4789
  walk: 0.3456
  active: 0.3567

============================================================
KEYPOINT METRICS
============================================================
Val keypoint MAE (norm): 0.0345
============================================================
```

---

## Complete Training Commands

### Action-Only Model (Recommended Settings)

```bash
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --model_type action_only \
    --epochs 200 \
    --batch_size 2 \
    \
    # 345-frame support
    --min_frames 345 \
    \
    # Augmentation
    --kp_jitter_std 0.02 \
    --aug_brightness 0.2 \
    --aug_contrast 0.2 \
    --aug_temporal_drop 0.1 \
    \
    # Class balancing
    --rare_boost_cap 30.0 \
    --focal_gamma 2.5 \
    --label_smoothing 0.1 \
    \
    # Regularization
    --dropout 0.3 \
    --head_dropout 0.2 \
    --weight_decay 5e-2 \
    --lr 5e-5 \
    \
    # Training strategy
    --early_stopping_patience 20 \
    --lr_schedule cosine \
    --warmup_epochs 10
```

### Multi-Task Model (Recommended Settings)

```bash
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --model_type multitask \
    --epochs 200 \
    --batch_size 2 \
    \
    # 345-frame support
    --min_frames 345 \
    \
    # Augmentation
    --kp_jitter_std 0.02 \
    --aug_brightness 0.2 \
    --aug_contrast 0.2 \
    --aug_temporal_drop 0.1 \
    \
    # Class balancing
    --rare_boost_cap 30.0 \
    --focal_gamma 2.5 \
    --label_smoothing 0.1 \
    \
    # Loss weights
    --lambda_kp 1.0 \
    --lambda_smooth 5e-4 \
    \
    # Regularization
    --dropout 0.3 \
    --head_dropout 0.2 \
    --weight_decay 5e-2 \
    --lr 5e-5 \
    \
    # Training strategy
    --early_stopping_patience 20 \
    --lr_schedule cosine \
    --warmup_epochs 10
```

### Quick Test (2 epochs)

```bash
# Action-only
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --model_type action_only \
    --epochs 2 \
    --batch_size 2 \
    --train_T 60 \
    --val_T 60 \
    --early_stopping_patience 0

# Multi-task
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --model_type multitask \
    --epochs 2 \
    --batch_size 2 \
    --train_T 60 \
    --val_T 60 \
    --early_stopping_patience 0
```

---

## Model Checkpoints

Both model types save checkpoints with the following structure:

```python
checkpoint = {
    'model_state_dict': ...,
    'ema_state_dict': ...,  # if --use_ema
    'classes': ['rest', 'paw_withdraw', ...],
    'orig_classes': ['rest', 'paw_withdraw', ...],
    'col_map': {0: 0, 1: 1, 2: 2, 3: 1, ...},
    'merge_idxs': [1, 3, 5],
    'best_thresholds': [...],
    'kp_names': ['mouth', 'tail_base', ...],
    'val_metrics': {...},
    'args': {...}  # includes 'model_type'
}
```

**Load checkpoint:**
```python
checkpoint = torch.load('best_model_multitask.pt')
model_type = checkpoint['args']['model_type']

if model_type == 'action_only':
    model = ActionOnlyModel(
        num_classes=len(checkpoint['classes']),
        num_pose_features=18,
        ...
    )
else:
    model = DINOv3_Temporal_MultiTask_VideoMAE2Preferred(
        num_classes=len(checkpoint['classes']),
        num_keypoints=len(checkpoint['kp_names']),
        num_pose_features=18,
        ...
    )

model.load_state_dict(checkpoint['model_state_dict'])
```

---

## FAQ

### Q: Should I use action_only or multitask?

**Use action_only if:**
- You only care about action recognition
- Want faster training
- Have limited computational resources
- Want to focus the model on actions only

**Use multitask if:**
- You need both actions and keypoint predictions
- Want joint learning (keypoint task helps action recognition)
- Have enough computational resources
- Believe auxiliary task improves feature learning

### Q: Why use pose graph features instead of raw coordinates?

**Benefits:**
- Translation/scale/rotation invariant
- More informative geometric relationships
- Better generalization across different camera angles/positions
- Reduces dimensionality (18 features vs 12 raw coords)

### Q: Does jittering affect pose graph features?

**Yes!** Jittering is applied **before** computing pose graph features:
1. Load keypoints
2. Apply jittering (add noise)
3. Compute pose graph from jittered keypoints
4. Pass to model

This makes the model robust to slight keypoint localization errors.

### Q: Can I disable pose graph features?

**No**, the current implementation requires pose graph features. If you want to use raw coordinates, you would need to modify the model architectures (not recommended).

### Q: How much slower is multitask vs action_only?

**Minimal difference:**
- Action-only: Trains slightly faster (no keypoint loss computation)
- Multi-task: Adds keypoint loss (very fast operation)
- Difference: ~5-10% slower per epoch

The main benefit of action_only is **conceptual focus**, not speed.

---

## Troubleshooting

### Issue: Pose graph features are all NaN

**Cause**: Missing or invalid keypoints (all visibility = 0)

**Solution**: Check your DLC CSV files have valid keypoint predictions:
```bash
python -c "
import pandas as pd
df = pd.read_csv('your_video_DLC_*.csv', header=[0,1,2])
print(df.describe())
"
```

### Issue: Action metrics are good but keypoint MAE is high

**Normal behavior!** The keypoint head predicts raw coordinates, which is a harder task than action classification. The main goal is action recognition; keypoints are auxiliary.

**Solutions:**
- Increase `--lambda_kp` to weight keypoint loss more
- Use action_only model if you don't need keypoint predictions

### Issue: Model overfits despite regularization

**Solutions:**
- Increase `--dropout` (try 0.4-0.5)
- Increase `--weight_decay` (try 1e-1)
- Reduce learning rate `--lr` (try 1e-5)
- Use stronger augmentation `--aug_brightness 0.3 --aug_contrast 0.3`

---

## Summary

✅ **Pose graph features**: 18 geometric features (edge lengths + angles) instead of raw coords
✅ **Action-only model**: Predicts actions only using visual + pose features
✅ **Multi-task model**: Predicts actions + keypoints jointly
✅ **Keypoint jittering**: Augments keypoints before pose graph computation
✅ **Separate metrics**: Action metrics and keypoint metrics reported independently

**Next steps:**
1. Choose model type based on your needs
2. Run training with recommended settings
3. Monitor action metrics (macro F1, per-class F1)
4. Adjust hyperparameters if needed (see MULTITASK_IMPROVEMENTS.md)
