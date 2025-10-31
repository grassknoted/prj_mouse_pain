# Changelog - Technical Fixes & Updates

This document consolidates all technical fixes, improvements, and updates made to the mouse pain recognition codebase.

---

## Table of Contents

1. [Multi-Task Training Fixes](#multi-task-training-fixes)
2. [Data Handling Improvements](#data-handling-improvements)
3. [Model Architecture Updates](#model-architecture-updates)
4. [Training Infrastructure](#training-infrastructure)
5. [Monitoring & Logging](#monitoring--logging)
6. [Bug Fixes](#bug-fixes)

---

## Multi-Task Training Fixes

### 345-Frame Annotation Support (CRITICAL)

**Problem**: Annotations had only 345 frames (not 360) due to annotation model using 15-frame context window, causing loss of ~20-30% of training data.

**Solution**:
- Accept `--min_frames 345` (new default)
- Auto-detect annotation length (345 or 360)
- For 345-frame: Use video frames `[15:360]` (last 345)
- For 360-frame: Use video frames `[0:360]` (all frames)
- Apply same offset to DLC keypoints for perfect alignment

**Impact**: ~20-30% more training data, better generalization

**Files Modified**:
- `train_multitask.py`: Added frame offset logic

---

### Multi-Task Training Complete Overhaul

**Problems Addressed**:
1. ✅ Severe overfitting (train loss 0.33 vs val loss 2.87)
2. ✅ Poor minority class F1:
   - paw_withdraw: 0.13 → Target 0.40+
   - walk: 0.08 → Target 0.30+
   - active: 0.19 → Target 0.30+

**Improvements**:

#### 1. Keypoint Coordinate Jittering
- Adds Gaussian noise to keypoints during training: `kp + N(0, 0.02)`
- Improves robustness to pose variations
- Controlled by `--kp_jitter_std` (default 0.02)

#### 2. Comprehensive Data Augmentation
ALL training samples now get augmentation (not just rare classes):

| Augmentation | Probability | Default | Argument |
|--------------|-------------|---------|----------|
| Brightness jitter | 50% | ±20% | `--aug_brightness 0.2` |
| Contrast jitter | 50% | ±20% | `--aug_contrast 0.2` |
| Temporal dropout | 30% | 10% frames | `--aug_temporal_drop 0.1` |
| Gaussian noise | 30% | 0.02 std | Fixed |
| Temporal roll | 20% | ±5 frames | Fixed |

#### 3. Stronger Class Balancing
- `--rare_threshold_share`: 0.03 → 0.05 (classify more as rare)
- `--rare_boost_cap`: 12 → 30 (allow higher replication)
- `--focal_gamma`: 1.5 → 2.5 (stronger focus on hard examples)

#### 4. Label Smoothing
- Added `--label_smoothing 0.1`
- Prevents overconfidence, improves generalization

#### 5. Increased Regularization
- Dropout: 0.1 → 0.3
- Weight decay: 1e-2 → 5e-2

#### 6. Early Stopping
- Added `--early_stop_patience 20`
- Prevents overfitting from excessive training

#### 7. ReduceLROnPlateau Scheduler
- Option to use plateau-based LR reduction
- `--scheduler plateau`

#### 8. Lower Learning Rate
- Default LR: 1e-4 → 5e-5
- More stable training

**Files Modified**:
- `train_multitask.py`: Complete rewrite of augmentation, balancing, training loop

**References**:
- Original fixes in: `MULTITASK_IMPROVEMENTS.md`

---

### CSV Pattern & Data Format Fixes

**Problems Fixed**:
1. ✅ CSV pattern: Changed `predictions_` → `prediction_`
2. ✅ Action indices: Use numeric 0-7 instead of string names
3. ✅ Model input size: Pass `img_size` with `dynamic_img_size=True`
4. ✅ Keypoint names: Updated to match DLC format
5. ✅ DLC CSV discovery: Search for files starting with video stem
6. ✅ DLC parsing: Robust multi-level header parsing with error handling

**Data Format Established**:

**Action CSVs**:
```
Pattern: prediction_<video_name>_<trial_number>.csv
Format: Frame,Action (where Action is 0-7)
```

**DLC CSVs**:
```
Pattern: <video_name>DLC_resnet50_pawtracking_comparisionMay7shuffle1_500000.csv
Format: Multi-level header (scorer, bodyparts, coords)
Keypoints: mouth, L_frontpaw, R_frontpaw, L_hindpaw, R_hindpaw, tail_base
```

**Files Modified**:
- `train_multitask.py`: CSV loading, pattern matching, DLC parsing

**References**:
- Original fixes in: `FIXES_SUMMARY.md`, `MULTITASK_TRAINING_FIXED.md`

---

## Data Handling Improvements

### Variable Trial Length Support

**Problem**: Videos have multiple trials with different lengths, causing data loading errors.

**Solution**:
- Support variable-length trials within same video
- Auto-detect trial boundaries from action CSVs
- Handle trials < minimum length gracefully
- Skip invalid trials with warning

**Impact**: Can process real-world messy datasets

**Files Modified**:
- `train_multitask.py`: Trial discovery and validation
- `multimodal_data_loader.py`: Added trial-aware loading

**References**:
- Original: `VARIABLE_TRIAL_LENGTHS.md`, `MULTI_TRIAL_UPDATE.md`

---

### Skipping Invalid Trials

**Problem**: Some trials have corrupted data, missing frames, or DLC errors.

**Solution**:
- Validate each trial during discovery:
  - Check video frame count
  - Check annotation row count
  - Check DLC availability and format
- Skip invalid trials with detailed warning
- Continue training with valid trials only

**Example**:
```
[warn] Trial 'video_123_trial_5' skipped: Video has 320 frames < min 345
[warn] Trial 'video_456_trial_2' skipped: No matching DLC file found
[info] Valid trials: 89/95 (6 skipped)
```

**Files Modified**:
- `train_multitask.py`: Validation logic
- Test scripts: Added validation utilities

**References**:
- Original: `SKIPPING_INVALID_TRIALS.md`

---

### Data Statistics & Normalization

**Problem**: Inconsistent understanding of class distributions and normalization.

**Solution**:
- Display detailed class distribution before and after boosting
- Show per-class sample counts and percentages
- Print rare class boost factors
- Normalize keypoints to [0, 1] range
- Z-score normalization for pose graph features

**Example Output**:
```
============================================================
Class Distribution BEFORE boosting
============================================================
Class                     Count      Share
------------------------------------------------------------
rest                     45678     54.32%
paw_withdraw               850      1.01%  ← RARE
paw_lick                  1234      1.47%  ← RARE
...
============================================================
After Boosting
============================================================
paw_withdraw: 850 → 21250 (boost 25.0x)
paw_lick: 1234 → 22212 (boost 18.0x)
```

**Files Modified**:
- `train_multitask.py`: Statistics printing
- `data_loader.py`: Normalization

**References**:
- Original: `DATA_STATISTICS_EXPLANATION.md`

---

## Model Architecture Updates

### Pose Graph Features & Model Types

**Enhancement**: Two model types with geometric pose features

**Model Types**:

1. **Action-Only** (`--model_type action_only`):
   - Predicts actions using visual + pose graph features
   - No keypoint prediction head
   - Faster training

2. **Multi-Task** (`--model_type multitask`):
   - Predicts actions using visual + pose graph features
   - Also predicts keypoint coordinates (auxiliary task)
   - Better feature learning through multi-task

**Pose Graph (18 features)**:
- 8 edge lengths: Normalized distances between keypoint pairs
- 10 angles: Joint angles from keypoint triplets
- Translation/scale/rotation invariant
- Computed from 6 keypoints

**Keypoint Order**: `[mouth, tail_base, L_frontpaw, R_frontpaw, L_hindpaw, R_hindpaw]`

**Files Modified**:
- `train_multitask.py`: PoseGraph class, model heads
- `multimodal_model.py`: Pose stream architecture

**References**:
- Original: `POSE_GRAPH_AND_MODEL_TYPES.md`

---

### VideoMAE Hugging Face Update

**Problem**: VideoMAE models moved from timm to Hugging Face transformers.

**Solution**:
- Use `transformers.VideoMAEModel` for VideoMAE2 3D backbone
- Fallback to 2D ViT with temporal mean pooling if transformers unavailable
- Graceful degradation with warnings

**Code**:
```python
try:
    from transformers import VideoMAEModel
    # Use VideoMAE2 for 3D temporal modeling
except ImportError:
    print("[warn] transformers not installed, falling back to 2D")
    VideoMAEModel = None
```

**Files Modified**:
- `train_multitask.py`: Import and backbone selection logic

**References**:
- Original: `VIDEOMAE_HUGGINGFACE_UPDATE.md`

---

### Clip Length Optimization

**Problem**: Fixed 16-frame clips may not capture full behavior context.

**Solution**:
- Make clip length configurable: `--clip_T` (default 16)
- Support variable clip lengths: 8, 16, 24, 32 frames
- Adjust temporal pooling based on clip length
- For 2D backbones: Use mean pooling across frames

**Recommendations**:
- **16 frames**: Standard (0.53s @ 30fps) - good balance
- **24-32 frames**: Longer behaviors (walk, active)
- **8 frames**: Quick responses (paw_withdraw)

**Files Modified**:
- `data_loader.py`: Clip extraction logic
- `train.py`, `multimodal_train.py`: Clip length parameter

**References**:
- Original: `CLIP_LENGTH_FIX.md`

---

## Training Infrastructure

### Multi-GPU Training Support

**Enhancement**: Distributed training for faster experiments.

**Approaches**:

#### 1. Video-Only (train.py)
```bash
bash train_multigpu.sh
```

Uses `torch.distributed` with `DistributedDataParallel`:
- Automatic process spawning
- Gradient synchronization
- Efficient data parallelism

#### 2. Multimodal (multimodal_train.py)
```bash
bash train_multimodal_multigpu.sh
```

Same as video-only but with dual-stream model.

#### 3. Multi-Task (train_multitask.py)
- **Automatic**: Detects multiple GPUs and uses them
- No script needed

**Files Added**:
- `train_multigpu.sh`: Video-only multi-GPU script
- `train_multimodal_multigpu.sh`: Multimodal multi-GPU script

**Files Modified**:
- `train.py`: Added DDP support
- `multimodal_train.py`: Added DDP support

**References**:
- Original: `MULTI_GPU_TRAINING.md`

---

### Gradient Tracking Fix

**Problem**: Gradients were being computed for validation samples, wasting memory.

**Solution**:
- Wrap validation loop with `torch.no_grad()`
- Wrap evaluation code with `torch.no_grad()`
- Disable gradient computation for inference

**Code**:
```python
# Before (WRONG)
for batch in val_loader:
    outputs = model(batch)
    loss = criterion(outputs, labels)

# After (CORRECT)
with torch.no_grad():
    for batch in val_loader:
        outputs = model(batch)
        loss = criterion(outputs, labels)
```

**Impact**: Reduced validation memory usage by ~30-40%

**Files Modified**:
- `train.py`: Validation loop
- `multimodal_train.py`: Validation loop
- `train_multitask.py`: Validation loop
- `evaluation.py`: Evaluation functions

**References**:
- Original: `GRADIENT_TRACKING_FIX.md`

---

### Progress Bars with tqdm

**Enhancement**: Added informative progress bars for all loops.

**Features**:
- Training batches: Show loss, accuracy, ETA
- Validation batches: Show loss, F1, ETA
- Discovery phase: Show video/trial discovery progress
- Epoch progress: Overall training progress

**Example**:
```
Discovering videos: 100%|████████████| 10/10 [00:02<00:00]
Epoch 5/50: 100%|████████████| 50/50 [02:15<00:00, 2.71s/it]
Training: 100%|████████████| 240/240 [02:10<00:00] Loss: 1.234
Validation: 100%|████████████| 12/12 [00:18<00:00] F1: 0.856
```

**Files Modified**:
- All training scripts: Added tqdm wrappers

**References**:
- Original: `TQDM_PROGRESS_BARS.md`

---

## Monitoring & Logging

### Weights & Biases (W&B) Integration

**Enhancement**: Professional experiment tracking.

**Features**:
- Auto-generated run names based on hyperparameters
- Comprehensive metric logging
- Hyperparameter tracking
- Model checkpoints (optional)
- Comparison dashboard

**Run Name Format**:
```
<encoder>_<config>
Example: vit_small_b2_lr5e-5_g2.5_ls0.1_wd5e-2
```

**Usage**:
```bash
# Enable W&B
python train_multitask.py --use_wandb ...

# Custom entity/project
python train_multitask.py \
    --use_wandb \
    --wandb_entity your_entity \
    --wandb_project your_project \
    ...
```

**Logged Metrics**:
- Training: loss, action_loss, kp_loss (if multitask)
- Validation: loss, macro_f1, macro_acc, segment_f1, kp_mae, per_class_f1
- Hyperparameters: All args
- System: Learning rate, epoch

**Files Modified**:
- `train_multitask.py`: W&B initialization, logging, run naming

**References**:
- Original: `WANDB_LOGGING.md`

---

## Bug Fixes

### Torch uint8 Fix

**Problem**: Some videos load as uint8 tensors, causing errors in models expecting float32.

**Solution**:
```python
# Before
frames = torch.tensor(frames)  # Might be uint8

# After
frames = torch.tensor(frames, dtype=torch.float32) / 255.0  # Normalize to [0,1]
```

**Files Modified**:
- `data_loader.py`: Frame loading
- `multimodal_data_loader.py`: Frame loading
- `train_multitask.py`: Video reading

**References**:
- Original: `TORCH_UINT8_FIX.md`

---

### DLC CSV Parsing Robustness

**Problem**: DLC CSVs have complex multi-level headers that often fail to parse.

**Solution**:
- Robust header detection (2-row or 3-row)
- Fallback parsing strategies
- Detailed error messages
- Skip malformed files with warnings

**Example Error Handling**:
```python
try:
    df = pd.read_csv(dlc_path, header=[0, 1, 2])
except:
    try:
        df = pd.read_csv(dlc_path, header=[0, 1])
    except:
        logger.warning(f"Failed to parse DLC file: {dlc_path}")
        return None
```

**Files Modified**:
- `train_multitask.py`: DLC loading
- `multimodal_data_loader.py`: DLC loading
- `debug_dlc_csv.py`: DLC inspection tool

---

### Missing DLC File Handling

**Problem**: Training crashes if DLC file missing for a trial.

**Solution**:
- Check DLC file existence during discovery
- Skip trials with missing DLC files
- Continue training with available trials
- Detailed warnings

**Example**:
```
[warn] No DLC file found for 'video_123.mp4', skipping trial
[info] 89/95 trials have valid DLC files
```

**Files Modified**:
- `train_multitask.py`: DLC discovery and validation
- `multimodal_data_loader.py`: DLC loading with error handling

---

## Version History

### v3.0 - Multi-Task Training (Oct 2024)
- Complete multi-task training pipeline (`train_multitask.py`)
- VideoMAE2/ViT backbones
- TCN temporal head
- Pose graph features
- W&B logging
- 345-frame support
- Comprehensive augmentation

### v2.0 - Multimodal Training (Sep 2024)
- Dual-stream architecture (visual + pose)
- DeepLabCut integration
- Temporal attention for pose
- Fusion layer
- Improved pain detection

### v1.0 - Video-Only Training (Aug 2024)
- 3D CNN baseline
- Basic training pipeline
- TensorBoard logging
- Class weighting

---

## Migration Guide

### From Video-Only to Multimodal

No code changes needed. Just provide DLC files:

```bash
# Before (video-only)
python train.py

# After (multimodal)
python multimodal_train.py  # Will auto-find DLC files in DLC/
```

### From Multimodal to Multi-Task

Update data organization and use new script:

```bash
# 1. Move DLC files from DLC/ to Videos/
mv DLC/*.csv Videos/

# 2. Rename annotation files to prediction_<video>_<trial>.csv pattern
# (see data format in MULTITASK_GUIDE.md)

# 3. Run multi-task training
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --model_type multitask
```

---

## Known Issues

### 1. VideoMAE2 Memory Usage
**Issue**: VideoMAE2 3D backbone requires significant GPU memory.

**Workaround**:
- Use 2D ViT backbone (default fallback)
- Reduce batch size
- Use lighter ViT variant (`vit_tiny_*`)

### 2. DLC Tracking Errors
**Issue**: DeepLabCut occasionally has low-confidence or missing keypoints.

**Workaround**:
- Pose graph computation handles missing keypoints (returns 0 features)
- Consider filtering low-confidence detections before training
- Use keypoint jittering to improve robustness

### 3. Class Imbalance
**Issue**: Pain responses are very rare (~1% of frames).

**Solutions Implemented**:
- Rare-class boosting (up to 30x replication)
- Focal loss (gamma=2.5)
- Class-weighted loss
- Data augmentation

**Still struggling?**
- Increase `--rare_boost_cap` to 50
- Increase `--focal_gamma` to 3.0
- Use per-class thresholds (computed automatically)

---

## Testing & Validation

### Validation Scripts

```bash
# Test video-only data loading
python test_pipeline.py

# Test data loaders
python test_updated_loaders.py

# Test multi-task data loading
python test_multitask_data.py

# Validate data structure
python validate_data_structure.py

# Simple validation
python validate_simple.py

# Debug DLC CSV
python debug_dlc_csv.py path/to/dlc.csv
```

### Common Validation Checks

1. **Frame count**: Videos have ≥345 frames
2. **Annotation rows**: CSVs have ≥345 rows
3. **DLC files**: Present and parseable
4. **Keypoint names**: Match expected names
5. **Action indices**: In range [0-7]

---

## Future Improvements

### Potential Enhancements

1. **Real-time inference**: Optimize for live video streams
2. **Semi-supervised learning**: Use unlabeled videos
3. **Active learning**: Intelligently select samples for annotation
4. **Ensemble models**: Combine multiple model predictions
5. **Attention visualization**: Show which frames/regions model focuses on
6. **Cross-validation**: K-fold instead of single train/val split
7. **Temporal post-processing**: Smooth predictions over time
8. **Confidence calibration**: Better uncertainty estimates

### Requested Features

- Support for more keypoints (currently limited to 6)
- RGB video support (currently grayscale)
- Online data augmentation (currently offline)
- Model distillation (compress to smaller model)
- ONNX export for deployment

---

## References

All fixes documented in separate markdown files (now archived):
- `MULTITASK_IMPROVEMENTS.md`
- `MULTITASK_TRAINING_FIXED.md`
- `FIXES_SUMMARY.md`
- `WANDB_LOGGING.md`
- `MULTI_TRIAL_UPDATE.md`
- `VARIABLE_TRIAL_LENGTHS.md`
- `SKIPPING_INVALID_TRIALS.md`
- `DATA_STATISTICS_EXPLANATION.md`
- `POSE_GRAPH_AND_MODEL_TYPES.md`
- `VIDEOMAE_HUGGINGFACE_UPDATE.md`
- `MULTI_GPU_TRAINING.md`
- `GRADIENT_TRACKING_FIX.md`
- `TQDM_PROGRESS_BARS.md`
- `TORCH_UINT8_FIX.md`
- `CLIP_LENGTH_FIX.md`

---

**Last Updated**: October 2024

For current usage, see [README.md](README.md) and relevant guides ([QUICKSTART.md](QUICKSTART.md), [MULTIMODAL_GUIDE.md](MULTIMODAL_GUIDE.md), [MULTITASK_GUIDE.md](MULTITASK_GUIDE.md)).
