# Weights & Biases (W&B) Logging - Complete Guide

## Summary

Added comprehensive W&B experiment tracking to the training script with automatic run naming based on encoder model and key hyperparameters.

---

## Installation

```bash
pip install wandb
```

**First-time setup:**
```bash
wandb login
```

This will prompt you to enter your W&B API key (get it from https://wandb.ai/authorize).

---

## Basic Usage

### Enable W&B Logging

```bash
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --model_type action_only \
    --train_T 345 \
    --val_T 345 \
    --batch_size 2 \
    --epochs 50 \
    --use_wandb
```

**Default settings:**
- Entity: `grassknoted`
- Project: `prj_mouse_pain`
- Run name: Auto-generated (see below)

### Custom Entity/Project

```bash
python train_multitask.py \
    ... \
    --use_wandb \
    --wandb_entity your_entity \
    --wandb_project your_project
```

### Custom Run Name

```bash
python train_multitask.py \
    ... \
    --use_wandb \
    --wandb_run_name "my_custom_experiment_v1"
```

---

## Auto-Generated Run Names

If you don't specify `--wandb_run_name`, the script will automatically generate a descriptive name:

### Format

```
{encoder_name}_{model_type}_lr{lr}_bs{batch_size}_drop{dropout}_T{train_T}
```

### Examples

**VideoMAE + Action-Only:**
```
videomae_action_only_lr5e-05_bs2_drop0.3_T345
```

**DINOv2 + Multi-Task:**
```
dinov2_multitask_lr5e-05_bs2_drop0.3_T345
```

**2D ViT (AugReg) + Action-Only:**
```
vit_augreg_action_only_lr5e-05_bs2_drop0.3_T180
```

### Encoder Names Detected

The script automatically detects which encoder is being used:

| Encoder Used | Name in Run |
|-------------|-------------|
| VideoMAE (3D) | `videomae` |
| DINOv2 (2D) | `dinov2` |
| ViT-AugReg (2D) | `vit_augreg` |
| Generic 2D ViT | `vit2d` |
| Fallback | `unknown` |

**Note:** The run name is updated **after model creation** to reflect the actual encoder used (not just what was attempted).

---

## Logged Metrics

### Every Epoch

**Training Metrics:**
- `train/loss` - Total training loss
- `train/action_loss` - Action classification loss
- `train/kp_loss` - Keypoint prediction loss (multitask only, 0.0 for action_only)

**Validation Metrics:**
- `val/loss` - Total validation loss
- `val/macro_f1` - Macro-averaged F1 score (main metric)
- `val/macro_acc` - Macro-averaged accuracy
- `val/segment_f1_0.3` - Segment-level F1 at 30% IoU threshold

**Per-Class F1 Scores:**
- `val/f1_rest`
- `val/f1_paw_withdraw`
- `val/f1_paw_lick`
- `val/f1_paw_shake`
- `val/f1_walk`
- `val/f1_active`

**Keypoint Metrics (multitask only):**
- `val/kp_mae_norm` - Normalized mean absolute error for keypoint predictions

**Optimization:**
- `lr` - Current learning rate

---

## Configuration Logged

The script automatically logs all hyperparameters via `config=vars(args)`:

**Model Architecture:**
- `model_type` (action_only / multitask)
- `dropout`, `head_dropout`
- `hidden_dim`, `num_layers`

**Training:**
- `lr`, `batch_size`, `epochs`
- `weight_decay`, `grad_clip_norm`
- `lr_schedule`, `warmup_epochs`

**Data:**
- `train_T`, `val_T`, `min_frames`
- `kp_likelihood_thr`

**Augmentation:**
- `kp_jitter_std`
- `aug_brightness`, `aug_contrast`
- `aug_temporal_drop`, `aug_hflip`

**Class Balancing:**
- `rare_boost_cap`, `rare_threshold_share`
- `focal_gamma`, `label_smoothing`

**Loss Weights:**
- `lambda_kp`, `lambda_smooth`

And many more... (see W&B dashboard for full config)

---

## W&B Dashboard

### Viewing Your Runs

After enabling W&B, you can view your runs at:

```
https://wandb.ai/grassknoted/prj_mouse_pain
```

(Replace with your entity/project if different)

### Key Visualizations

**1. Training Curves:**
- Compare `train/loss` vs `val/loss` to check overfitting
- Monitor `val/macro_f1` (main metric)

**2. Per-Class Performance:**
- Compare `val/f1_*` across classes to identify which actions are hardest
- Track rare class improvement (e.g., `val/f1_paw_shake`)

**3. Learning Rate Schedule:**
- Verify `lr` follows expected schedule (cosine/step/plateau)

**4. Comparing Runs:**
- Compare different encoders (videomae vs dinov2 vs vit_augreg)
- Compare model types (action_only vs multitask)
- Compare hyperparameters (learning rate, dropout, etc.)

---

## Example Training Commands

### Action-Only with W&B

```bash
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --model_type action_only \
    --train_T 345 \
    --val_T 345 \
    --batch_size 2 \
    --epochs 50 \
    --lr 5e-5 \
    --dropout 0.3 \
    --use_wandb
```

**Expected run name:**
```
videomae_action_only_lr5e-05_bs2_drop0.3_T345
```

### Multi-Task with W&B

```bash
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --model_type multitask \
    --train_T 345 \
    --val_T 345 \
    --batch_size 2 \
    --epochs 50 \
    --lr 5e-5 \
    --dropout 0.3 \
    --lambda_kp 1.0 \
    --use_wandb
```

**Expected run name:**
```
videomae_multitask_lr5e-05_bs2_drop0.3_T345
```

### Custom Run Name

```bash
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --model_type action_only \
    --train_T 345 \
    --val_T 345 \
    --batch_size 2 \
    --epochs 50 \
    --use_wandb \
    --wandb_run_name "experiment_baseline_v1"
```

---

## Comparing Multiple Runs

### Run a Hyperparameter Sweep Manually

```bash
# Baseline
python train_multitask.py ... --use_wandb --lr 5e-5 --dropout 0.3

# Higher dropout
python train_multitask.py ... --use_wandb --lr 5e-5 --dropout 0.4

# Lower learning rate
python train_multitask.py ... --use_wandb --lr 1e-5 --dropout 0.3

# Different encoder (force 2D by setting CUDA_VISIBLE_DEVICES="")
CUDA_VISIBLE_DEVICES="" python train_multitask.py ... --use_wandb --lr 5e-5
```

All runs will appear in your W&B project with auto-generated names that make them easy to identify and compare.

---

## Tags

Each run is automatically tagged with:
- `action_only` or `multitask` (based on `--model_type`)
- `pose_graph` (since all runs use pose graph features)

### Custom Tags

To add custom tags, modify the `wandb.init()` call in `train_multitask.py:2086`:

```python
wandb.init(
    entity=args.wandb_entity,
    project=args.wandb_project,
    name=run_name,
    config=vars(args),
    tags=[args.model_type, "pose_graph", "my_custom_tag"]
)
```

---

## Troubleshooting

### Issue: "wandb not installed"

**Solution:**
```bash
pip install wandb
```

### Issue: "wandb login required"

**Solution:**
```bash
wandb login
```

Then enter your API key from https://wandb.ai/authorize

### Issue: W&B slow or timing out

**Cause:** Network issues or W&B servers down

**Solution 1:** Use offline mode (logs saved locally, uploaded later):
```bash
wandb offline
python train_multitask.py ... --use_wandb
wandb sync  # Upload logs when network is back
```

**Solution 2:** Disable W&B for this run:
```bash
python train_multitask.py ...  # No --use_wandb flag
```

### Issue: Run name not updating with correct encoder

**Check:** Make sure the model is created successfully. The run name updates after model creation.

**If VideoMAE fails to load:**
- Run name will show encoder that was actually used (e.g., `dinov2` instead of `videomae`)

### Issue: Want to change entity/project mid-run

**Not recommended!** Each run is tied to a specific entity/project at initialization.

**Solution:** Stop the current run and start a new one with `--wandb_entity` and `--wandb_project` flags.

---

## Fallback Behavior

### W&B Not Installed

```
[warn] wandb not installed. Logging to W&B will not be available.
[warn] Install with: pip install wandb
```

Training will continue normally without W&B logging.

### W&B Flag Specified but Not Installed

```
[warn] --use_wandb specified but wandb not installed. Skipping W&B logging.
[warn] Install with: pip install wandb
```

Training will continue normally without W&B logging.

### W&B Initialization Fails

If `wandb.init()` fails (e.g., network issue), the training will continue but logging will be skipped for that run.

---

## Best Practices

### 1. Always Use W&B for Experiments

Unless doing quick debugging, always add `--use_wandb` to track experiments.

### 2. Use Auto-Generated Run Names

Auto-generated names include key hyperparameters, making runs easy to identify:
```bash
--use_wandb  # Let script auto-generate name
```

### 3. Compare Encoders

Run experiments with different encoders to find the best one:
```bash
# VideoMAE (3D)
python train_multitask.py ... --use_wandb

# Force 2D (by limiting GPU memory or setting transformers unavailable)
# (2D models will be tried automatically if VideoMAE fails)
```

### 4. Group Related Runs

Use consistent hyperparameters for fair comparison:
```bash
# All runs use same settings except the variable you're testing
--train_T 345 --val_T 345 --batch_size 2 --epochs 50
```

### 5. Add Notes to Runs

In the W&B dashboard, add notes to runs explaining:
- What you were testing
- Any issues encountered
- Observations about performance

---

## Summary

✅ **Automatic W&B logging** with `--use_wandb` flag
✅ **Auto-generated run names** including encoder and key hyperparameters
✅ **Comprehensive metrics** logged every epoch (train/val losses, F1 scores, per-class metrics)
✅ **Full hyperparameter tracking** via config logging
✅ **Graceful fallback** if W&B not installed
✅ **Entity/Project defaults** set to `grassknoted/prj_mouse_pain`

**Quick Start:**
```bash
pip install wandb
wandb login
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --model_type action_only \
    --train_T 345 \
    --val_T 345 \
    --batch_size 2 \
    --epochs 50 \
    --use_wandb
```

View your runs at: https://wandb.ai/grassknoted/prj_mouse_pain
