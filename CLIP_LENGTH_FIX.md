# Clip Length Fix

## Issue
The data loaders require `clip_length` to be an **odd number** for proper centering. The original code used `clip_length=16` (even number), which caused an `AssertionError`.

## Why Odd Numbers?

The clip extraction works by taking frames **centered** around a target frame:
- Center frame: frame index `i`
- Frames before: `i - clip_length//2` to `i - 1`
- Center frame: `i`
- Frames after: `i + 1` to `i + clip_length//2`

With an **odd** clip_length, you get equal frames on both sides:
- `clip_length=17`: 8 frames before + 1 center + 8 frames after = 17 total ✓
- `clip_length=16`: Would be unbalanced (can't center properly) ✗

## Changes Made

All default `clip_length` values changed from **16 → 17**:

### Files Updated:
1. `train.py` - Default parameter and example usage
2. `multimodal_train.py` - Default parameter and example usage
3. `model.py` - Default parameter in model creation functions
4. `data_loader.py` - Improved error message
5. `multimodal_data_loader.py` - Improved error message

## Usage

### Training Scripts
Both training scripts now default to `clip_length=17`:

```bash
python train.py  # Uses clip_length=17 by default
python multimodal_train.py  # Uses clip_length=17 by default
```

### Custom Clip Length
You can use any **odd** number:

```python
from train import train_model

# Valid clip lengths: 9, 11, 13, 15, 17, 19, 21, 23, 25, etc.
train_model(
    video_dir="./Videos",
    annotation_dir="./Annotations",
    clip_length=15,  # ✓ Odd number
    # clip_length=16,  # ✗ Would raise AssertionError
)
```

### Recommended Values
- **Small/Fast**: `clip_length=9` (0.3 seconds @ 30 FPS)
- **Standard**: `clip_length=17` (0.57 seconds @ 30 FPS) ← **Default**
- **Long context**: `clip_length=25` (0.83 seconds @ 30 FPS)
- **Very long**: `clip_length=31` (1.03 seconds @ 30 FPS)

**Trade-offs:**
- Smaller clip_length: Faster training, less temporal context
- Larger clip_length: Slower training, more temporal context, higher memory usage

## Temporal Context

With 30 FPS video:
- `clip_length=17` = 17 frames / 30 fps ≈ **0.57 seconds** of context
- This captures short-term movements and transitions
- Good balance between context and computational efficiency

## Impact on Model

The 3D CNN model automatically adapts to any clip_length:
- Temporal convolutions preserve temporal dimension initially
- MaxPooling3D with stride (2,2,2) reduces temporal dimension
- GlobalAvgPool3D at the end pools over all temporal frames

**Model still works with any clip_length**, just needs to be odd!

## Error Message

If you accidentally use an even number, you'll see:
```
AssertionError: clip_length must be odd for centering (got 16). Use 15, 17, 19, etc.
```

Simply change to the nearest odd number (15 or 17).
