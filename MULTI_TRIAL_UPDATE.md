# Multi-Trial Data Structure Update

## Summary

Both data loaders (`data_loader.py` and `multimodal_data_loader.py`) have been updated to handle the multi-trial structure where:
- **One video contains multiple trials**
- **Each trial is 360 frames (12 seconds @ 30 FPS)**
- **Video length can be any multiple of 360** (e.g., 1080 for 3 trials, 1800 for 5 trials)

## Data Structure

### Your Current Data
```
Videos/
├── shortened_2023-10-25_CFA_010_267M_tracking.mp4  (1080 frames = 3 trials)
└── shortened_2023-10-25_CFA_010_267M_trackingDLC_resnet50_pawtracking_comparisionMay7shuffle1_500000.csv

Annotations/
├── prediction_shortened_2023-10-25_CFA_010_267M_tracking.mp4_1.csv  (360 frames, trial 1)
├── prediction_shortened_2023-10-25_CFA_010_267M_tracking.mp4_2.csv  (360 frames, trial 2)
└── prediction_shortened_2023-10-25_CFA_010_267M_tracking.mp4_3.csv  (360 frames, trial 3)
```

### Trial-to-Frame Mapping
- **Trial 1**: Video frames 0-359 (offset = 0)
- **Trial 2**: Video frames 360-719 (offset = 360)
- **Trial 3**: Video frames 720-1079 (offset = 720)
- **Trial N**: Video frames (N-1)×360 to N×360-1 (offset = (N-1)×360)

## Key Changes Made

### 1. Data Loader (`data_loader.py`)

#### Changes to `_preprocess_data()`:
- Now treats each annotation file as a separate trial
- Extracts trial number from filename: `prediction_<video_name>_<trial_number>.csv`
- Calculates frame offset: `frame_offset = (trial_number - 1) * 360`
- Validates that video has enough frames for each trial
- Each trial becomes an independent entry in `video_data` list

#### Changes to `__getitem__()`:
- Converts trial-relative frame indices to absolute video frame indices
- Formula: `absolute_frame = trial_info['frame_offset'] + relative_frame`
- Loads video frames from the correct position

#### Changes to `create_data_loaders()`:
- Finds all annotation files matching pattern `prediction_*.csv`
- Parses video name from annotation filename (handles `.mp4` already in filename)
- Groups trials by video
- **Splits by trial** (not by video) for train/val sets
  - This means trials from the same video can be in different splits
  - If you prefer keeping all trials from same video together, this can be modified

### 2. Multimodal Data Loader (`multimodal_data_loader.py`)

#### Changes to `_preprocess_data()`:
- Same trial extraction logic as standard loader
- Loads **full video DLC data** once per unique video
- **Slices DLC coordinates per trial**: `dlc_coords[frame_offset:frame_offset+360]`
- Computes pose features only for trial-specific DLC data
- Validates DLC has enough frames for each trial

#### Changes to `__getitem__()`:
- Converts trial-relative indices to absolute for video loading
- Pose features are already trial-specific (sliced during preprocessing)
- Uses trial-relative indices for pose features

#### Changes to `create_multimodal_data_loaders()`:
- Same annotation parsing as standard loader
- Finds DLC files with smart pattern matching:
  - Searches for `<video_stem>*DLC*.csv`
  - Falls back to checking video_dir if not in dlc_dir
  - **One DLC file per video** (shared across all trials)
- Caches video-to-DLC mapping to avoid redundant lookups

## File Naming Conventions

### Annotation Files
**Format**: `prediction_<video_name>_<trial_number>.csv`

**Example**:
```
prediction_shortened_2023-10-25_CFA_010_267M_tracking.mp4_1.csv
                                                           ^
                                                      trial number
```

**Note**: The video name can include `.mp4` extension - the loader handles this automatically.

### DLC Files
**Format**: `<video_stem>DLC_<model_info>.csv`

**Example**:
```
shortened_2023-10-25_CFA_010_267M_trackingDLC_resnet50_pawtracking_comparisionMay7shuffle1_500000.csv
```

**Location**: Can be in either `./Videos/` or a separate `./DLC/` directory

## Usage

### Training with Visual-Only Model

```python
from data_loader import create_data_loaders

train_loader, val_loader, class_weights = create_data_loaders(
    video_dir="./Videos",
    annotation_dir="./Annotations",
    batch_size=32,
    clip_length=15,  # Must be odd
    stride=1,
    test_size=0.2
)

# Each batch contains clips from different trials
for clips, labels in train_loader:
    # clips: (batch, clip_length, height, width)
    # labels: (batch,)
    pass
```

### Training with Multimodal Model

```python
from multimodal_data_loader import create_multimodal_data_loaders

train_loader, val_loader, class_weights = create_multimodal_data_loaders(
    video_dir="./Videos",
    annotation_dir="./Annotations",
    dlc_dir="./Videos",  # Or "./DLC" if you organize them separately
    batch_size=32,
    clip_length=15,
    stride=1,
    test_size=0.2
)

# Each batch contains clips from different trials with pose features
for visual, pose, labels in train_loader:
    # visual: (batch, clip_length, height, width)
    # pose: (batch, clip_length, 18)  # 8 lengths + 10 angles
    # labels: (batch,)
    pass
```

## Validation

Run the validation script to verify your data structure:

```bash
python3 validate_simple.py
```

This will:
- List all annotation files found
- Parse video names and trial numbers
- Verify video files exist
- Check annotation frame counts (should be 360)
- Calculate frame offsets
- Find DLC files
- Show trial-to-video mapping

## Train/Val Split Strategy

**Current Implementation**: Split by trial
- 3 trials × 0.8 = 2.4 → 2 trials for training
- 3 trials × 0.2 = 0.6 → 1 trial for validation
- Trials are randomly assigned to train/val

**Alternative** (if needed): Split by video
- Would keep all trials from the same video in the same split
- Useful if you want to avoid any data leakage between trials
- Can be implemented by grouping trials before splitting

To implement video-level splitting, modify the split logic in `create_data_loaders()`:

```python
# Group trials by video first
video_groups = {}
for video_path, anno_path in zip(all_video_paths, all_anno_paths):
    if video_path not in video_groups:
        video_groups[video_path] = []
    video_groups[video_path].append((video_path, anno_path))

# Split by unique videos
video_paths_unique = list(video_groups.keys())
train_videos, val_videos = train_test_split(video_paths_unique, ...)

# Flatten back to trials
train_trials = [trial for v in train_videos for trial in video_groups[v]]
val_trials = [trial for v in val_videos for trial in video_groups[v]]
```

## Important Notes

1. **Frame Offset Calculation**: Always `(trial_number - 1) × 360`
   - Trial numbering starts at 1 (not 0)
   - Frame indexing starts at 0

2. **DLC Data Sharing**: One DLC file covers all trials in a video
   - DLC file must have at least `max_trial_number × 360` frames
   - Each trial gets its slice: `dlc_coords[offset:offset+360]`

3. **Validation**: Each trial must have exactly 360 frames in annotations
   - This is checked during preprocessing
   - Videos must have length = `num_trials × 360`

4. **Memory Efficiency**:
   - Videos are not loaded into memory - frames are read on-demand
   - DLC coordinates are loaded once per video and sliced per trial
   - Pose features are computed during preprocessing (not on-the-fly)

## Example Output

When running `validate_simple.py`:

```
============================================================
Validating Multi-Trial Data Structure
============================================================

Found 3 annotation files (trials):
  - prediction_shortened_2023-10-25_CFA_010_267M_tracking.mp4_1.csv
  - prediction_shortened_2023-10-25_CFA_010_267M_tracking.mp4_2.csv
  - prediction_shortened_2023-10-25_CFA_010_267M_tracking.mp4_3.csv

============================================================
Analyzing File Naming Structure
============================================================

  Annotation: prediction_shortened_2023-10-25_CFA_010_267M_tracking.mp4_1.csv
    Parsed video name: shortened_2023-10-25_CFA_010_267M_tracking.mp4
    Parsed trial number: 1
    Video file: shortened_2023-10-25_CFA_010_267M_tracking.mp4 ✓ EXISTS
    Annotation frames: 360 ✓
    Frame offset in video: 0
    Frame range: 0 to 359

  [... similar for trials 2 and 3 ...]

============================================================
Summary
============================================================

  Unique videos: 1
    shortened_2023-10-25_CFA_010_267M_tracking.mp4: 3 trials (trials 1, 2, 3)

  Total trials: 3

✓ Structure Analysis Complete
```

## Training the Models

Now that the data loaders are updated, you can train as before:

### Visual-Only Model
```bash
python train.py
```

### Multimodal Model (Visual + DLC)
```bash
python multimodal_train.py
```

The training scripts don't need modification - they use `create_data_loaders()` and `create_multimodal_data_loaders()` which now handle the multi-trial structure automatically.

## Troubleshooting

### Issue: "No annotation files found"
- Check that annotation files start with `prediction_`
- Verify they are in `./Annotations/` directory

### Issue: "Video not found"
- Annotation filename contains video name with `.mp4` extension
- Video file should match exactly (after removing `prediction_` prefix and `_N` suffix)

### Issue: "DLC file not found"
- DLC file should be in `./Videos/` or specified `dlc_dir`
- Filename should contain "DLC" or "dlc"
- Base name should match video name

### Issue: "Frame count mismatch"
- Each annotation file must have exactly 360 frames
- Video must have at least `max_trial_number × 360` frames
- DLC must have at least `max_trial_number × 360` frames

## Files Modified

1. `data_loader.py` - Visual-only data loading
2. `multimodal_data_loader.py` - Multimodal (visual + DLC) data loading
3. `validate_simple.py` - Validation script (new)
4. `MULTI_TRIAL_UPDATE.md` - This documentation (new)
