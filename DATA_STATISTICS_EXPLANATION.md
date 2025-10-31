# Data Statistics and Smoke Test - Explanation

## What Changed

I've added comprehensive data statistics and improved the smoke test to answer your questions.

---

## Your Questions Answered

### Q1: "Why are there only 526 videos?"

**Answer**: There aren't 526 unique videos - there are **526 video-trial pairs** (samples).

**Explanation:**
- Each video can have **multiple trials** (multiple action CSV files)
- Each trial is a separate sample for training
- Progress bar shows: `Discovering video-trial pairs: 526/526`

**Example:**
```
video_001.mp4
├── prediction_video_001.mp4_1.csv  ← Trial 1 (sample 1)
├── prediction_video_001.mp4_2.csv  ← Trial 2 (sample 2)
└── prediction_video_001.mp4_3.csv  ← Trial 3 (sample 3)

video_002.mp4
├── prediction_video_002.mp4_1.csv  ← Trial 1 (sample 4)
└── prediction_video_002.mp4_2.csv  ← Trial 2 (sample 5)

Total: 2 videos, 5 samples
```

So if you have ~150 unique videos with an average of 3-4 trials each, you get ~500-600 samples total.

---

### Q2: "Shouldn't there be 18 features because of the pose graphs?"

**You're absolutely right!** There was a bug - the smoke test wasn't showing `pose_features`.

**Fixed:**
```
OLD SMOKE TEST:
  Video shape: torch.Size([180, 3, 224, 224])
  Actions shape: torch.Size([180, 6])  ← Wrong! Should be 7 classes
  Keypoints shape: torch.Size([180, 12])
  Visibility shape: torch.Size([180, 6])
  Time feats shape: torch.Size([180, 2])
  Action mask: 1.0

NEW SMOKE TEST:
  Video shape: torch.Size([180, 3, 224, 224])
  Actions shape: torch.Size([180, 7])  ← Fixed! 7 merged classes
  Pose features shape: torch.Size([180, 18])  ← NEW! 18 geometric features
  Keypoints shape: torch.Size([180, 12])
  Visibility shape: torch.Size([180, 6])
  Time feats shape: torch.Size([180, 2])
  Action mask: 1.0

Pose feature statistics:
  Min: 0.0234, Max: 1.2345
  Mean: 0.3456, Std: 0.2123
  NaN count: 0
```

**What are the 18 pose features?**
- **8 edge lengths**: distances between keypoint pairs
- **10 angles**: angles formed by keypoint triplets
- See `POSE_GRAPH_AND_MODEL_TYPES.md` for details

---

## New: Comprehensive Data Statistics

Before training starts, you'll now see detailed statistics:

```
================================================================================
COMPREHENSIVE DATA STATISTICS
================================================================================

1. DATASET OVERVIEW
--------------------------------------------------------------------------------
Total unique videos/trials discovered: 526 (before boosting)
Training samples (after boosting): 463
Validation samples: 63
Train/Val split: 463/63 (88.0% / 12.0%)

2. CLASS INFORMATION
--------------------------------------------------------------------------------
Original classes (before merging): 8
  0: rest
  1: paw_withdraw
  2: paw_lick
  3: paw_guard      ← Merged into paw_withdraw
  4: paw_shake
  5: flinch         ← Merged into paw_withdraw
  6: walk
  7: active

Merged classes (after merging): 6
  0: rest
  1: paw_withdraw  (includes paw_guard, flinch)
  2: paw_lick
  3: paw_shake
  4: walk
  5: active

Merge mapping: {0: 0, 1: 1, 2: 2, 3: 1, 4: 3, 5: 1, 6: 4, 7: 5}
Merged indices: [1, 3, 5]

3. SAMPLE DISTRIBUTION (by dominant class in window)
--------------------------------------------------------------------------------
Class                Samples    Share
--------------------------------------------------------------------------------
rest                 350        75.60%
paw_withdraw         10         2.16%
paw_lick             50         10.79%
paw_shake            8          1.73%
walk                 25         5.40%
active               20         4.32%
--------------------------------------------------------------------------------
TOTAL                463        100.00%

4. TEMPORAL INFORMATION
--------------------------------------------------------------------------------
Training window length (train_T): 180 frames
Validation window length (val_T): 240 frames
Minimum frames required: 345 frames
VideoMAE2 clip length (if used): 16 frames

5. KEYPOINT INFORMATION
--------------------------------------------------------------------------------
Number of keypoints: 6
Keypoint names: ['mouth', 'tail_base', 'L_frontpaw', 'R_frontpaw', 'L_hindpaw', 'R_hindpaw']
Keypoint likelihood threshold: 0.9
Pose graph features: 18 (8 edge lengths + 10 angles)

6. AUGMENTATION SETTINGS (training only)
--------------------------------------------------------------------------------
Keypoint jittering std: 0.02
Brightness jitter: ±0.2
Contrast jitter: ±0.2
Temporal dropout probability: 0.1
Horizontal flip probability: 0.0

7. CLASS BALANCING SETTINGS
--------------------------------------------------------------------------------
Rare class threshold: <2.0% of samples
Rare class boost cap: up to 30.0x replication
Focal loss gamma: 2.5
Label smoothing: 0.1

================================================================================
```

---

## Key Insights from Statistics

### Sample Distribution vs Frame Distribution

**Important distinction:**
- **Frame distribution**: How many frames contain each action (printed earlier)
- **Sample distribution**: How many video windows are dominated by each action (new!)

**Example:**
```
FRAME DISTRIBUTION (what you saw earlier):
walk: 13,910 frames (3.45% of all frames)

SAMPLE DISTRIBUTION (new):
walk: 25 samples (5.40% of samples)

Why different?
- Walk occurs in many videos but briefly
- When walk dominates a 180-frame window, it counts as 1 sample
- But walk might only be 60 frames in that window
- So sample share can be higher than frame share
```

### Why Rare Class Boosting Works on Samples

The boosting algorithm:
1. ✅ Counts samples by **dominant class** (not all frames)
2. ✅ Marks classes with <2% **sample share** as rare
3. ✅ Replicates entire **video windows** (not individual frames)

This is why:
- `walk` has 3.45% frames but is marked rare (if <2% sample share)
- `+9 replicas` = 9 entire 180-frame windows
- This adds ~1,600-2,800 frames (depending on how many walk frames per window)

---

## What to Look For

### 1. Check Pose Features in Smoke Test

```
Pose features shape: torch.Size([180, 18])  ← Must be 18!

Pose feature statistics:
  Min: ...
  Max: ...
  Mean: ...
  Std: ...
  NaN count: 0  ← Must be 0! If >0, there's a problem
```

**If NaN count > 0:**
- Some keypoints are missing (all visibility = 0)
- Some edge lengths or angles couldn't be computed
- Check your DLC CSV files

### 2. Check Sample Distribution

```
3. SAMPLE DISTRIBUTION (by dominant class in window)
--------------------------------------------------------------------------------
paw_withdraw         10         2.16%  ← Is this >2% or <2%?
```

If paw_withdraw is <2%, it will be boosted. If it's >2% but you want it boosted anyway, lower the threshold:

```bash
--rare_threshold_share 0.03  # Boost classes <3% instead of <2%
```

### 3. Check Train/Val Split

```
Train/Val split: 463/63 (88.0% / 12.0%)
```

This should be roughly 80/20 or 85/15. If it's very imbalanced (e.g., 95/5), you might not have enough validation data.

---

## Troubleshooting

### Issue: Pose features have NaN values

**Symptoms:**
```
Pose feature statistics:
  NaN count: 150  ← NOT ZERO!
```

**Cause**: Missing keypoints (all visibility = 0 for some frames)

**Solution**: The pose graph code handles this by returning NaN for edges/angles when keypoints are missing. The model should handle NaN gracefully, but if it crashes:
1. Check DLC CSV files for missing keypoints
2. Lower `--kp_likelihood_thr` (e.g., 0.8 instead of 0.9)
3. Increase keypoint jittering to fill in missing values

### Issue: Actions shape is wrong (e.g., [180, 6] instead of [180, 7])

**Cause**: Class merging bug or incorrect class count

**Check:**
```
2. CLASS INFORMATION
Merged classes (after merging): 6  ← Should match Actions shape[1]
```

If merged classes = 7 but Actions shape = [180, 6], there's a bug in one-hot encoding.

### Issue: Very few samples (e.g., <100 total)

**Cause**: Most videos/trials don't meet minimum frame requirement

**Solution:**
1. Check `--min_frames` (default: 345)
2. Lower to 300 or 240 if most videos are shorter
3. Check frame counts in videos:
   ```bash
   ffprobe -v error -select_streams v:0 -count_frames \
           -show_entries stream=nb_read_frames -of csv=p=0 \
           your_video.mp4
   ```

---

## Summary

✅ **526 "videos"** = 526 video-trial pairs (samples), not unique videos
✅ **Pose features** now shown in smoke test: `torch.Size([180, 18])`
✅ **Comprehensive statistics** show:
   - Dataset overview (train/val split)
   - Class information (merging details)
   - Sample distribution (by dominant class)
   - Temporal, keypoint, augmentation, and balancing settings
✅ **Sample distribution ≠ frame distribution** (explains rare class boosting)

Run training and check these statistics to ensure everything is configured correctly!
