# Skipping Invalid Trials

## Change Summary

Both data loaders now **skip trials** that don't have exactly 360 frames instead of crashing with an assertion error.

## What Happens Now

### Before (Old Behavior)
```python
AssertionError: Trial .../prediction_..._1.csv has 345 frames, expected 360
```
→ Training stops immediately ❌

### After (New Behavior)
```python
Warning: Skipping trial .../prediction_..._1.csv - has 345 frames, expected 360
```
→ Training continues with valid trials ✅

## Files Updated

1. **`data_loader.py`** - Visual-only data loader
2. **`multimodal_data_loader.py`** - Multimodal (visual + DLC) data loader

## How It Works

During data preprocessing, the loaders now:

1. **Check each trial** for exactly 360 frames
2. **If frame count ≠ 360:**
   - Print a warning message with the trial path and actual frame count
   - Skip that trial (don't add it to the dataset)
   - Continue processing remaining trials
3. **If frame count = 360:**
   - Process normally

## Example Output

When loading data, you'll see warnings for any invalid trials:

```
Found 100 annotation files (trials)
Warning: Skipping trial .../prediction_video1_1.csv - has 345 frames, expected 360
Warning: Skipping trial .../prediction_video3_2.csv - has 358 frames, expected 360
Found 1 unique videos with 98 total trials
Train trials: 78, Val trials: 20
```

## Why This Matters

In your dataset, some trials may be:
- **Truncated** (e.g., 345 frames instead of 360)
- **Corrupted** during annotation
- **From different experiment protocols** with different lengths

By skipping invalid trials, you can:
- ✅ Train on the majority of valid data
- ✅ Avoid crashes during training
- ✅ See which trials have issues (check the warnings)

## Checking Your Data

To see which trials will be skipped before training, run:

```bash
python3 validate_simple.py
```

This will show frame counts for all trials and flag any that aren't 360 frames.

## What If Too Many Trials Are Skipped?

If you see many trials being skipped:

1. **Check the warnings** to see which trials have issues
2. **Verify your annotation files:**
   - Do they have a header row? (They should)
   - Are they complete?
   - Were they generated correctly?

3. **Check video/annotation matching:**
   - Does the trial number match the video segment?
   - Is the video long enough for that trial number?

## Manual Fix (If Needed)

If you have trials with different frame counts that you want to use, you can modify the data loader to:

1. **Pad short trials** with zeros or repeat last frame
2. **Truncate long trials** to 360 frames
3. **Use variable-length trials** (requires more complex changes)

For now, the simple approach is to **skip invalid trials** and train on the valid ones.

## Impact on Training

- **Good**: Dataset will only contain valid, consistent-length trials
- **Good**: No crashes during training
- **Note**: Total number of training samples will be reduced by the number of skipped trials
- **Check**: Make sure you have enough valid trials left for training (at least 50-100)

## Statistics After Loading

After calling `create_data_loaders()` or `create_multimodal_data_loaders()`, you'll see:

```python
Found 1 unique videos with 98 total trials  # 2 trials were skipped
Train trials: 78, Val trials: 20
```

This tells you:
- How many valid trials were found
- How many were split into train/val sets
- (Implicitly: how many were skipped = total found - total loaded)

## Recommendation

Before training on your full dataset:

1. Run `validate_simple.py` to see trial statistics
2. Check warnings during data loading
3. If too many trials are invalid, investigate the annotation files
4. Once you have mostly valid trials, proceed with training!
