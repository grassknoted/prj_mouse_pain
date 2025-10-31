"""
Validate the data structure and file matching logic without loading video data.
"""

from pathlib import Path
import pandas as pd

print("=" * 60)
print("Validating Multi-Trial Data Structure")
print("=" * 60)

# Find all annotation files
anno_files = sorted(Path("./Annotations").glob("prediction_*.csv"))
print(f"\nFound {len(anno_files)} annotation files (trials):")

# Group by video
video_to_trials = {}

for anno_file in anno_files:
    # Extract video name from annotation filename
    # Format: prediction_<video_name>_<trial_number>.csv
    anno_stem = anno_file.stem
    parts = anno_stem.split('_')

    # Reconstruct video name
    video_name_parts = parts[1:-1]
    video_name = '_'.join(video_name_parts)
    trial_number = int(parts[-1])

    # Find the actual video file
    video_file = Path("./Videos") / f"{video_name}.mp4"

    if not video_file.exists():
        print(f"  ✗ Video not found: {video_file}")
        continue

    video_path = str(video_file)

    if video_path not in video_to_trials:
        video_to_trials[video_path] = []

    # Load annotation to check frame count
    df = pd.read_csv(anno_file)
    frame_count = len(df)

    video_to_trials[video_path].append({
        'trial_number': trial_number,
        'anno_file': str(anno_file),
        'frame_count': frame_count
    })

print(f"\n✓ Found {len(video_to_trials)} unique video(s)")

# Display video-trial structure
for video_path, trials in video_to_trials.items():
    video_name = Path(video_path).name
    print(f"\n  Video: {video_name}")

    sorted_trials = sorted(trials, key=lambda x: x['trial_number'])

    for trial in sorted_trials:
        print(f"    Trial {trial['trial_number']}: {trial['frame_count']} frames")
        print(f"      Frame range: {(trial['trial_number'] - 1) * 360} to {trial['trial_number'] * 360 - 1}")
        print(f"      Annotation: {Path(trial['anno_file']).name}")

        # Validate 360 frames
        if trial['frame_count'] != 360:
            print(f"      ⚠️  WARNING: Expected 360 frames, got {trial['frame_count']}")

    # Calculate expected video length
    num_trials = len(sorted_trials)
    expected_frames = num_trials * 360
    print(f"    Expected video length: {expected_frames} frames ({num_trials} trials × 360)")

# Find DLC files
print(f"\n{'=' * 60}")
print("Checking DLC Files")
print("=" * 60)

for video_path in video_to_trials.keys():
    video_name = Path(video_path).stem

    # Look for DLC files in Videos directory
    dlc_pattern = f"{video_name}*.csv"
    dlc_files = list(Path("./Videos").glob(dlc_pattern))

    print(f"\n  Video: {Path(video_path).name}")

    if dlc_files:
        dlc_file = dlc_files[0]
        print(f"    ✓ DLC file found: {dlc_file.name}")

        # Try to load and check structure
        try:
            df_dlc = pd.read_csv(dlc_file, header=[1, 2], nrows=5)
            print(f"    ✓ DLC file format is valid (DeepLabCut multi-index)")
            print(f"      Body parts: {df_dlc.columns.get_level_values(0).unique().tolist()}")
        except Exception as e:
            print(f"    ⚠️  Could not parse DLC file: {e}")
    else:
        print(f"    ✗ No DLC file found matching pattern: {dlc_pattern}")

print(f"\n{'=' * 60}")
print("Summary")
print("=" * 60)

total_trials = sum(len(trials) for trials in video_to_trials.values())
print(f"  Total videos: {len(video_to_trials)}")
print(f"  Total trials: {total_trials}")
print(f"  Average trials per video: {total_trials / len(video_to_trials):.1f}")

print(f"\n✓ Data structure validation complete!")
print(f"\nThe updated data loaders will:")
print(f"  1. Treat each trial as an independent sample")
print(f"  2. Extract frames from correct video positions using frame offsets")
print(f"  3. Match DLC data to each trial's frame range")
print(f"  4. Split by trials (not videos) for train/val sets")
