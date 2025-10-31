"""
Simple validation of data structure using only built-in Python modules.
"""

from pathlib import Path

print("=" * 60)
print("Validating Multi-Trial Data Structure")
print("=" * 60)

# Find all annotation files
anno_files = sorted(Path("./Annotations").glob("prediction_*.csv"))
print(f"\nFound {len(anno_files)} annotation files (trials):")

for anno_file in anno_files:
    print(f"  - {anno_file.name}")

# Analyze structure
print(f"\n{'=' * 60}")
print("Analyzing File Naming Structure")
print("=" * 60)

video_to_trials = {}

for anno_file in anno_files:
    anno_stem = anno_file.stem
    parts = anno_stem.split('_')

    # Get trial number (last part)
    trial_number = parts[-1]

    # Reconstruct video name (everything except 'prediction' and trial number)
    video_name_parts = parts[1:-1]
    video_name = '_'.join(video_name_parts)

    # Check if video_name already has .mp4 extension
    if not video_name.endswith('.mp4'):
        video_name = f"{video_name}.mp4"

    print(f"\n  Annotation: {anno_file.name}")
    print(f"    Parsed video name: {video_name}")
    print(f"    Parsed trial number: {trial_number}")

    # Check if video exists
    video_file = Path("./Videos") / video_name
    print(f"    Video file: {video_file.name} {'✓ EXISTS' if video_file.exists() else '✗ NOT FOUND'}")

    # Count lines in annotation
    with open(anno_file, 'r') as f:
        line_count = sum(1 for line in f) - 1  # Subtract header
    print(f"    Annotation frames: {line_count} {'✓' if line_count == 360 else '⚠️  (expected 360)'}")

    # Calculate frame offset
    try:
        trial_num_int = int(trial_number)
        frame_offset = (trial_num_int - 1) * 360
        frame_range = f"{frame_offset} to {frame_offset + 359}"
        print(f"    Frame offset in video: {frame_offset}")
        print(f"    Frame range: {frame_range}")
    except ValueError:
        print(f"    ⚠️  Could not parse trial number as integer")

    if video_file.exists():
        video_path = str(video_file)
        if video_path not in video_to_trials:
            video_to_trials[video_path] = []
        video_to_trials[video_path].append(trial_number)

# Find DLC files
print(f"\n{'=' * 60}")
print("Checking DLC Files")
print("=" * 60)

dlc_files = list(Path("./Videos").glob("*DLC*.csv"))
print(f"\nFound {len(dlc_files)} DLC files in Videos/:")
for dlc_file in dlc_files:
    print(f"  - {dlc_file.name}")

# Summary
print(f"\n{'=' * 60}")
print("Summary")
print("=" * 60)

print(f"\n  Unique videos: {len(video_to_trials)}")
for video_path, trial_numbers in video_to_trials.items():
    video_name = Path(video_path).name
    print(f"    {video_name}: {len(trial_numbers)} trials (trials {', '.join(sorted(trial_numbers))})")

print(f"\n  Total trials: {len(anno_files)}")

print(f"\n{'=' * 60}")
print("✓ Structure Analysis Complete")
print("=" * 60)

print("\nKey Changes Made to Data Loaders:")
print("  1. Each trial (360 frames) is treated as an independent sample")
print("  2. Trial number is extracted from annotation filename")
print("  3. Frame offset = (trial_number - 1) × 360")
print("  4. Video frames are loaded from offset to offset+360")
print("  5. DLC coordinates are sliced per trial from full video DLC data")
print("  6. Train/val split is done by trial (not by video)")
