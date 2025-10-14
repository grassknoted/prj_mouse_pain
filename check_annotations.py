import os
import re

def analyze_annotation_files(folder_path):
    """
    Scans `folder_path` for CSV files matching the format:
      prediction_<VIDEO_NAME>.mp4_<TRIAL_NUM>.csv
      OR
      predictions_<VIDEO_NAME>.mp4_<TRIAL_NUM>.csv

    Where <TRIAL_NUM> is digits only (e.g., 1, 2, 3).

    Excludes any file that has an extra underscore after the trial number
    or doesn't end exactly with .csv in that format.

    Prints:
      - The count of files per video name
      - Summary stats (total files, unique videos, average files per video)
    """

    # This regex enforces:
    # 1) predictions?_ at the start ("prediction" or "predictions")
    # 2) (.+\.mp4)   : captures any characters until ".mp4" (this includes .mp4 in the group)
    # 3) _(\\d+)     : underscore + digits (the trial number)
    # 4) \.csv$      : must end with .csv (no extra underscores)
    pattern = re.compile(r"^predictions?_(.+\.mp4)_(\d+)\.csv$")

    video_counts = {}  # key = the string "<VIDEO_NAME>.mp4", value = count of matching files

    for fname in os.listdir(folder_path):
        # Only consider .csv files
        if not fname.endswith(".csv"):
            continue

        match = pattern.match(fname)
        if match:
            # match.group(1) will be something like "MyVideo.mp4"
            # match.group(2) is the trial number (digits)
            video_file = match.group(1)  # includes ".mp4" at the end
            # trial_num = int(match.group(2))  # if you need it

            if video_file not in video_counts:
                video_counts[video_file] = 0
            video_counts[video_file] += 1
        else:
            # This file does NOT match the exact pattern -> excluded
            pass

    # Print counts per video
    print("Counts per video (including the .mp4 in the name):")
    for video_file, count in sorted(video_counts.items()):
        print(f"  {video_file}: {count}")

    # Compute summary statistics
    total_files = sum(video_counts.values())
    unique_videos = len(video_counts)
    avg_per_video = (total_files / unique_videos) if unique_videos > 0 else 0.0

    print("\nSummary statistics:")
    print(f"  Total valid files found: {total_files}")
    print(f"  Unique video names: {unique_videos}")
    print(f"  Average files per video: {avg_per_video:.2f}")

if __name__ == "__main__":
    FOLDER_TO_CHECK = "PROCESSED_ANNOTATIONS"  # Change to your folder path if needed
    analyze_annotation_files(FOLDER_TO_CHECK)
