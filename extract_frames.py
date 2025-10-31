"""
Extract all video frames and save as numpy arrays for fast loading during training.
This is a one-time preprocessing step that dramatically speeds up training.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def extract_video_frames(video_path: str, output_dir: str):
    """
    Extract all frames from a video and save as a single .npy file.

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames

    Returns:
        Number of frames extracted
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output filename (same as video, but .npy extension)
    output_file = output_dir / f"{video_path.stem}.npy"

    # Skip if already extracted
    if output_file.exists():
        print(f"  Skipping {video_path.name} (already extracted)")
        return 0

    # Open video
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        print(f"  ERROR: Could not open {video_path}")
        return 0

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"  Extracting {frame_count} frames ({height}x{width}) from {video_path.name}")

    # Pre-allocate array for all frames (grayscale)
    frames = np.zeros((frame_count, height, width), dtype=np.uint8)

    # Extract all frames
    for frame_idx in tqdm(range(frame_count), desc=f"  {video_path.name}", leave=False):
        ret, frame = video.read()

        if not ret:
            print(f"  WARNING: Failed to read frame {frame_idx}, stopping early")
            frames = frames[:frame_idx]  # Trim to actual frames read
            break

        # Convert to grayscale
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frames[frame_idx] = frame

    video.release()

    # Save as compressed numpy array
    np.save(output_file, frames)

    # Print size info
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved {len(frames)} frames to {output_file.name} ({size_mb:.1f} MB)")

    return len(frames)


def extract_all_videos(video_dir: str, output_dir: str):
    """
    Extract frames from all videos in a directory.

    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save extracted frames
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)

    # Find all video files
    all_videos = sorted(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")))

    # Filter out macOS metadata files (._filename)
    video_files = [v for v in all_videos if not v.name.startswith("._")]

    if not video_files:
        print(f"No video files found in {video_dir}")
        return

    skipped = len(all_videos) - len(video_files)
    if skipped > 0:
        print(f"Skipping {skipped} macOS metadata files (._*)")
        print()

    print(f"Found {len(video_files)} videos to extract")
    print(f"Output directory: {output_dir}")
    print("="*80)

    total_frames = 0

    for video_file in video_files:
        num_frames = extract_video_frames(str(video_file), str(output_dir))
        total_frames += num_frames

    print("="*80)
    print(f"✓ Extraction complete!")
    print(f"  Total frames extracted: {total_frames:,}")
    print(f"  Output directory: {output_dir}")

    # Estimate disk usage
    total_size_gb = sum(f.stat().st_size for f in output_dir.glob("*.npy")) / (1024**3)
    print(f"  Total disk usage: {total_size_gb:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video frames for fast training")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save extracted frames")

    args = parser.parse_args()

    extract_all_videos(args.video_dir, args.output_dir)
