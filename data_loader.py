"""
Data loading pipeline for mouse pain action recognition.
Handles video frame extraction, temporal clip creation, and label management.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split


class MouseActionDataset(Dataset):
    """
    Dataset for mouse action recognition from video clips.

    Each sample is a temporal clip of frames with a center frame label.
    Args:
        video_paths: List of video file paths
        annotation_paths: List of annotation CSV file paths
        clip_length: Number of frames in each clip (must be odd for centering)
        stride: Stride for sampling clips (1 = every frame, 2 = every other frame)
        transform: Optional transforms to apply to frames
    """

    # Action class mapping
    ACTION_CLASSES = {
        0: "rest",
        1: "paw_withdraw",  # paw_guard and flinch merged into this
        2: "paw_lick",
        3: "paw_guard",     # Will be merged to paw_withdraw
        4: "paw_shake",
        5: "flinch",        # Will be merged to paw_withdraw
        6: "walk",
        7: "active"
    }

    NUM_CLASSES = 7  # After merging paw_guard and flinch

    def __init__(
        self,
        video_paths: List[str],
        annotation_paths: List[str],
        clip_length: int = 16,
        stride: int = 1,
        transform=None,
        normalize: bool = True,
        use_extracted_frames: bool = True,
        extracted_frames_dir: str = None
    ):
        assert len(video_paths) == len(annotation_paths), \
            "Number of videos must match number of annotations"
        assert clip_length % 2 == 1, \
            f"clip_length must be odd for centering (got {clip_length}). Use 15, 17, 19, etc."

        self.video_paths = video_paths
        self.annotation_paths = annotation_paths
        self.clip_length = clip_length
        self.stride = stride
        self.transform = transform
        self.normalize = normalize
        self.half_clip = clip_length // 2
        self.use_extracted_frames = use_extracted_frames
        self.extracted_frames_dir = extracted_frames_dir

        # Preload annotations and video metadata
        self.video_data = []
        self.clip_indices = []  # (video_idx, center_frame_idx)
        self._frames_cache = {}  # Cache loaded .npy files

        self._preprocess_data()

    def _preprocess_data(self):
        """Load annotations and build clip indices for multi-trial structure."""
        # First pass: group trials by video and calculate cumulative offsets
        video_trial_info = {}  # Maps video_path -> list of (anno_path, trial_number, labels)

        for video_path, anno_path in zip(self.video_paths, self.annotation_paths):
            # Load annotation for this trial
            df = pd.read_csv(anno_path)
            labels = df['Action'].values.astype(np.int64)

            # Merge action classes
            labels = self._merge_action_classes(labels)

            # Get trial length
            trial_length = len(labels)

            # Validate minimum length
            if trial_length < self.clip_length:
                print(f"Warning: Skipping trial {anno_path} - has {trial_length} frames, need at least {self.clip_length}")
                continue

            # Extract trial number
            anno_filename = Path(anno_path).stem
            trial_number = int(anno_filename.split('_')[-1])

            # Group by video
            if video_path not in video_trial_info:
                video_trial_info[video_path] = []

            video_trial_info[video_path].append({
                'anno_path': anno_path,
                'trial_number': trial_number,
                'labels': labels,
                'trial_length': trial_length
            })

        # Second pass: calculate cumulative frame offsets for each video
        trial_idx = 0

        for video_path, trials in video_trial_info.items():
            # Sort trials by trial number
            trials_sorted = sorted(trials, key=lambda x: x['trial_number'])

            # Get video metadata
            video = cv2.VideoCapture(video_path)
            video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            video.release()

            # Calculate cumulative frame offsets
            cumulative_offset = 0

            for trial_info in trials_sorted:
                trial_length = trial_info['trial_length']

                # Check if video has enough frames
                if cumulative_offset + trial_length > video_frame_count:
                    print(f"Warning: Skipping trial {trial_info['anno_path']} - "
                          f"video has {video_frame_count} frames, but trial needs frames "
                          f"{cumulative_offset} to {cumulative_offset + trial_length - 1}")
                    cumulative_offset += trial_length  # Still update offset for subsequent trials
                    continue

                self.video_data.append({
                    'path': video_path,
                    'labels': trial_info['labels'],
                    'frame_offset': cumulative_offset,
                    'trial_length': trial_length,
                    'height': height,
                    'width': width,
                    'trial_number': trial_info['trial_number'],
                    'anno_path': trial_info['anno_path']
                })

                # Create clip indices for this trial
                for center_frame in range(self.half_clip, trial_length - self.half_clip, self.stride):
                    self.clip_indices.append((trial_idx, center_frame))

                trial_idx += 1
                cumulative_offset += trial_length

    def _merge_action_classes(self, labels: np.ndarray) -> np.ndarray:
        """Merge paw_guard (3) and flinch (5) into paw_withdraw (1)."""
        labels = labels.copy()
        # Map class indices after merge:
        # 0: rest -> 0
        # 1: paw_withdraw -> 1
        # 2: paw_lick -> 2
        # 3: paw_guard -> 1 (merge to paw_withdraw)
        # 4: paw_shake -> 3
        # 5: flinch -> 1 (merge to paw_withdraw)
        # 6: walk -> 4
        # 7: active -> 5

        mapping = {0: 0, 1: 1, 2: 2, 3: 1, 4: 3, 5: 1, 6: 4, 7: 5}
        labels = np.array([mapping[label] for label in labels])
        return labels

    def __len__(self):
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a temporal clip and its center frame label.

        Returns:
            clip: (T, H, W) tensor of frames, normalized to [0, 1]
            label: integer class label
        """
        trial_idx, center_frame = self.clip_indices[idx]
        trial_info = self.video_data[trial_idx]

        # Extract frame range relative to trial
        start_frame = center_frame - self.half_clip
        end_frame = center_frame + self.half_clip + 1

        # Convert to absolute frame indices in the video
        start_frame_abs = trial_info['frame_offset'] + start_frame
        end_frame_abs = trial_info['frame_offset'] + end_frame

        # Load clip from video using absolute frame indices
        clip = self._load_clip(trial_info['path'], start_frame_abs, end_frame_abs)

        # Get label from center frame (relative to trial)
        label = int(trial_info['labels'][center_frame])

        # Apply transform if any
        if self.transform:
            clip = self.transform(clip)

        # Normalize to [0, 1]
        if self.normalize:
            clip = clip.float() / 255.0

        return clip, label

    def _load_clip(self, video_path: str, start_frame: int, end_frame: int) -> torch.Tensor:
        """Load consecutive frames from extracted .npy file or video.

        If use_extracted_frames is True, loads from pre-extracted numpy arrays (FAST).
        Otherwise, falls back to reading from video file (SLOW).
        """
        if self.use_extracted_frames:
            # Use pre-extracted frames (FAST - ~100x speedup)
            video_name = Path(video_path).stem
            frames_path = Path(self.extracted_frames_dir) / f"{video_name}.npy"

            # Load entire video frames if not cached
            if video_path not in self._frames_cache:
                if not frames_path.exists():
                    raise RuntimeError(
                        f"Extracted frames not found: {frames_path}\n"
                        f"Run: python extract_frames.py --video_dir {Path(video_path).parent} "
                        f"--output_dir {self.extracted_frames_dir}"
                    )
                # Load all frames at once using memory mapping (cached for subsequent clips from same video)
                # Memory mapping means frames are loaded on-demand without reading entire file into RAM
                self._frames_cache[video_path] = np.load(frames_path, mmap_mode='r')

            all_frames = self._frames_cache[video_path]

            # Extract the clip (simple array slicing - instant)
            clip = all_frames[start_frame:end_frame]
            return torch.from_numpy(np.array(clip)).to(torch.uint8)

        else:
            # Fallback to reading from video (SLOW)
            # Note: We don't cache video handles because they can't be shared across
            # DataLoader worker processes. Each worker will open/close as needed.
            frames = []
            video = cv2.VideoCapture(video_path)

            if not video.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")

            for frame_idx in range(start_frame, end_frame):
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = video.read()

                if not ret:
                    video.release()
                    raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

                # Convert BGR to grayscale (already grayscale in your case, but ensure it)
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                frames.append(frame)

            video.release()

            # Stack frames: (T, H, W)
            clip = np.stack(frames, axis=0)
            return torch.from_numpy(clip).to(torch.uint8)

    @staticmethod
    def get_class_weights(dataset: 'MouseActionDataset') -> torch.Tensor:
        """
        Compute class weights to handle imbalance.
        Used for weighted loss functions.
        """
        all_labels = []
        for trial_info in dataset.video_data:
            all_labels.extend(trial_info['labels'])

        all_labels = np.array(all_labels)
        unique, counts = np.unique(all_labels, return_counts=True)

        # Inverse frequency weighting
        weights = np.zeros(dataset.NUM_CLASSES)
        total = len(all_labels)

        # Map from merged class indices to names
        merged_class_names = {
            0: "rest",
            1: "paw_withdraw",  # Merged: paw_withdraw + paw_guard + flinch
            2: "paw_lick",
            3: "paw_shake",
            4: "walk",
            5: "active"
        }

        print("\n" + "="*60)
        print("Class Distribution and Weights")
        print("="*60)

        for cls_idx, count in zip(unique, counts):
            weight = total / (dataset.NUM_CLASSES * count)
            weights[cls_idx] = weight
            class_name = merged_class_names.get(cls_idx, f"unknown_{cls_idx}")
            percentage = (count / total) * 100
            print(f"Class {cls_idx} ({class_name:15s}): {count:6d} samples ({percentage:5.2f}%) -> weight: {weight:.4f}")

        # Check for missing classes
        for cls_idx in range(dataset.NUM_CLASSES):
            if cls_idx not in unique:
                class_name = merged_class_names.get(cls_idx, f"unknown_{cls_idx}")
                print(f"Class {cls_idx} ({class_name:15s}): {0:6d} samples (  0.00%) -> weight: 0.0000 (NOT PRESENT)")

        print("="*60 + "\n")

        return torch.FloatTensor(weights)


def create_data_loaders(
    video_dir: str,
    annotation_dir: str,
    batch_size: int = 32,
    clip_length: int = 16,
    stride: int = 1,
    num_workers: int = 4,
    test_size: float = 0.2,
    random_seed: int = 42,
    distributed: bool = False,
    use_extracted_frames: bool = True,
    extracted_frames_dir: str = None
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Create train and validation dataloaders for multi-trial structure.

    Args:
        video_dir: Directory containing video files
        annotation_dir: Directory containing annotation CSVs (one per trial)
        batch_size: Batch size for loaders
        clip_length: Temporal clip length
        stride: Stride for clip sampling
        num_workers: Number of workers for data loading
        test_size: Fraction for validation set
        random_seed: Random seed for reproducibility

    Returns:
        train_loader, val_loader, class_weights
    """
    # Find all annotation files (each represents one trial)
    anno_files = sorted(Path(annotation_dir).glob("prediction_*.csv"))

    if not anno_files:
        raise RuntimeError(f"No annotation files found in {annotation_dir}")

    # Group annotations by video
    video_to_annos = {}
    for anno_file in anno_files:
        # Extract video name from annotation filename
        # Format: prediction_<video_name>_<trial_number>.csv
        # Note: video_name may already include .mp4 extension
        anno_stem = anno_file.stem
        parts = anno_stem.split('_')

        # Remove 'prediction' prefix and trial number suffix
        # Reconstruct video name
        video_name_parts = parts[1:-1]  # Everything except 'prediction' and trial number
        video_name = '_'.join(video_name_parts)

        # Check if video_name already has .mp4 extension
        if not video_name.endswith('.mp4'):
            video_name = f"{video_name}.mp4"

        # Find the actual video file
        video_file = Path(video_dir) / video_name

        if not video_file.exists():
            print(f"Warning: Video not found for {anno_file.name}: {video_file}")
            continue

        video_path = str(video_file)
        if video_path not in video_to_annos:
            video_to_annos[video_path] = []
        video_to_annos[video_path].append(str(anno_file))

    if not video_to_annos:
        raise RuntimeError(f"No matching video-annotation pairs found")

    # Flatten to get all trials
    all_video_paths = []
    all_anno_paths = []

    for video_path, anno_list in video_to_annos.items():
        for anno_path in sorted(anno_list):  # Sort to ensure trial order
            all_video_paths.append(video_path)
            all_anno_paths.append(anno_path)

    print(f"Found {len(set(all_video_paths))} unique videos with {len(all_anno_paths)} total trials")

    # Split by TRIAL (not by video) to treat each trial independently
    # If you want to split by video (keep all trials from same video together),
    # you would need to group differently
    trial_indices = np.arange(len(all_anno_paths))
    train_indices, val_indices = train_test_split(
        trial_indices,
        test_size=test_size,
        random_state=random_seed
    )

    train_videos = [all_video_paths[i] for i in train_indices]
    train_annos = [all_anno_paths[i] for i in train_indices]
    val_videos = [all_video_paths[i] for i in val_indices]
    val_annos = [all_anno_paths[i] for i in val_indices]

    print(f"Train trials: {len(train_videos)}, Val trials: {len(val_videos)}")

    # Set extracted frames directory if not provided
    if use_extracted_frames and extracted_frames_dir is None:
        extracted_frames_dir = str(Path(video_dir).parent / "extracted_frames")

    # Create datasets
    train_dataset = MouseActionDataset(
        train_videos, train_annos,
        clip_length=clip_length,
        stride=stride,
        normalize=True,
        use_extracted_frames=use_extracted_frames,
        extracted_frames_dir=extracted_frames_dir
    )

    val_dataset = MouseActionDataset(
        val_videos, val_annos,
        clip_length=clip_length,
        stride=stride,
        normalize=True,
        use_extracted_frames=use_extracted_frames,
        extracted_frames_dir=extracted_frames_dir
    )

    # Get class weights
    class_weights = MouseActionDataset.get_class_weights(train_dataset)
    print(f"Class weights: {class_weights}")

    # Create dataloaders with optional distributed sampling
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        shuffle_train = False  # Sampler handles shuffling
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    # Note: persistent_workers can cause deadlocks with distributed training
    # Disable it when using multiple GPUs
    use_persistent = (num_workers > 0) and not distributed

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else 2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else 2
    )

    return train_loader, val_loader, class_weights


if __name__ == "__main__":
    # Test the data loader
    video_dir = "/Users/anagara8/Documents/prj_mouse_pain/Videos"
    annotation_dir = "/Users/anagara8/Documents/prj_mouse_pain/Annotations"

    train_loader, val_loader, class_weights = create_data_loaders(
        video_dir, annotation_dir,
        batch_size=4,
        clip_length=16,
        stride=1
    )

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")

    # Inspect one batch
    for batch_idx, (clips, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: clips shape {clips.shape}, labels {labels}")
        break
