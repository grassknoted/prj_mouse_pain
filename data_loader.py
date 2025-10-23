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
        normalize: bool = True
    ):
        assert len(video_paths) == len(annotation_paths), \
            "Number of videos must match number of annotations"
        assert clip_length % 2 == 1, "clip_length must be odd for centering"

        self.video_paths = video_paths
        self.annotation_paths = annotation_paths
        self.clip_length = clip_length
        self.stride = stride
        self.transform = transform
        self.normalize = normalize
        self.half_clip = clip_length // 2

        # Preload annotations and video metadata
        self.video_data = []
        self.clip_indices = []  # (video_idx, center_frame_idx)

        self._preprocess_data()

    def _preprocess_data(self):
        """Load annotations and build clip indices."""
        for video_idx, (video_path, anno_path) in enumerate(
            zip(self.video_paths, self.annotation_paths)
        ):
            # Load annotation
            df = pd.read_csv(anno_path)
            labels = df['Action'].values.astype(np.int64)

            # Merge action classes: paw_guard (3) and flinch (5) -> paw_withdraw (1)
            labels = self._merge_action_classes(labels)

            # Get video metadata without loading entire video
            video = cv2.VideoCapture(video_path)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            video.release()

            assert frame_count == len(labels), \
                f"Frame count mismatch: {frame_count} vs {len(labels)}"

            self.video_data.append({
                'path': video_path,
                'labels': labels,
                'frame_count': frame_count,
                'height': height,
                'width': width
            })

            # Create clip indices for this video
            # Only create clips that have enough context (half_clip frames before/after)
            for center_frame in range(self.half_clip, frame_count - self.half_clip, self.stride):
                self.clip_indices.append((video_idx, center_frame))

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
        video_idx, center_frame = self.clip_indices[idx]
        video_info = self.video_data[video_idx]

        # Extract frame range
        start_frame = center_frame - self.half_clip
        end_frame = center_frame + self.half_clip + 1

        # Load clip from video
        clip = self._load_clip(video_info['path'], start_frame, end_frame)

        # Get label from center frame
        label = int(video_info['labels'][center_frame])

        # Apply transform if any
        if self.transform:
            clip = self.transform(clip)

        # Normalize to [0, 1]
        if self.normalize:
            clip = clip.float() / 255.0

        return clip, label

    def _load_clip(self, video_path: str, start_frame: int, end_frame: int) -> torch.Tensor:
        """Load consecutive frames from video."""
        frames = []
        video = cv2.VideoCapture(video_path)

        for frame_idx in range(start_frame, end_frame):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video.read()

            if not ret:
                raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

            # Convert BGR to grayscale (already grayscale in your case, but ensure it)
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames.append(frame)

        video.release()

        # Stack frames: (T, H, W)
        clip = np.stack(frames, axis=0)
        return torch.from_numpy(clip).uint8()

    @staticmethod
    def get_class_weights(dataset: 'MouseActionDataset') -> torch.Tensor:
        """
        Compute class weights to handle imbalance.
        Used for weighted loss functions.
        """
        all_labels = []
        for video_info in dataset.video_data:
            all_labels.extend(video_info['labels'])

        all_labels = np.array(all_labels)
        unique, counts = np.unique(all_labels, return_counts=True)

        # Inverse frequency weighting
        weights = np.zeros(dataset.NUM_CLASSES)
        total = len(all_labels)

        for cls_idx, count in zip(unique, counts):
            weights[cls_idx] = total / (dataset.NUM_CLASSES * count)

        return torch.FloatTensor(weights)


def create_data_loaders(
    video_dir: str,
    annotation_dir: str,
    batch_size: int = 32,
    clip_length: int = 16,
    stride: int = 1,
    num_workers: int = 4,
    test_size: float = 0.2,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Create train and validation dataloaders.

    Args:
        video_dir: Directory containing video files
        annotation_dir: Directory containing annotation CSVs
        batch_size: Batch size for loaders
        clip_length: Temporal clip length
        stride: Stride for clip sampling
        num_workers: Number of workers for data loading
        test_size: Fraction for validation set
        random_seed: Random seed for reproducibility

    Returns:
        train_loader, val_loader, class_weights
    """
    # Find all video-annotation pairs
    video_files = sorted(Path(video_dir).glob("*.mp4"))

    video_paths = []
    annotation_paths = []

    for video_file in video_files:
        # Try to find matching annotation file
        anno_pattern = f"{video_file.stem}*.csv"
        anno_files = list(Path(annotation_dir).glob(anno_pattern))

        if anno_files:
            # Use the first matching annotation
            video_paths.append(str(video_file))
            annotation_paths.append(str(anno_files[0]))

    if not video_paths:
        raise RuntimeError(f"No matching video-annotation pairs found in {video_dir} and {annotation_dir}")

    print(f"Found {len(video_paths)} video-annotation pairs")

    # Split by video (not by frame) to prevent data leakage
    video_indices = np.arange(len(video_paths))
    train_indices, val_indices = train_test_split(
        video_indices,
        test_size=test_size,
        random_state=random_seed
    )

    train_videos = [video_paths[i] for i in train_indices]
    train_annos = [annotation_paths[i] for i in train_indices]
    val_videos = [video_paths[i] for i in val_indices]
    val_annos = [annotation_paths[i] for i in val_indices]

    print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")

    # Create datasets
    train_dataset = MouseActionDataset(
        train_videos, train_annos,
        clip_length=clip_length,
        stride=stride,
        normalize=True
    )

    val_dataset = MouseActionDataset(
        val_videos, val_annos,
        clip_length=clip_length,
        stride=stride,
        normalize=True
    )

    # Get class weights
    class_weights = MouseActionDataset.get_class_weights(train_dataset)
    print(f"Class weights: {class_weights}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
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
