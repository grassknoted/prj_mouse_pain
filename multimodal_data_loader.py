"""
Multimodal data loading pipeline for mouse pain action recognition.
Combines visual features (video frames) with pose features (DLC coordinates).
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pose_graph import PoseGraph


class MultimodalMouseActionDataset(Dataset):
    """
    Multimodal dataset combining visual frames and pose features.

    Each sample contains:
    - Visual clip: (T, H, W) temporal frames
    - Pose features: (T, 18) temporal pose features (8 lengths + 10 angles from pose_graph)
    - Label: Action class for center frame

    Args:
        video_paths: List of video file paths
        annotation_paths: List of annotation CSV file paths
        dlc_paths: List of DLC coordinate CSV file paths
        clip_length: Number of frames in temporal clip (must be odd)
        stride: Stride for sampling clips
        normalize_visual: Normalize frames to [0,1]
        normalize_pose: Normalize pose features (z-score normalization)
    """

    ACTION_CLASSES = {
        0: "rest",
        1: "paw_withdraw",
        2: "paw_lick",
        3: "paw_guard",     # Will merge to paw_withdraw
        4: "paw_shake",
        5: "flinch",        # Will merge to paw_withdraw
        6: "walk",
        7: "active"
    }

    NUM_CLASSES = 7  # After merging

    def __init__(
        self,
        video_paths: List[str],
        annotation_paths: List[str],
        dlc_paths: List[str],
        clip_length: int = 16,
        stride: int = 1,
        normalize_visual: bool = True,
        normalize_pose: bool = True
    ):
        assert len(video_paths) == len(annotation_paths) == len(dlc_paths), \
            "Number of videos, annotations, and DLC files must match"
        assert clip_length % 2 == 1, "clip_length must be odd"

        self.video_paths = video_paths
        self.annotation_paths = annotation_paths
        self.dlc_paths = dlc_paths
        self.clip_length = clip_length
        self.stride = stride
        self.normalize_visual = normalize_visual
        self.normalize_pose = normalize_pose
        self.half_clip = clip_length // 2

        # Data storage
        self.video_data = []
        self.clip_indices = []
        self.pose_stats = None  # For normalization

        self._preprocess_data()

        if normalize_pose:
            self._compute_pose_statistics()

    def _preprocess_data(self):
        """Load all data and validate consistency."""
        for video_idx, (video_path, anno_path, dlc_path) in enumerate(
            zip(self.video_paths, self.annotation_paths, self.dlc_paths)
        ):
            # Load annotation
            df_anno = pd.read_csv(anno_path)
            labels = df_anno['Action'].values.astype(np.int64)
            labels = self._merge_action_classes(labels)

            # Load DLC coordinates
            dlc_coords = self._load_dlc_coordinates(dlc_path)

            # Get video metadata
            video = cv2.VideoCapture(video_path)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            video.release()

            # Validate consistency
            assert frame_count == len(labels), \
                f"Frame count mismatch in {video_path}: {frame_count} vs {len(labels)}"
            assert dlc_coords.shape[0] == len(labels), \
                f"DLC frame count mismatch in {dlc_path}: {dlc_coords.shape[0]} vs {len(labels)}"

            # Compute pose features from DLC coordinates
            pose_features = self._compute_pose_features(dlc_coords)

            self.video_data.append({
                'video_path': video_path,
                'labels': labels,
                'pose_features': pose_features,
                'frame_count': frame_count,
                'height': height,
                'width': width
            })

            # Create clip indices (only for frames with full context)
            for center_frame in range(self.half_clip, frame_count - self.half_clip, self.stride):
                self.clip_indices.append((video_idx, center_frame))

    def _load_dlc_coordinates(self, dlc_path: str) -> np.ndarray:
        """
        Load DLC coordinates from CSV.

        Expected format: CSV with columns like:
        bodypart1_x, bodypart1_y, bodypart1_likelihood,
        bodypart2_x, bodypart2_y, bodypart2_likelihood, ...

        Returns:
            (N_frames, N_keypoints, 2) array of (x, y) coordinates
        """
        df = pd.read_csv(dlc_path, header=[1, 2])  # DLC default headers

        # Extract x, y coordinates (skip likelihood)
        coords_list = []

        # Get unique bodyparts
        bodyparts = df.columns.get_level_values(0).unique()

        for bodypart in bodyparts:
            x = df[(bodypart, 'x')].values
            y = df[(bodypart, 'y')].values
            coords_list.append(np.stack([x, y], axis=1))

        # Stack: (N_frames, N_keypoints, 2)
        coords = np.stack(coords_list, axis=1)

        return coords

    def _compute_pose_features(self, dlc_coords: np.ndarray) -> np.ndarray:
        """
        Compute pose features (lengths + angles) from DLC coordinates.

        Args:
            dlc_coords: (N_frames, N_keypoints, 2) array

        Returns:
            (N_frames, 18) array of pose features
        """
        n_frames = dlc_coords.shape[0]
        pose_features = np.zeros((n_frames, 18))

        for frame_idx in range(n_frames):
            keypoints = dlc_coords[frame_idx]  # (N_keypoints, 2)

            # Create PoseGraph and extract features
            try:
                pose_graph = PoseGraph(keypoints)
                features = pose_graph.construct_graph()
                pose_features[frame_idx] = np.array(features)
            except Exception as e:
                # If pose extraction fails, use zeros
                pose_features[frame_idx] = 0.0

        return pose_features

    def _compute_pose_statistics(self):
        """Compute mean and std for pose feature normalization."""
        all_pose_features = []

        for video_info in self.video_data:
            all_pose_features.append(video_info['pose_features'])

        all_pose_features = np.concatenate(all_pose_features, axis=0)

        self.pose_stats = {
            'mean': all_pose_features.mean(axis=0),
            'std': all_pose_features.std(axis=0) + 1e-8  # Avoid division by zero
        }

    def _merge_action_classes(self, labels: np.ndarray) -> np.ndarray:
        """Merge paw_guard (3) and flinch (5) into paw_withdraw (1)."""
        labels = labels.copy()
        mapping = {0: 0, 1: 1, 2: 2, 3: 1, 4: 3, 5: 1, 6: 4, 7: 5}
        labels = np.array([mapping[label] for label in labels])
        return labels

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a multimodal sample.

        Returns:
            visual_clip: (T, H, W) visual frames
            pose_clip: (T, 18) pose features
            label: integer class label
        """
        video_idx, center_frame = self.clip_indices[idx]
        video_info = self.video_data[video_idx]

        # Extract frame range
        start_frame = center_frame - self.half_clip
        end_frame = center_frame + self.half_clip + 1

        # Load visual clip
        visual_clip = self._load_visual_clip(
            video_info['video_path'], start_frame, end_frame
        )

        # Extract pose clip
        pose_clip = video_info['pose_features'][start_frame:end_frame]

        # Normalize pose features
        if self.normalize_pose and self.pose_stats is not None:
            pose_clip = (pose_clip - self.pose_stats['mean']) / self.pose_stats['std']

        # Normalize visual
        if self.normalize_visual:
            visual_clip = visual_clip.float() / 255.0

        # Get label
        label = int(video_info['labels'][center_frame])

        return visual_clip, torch.FloatTensor(pose_clip), label

    def _load_visual_clip(self, video_path: str, start_frame: int, end_frame: int) -> torch.Tensor:
        """Load consecutive frames from video."""
        frames = []
        video = cv2.VideoCapture(video_path)

        for frame_idx in range(start_frame, end_frame):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video.read()

            if not ret:
                raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

            # Convert to grayscale
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames.append(frame)

        video.release()

        # Stack frames: (T, H, W)
        clip = np.stack(frames, axis=0)
        return torch.from_numpy(clip).uint8()

    @staticmethod
    def get_class_weights(dataset: 'MultimodalMouseActionDataset') -> torch.Tensor:
        """Compute class weights for handling imbalance."""
        all_labels = []
        for video_info in dataset.video_data:
            all_labels.extend(video_info['labels'])

        all_labels = np.array(all_labels)
        unique, counts = np.unique(all_labels, return_counts=True)

        weights = np.zeros(dataset.NUM_CLASSES)
        total = len(all_labels)

        for cls_idx, count in zip(unique, counts):
            weights[cls_idx] = total / (dataset.NUM_CLASSES * count)

        return torch.FloatTensor(weights)


def create_multimodal_data_loaders(
    video_dir: str,
    annotation_dir: str,
    dlc_dir: str,
    batch_size: int = 32,
    clip_length: int = 16,
    stride: int = 1,
    num_workers: int = 4,
    test_size: float = 0.2,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Create train and validation dataloaders with multimodal data.

    Args:
        video_dir: Directory with video files
        annotation_dir: Directory with annotation CSVs
        dlc_dir: Directory with DLC coordinate CSVs
        batch_size: Batch size
        clip_length: Temporal clip length
        stride: Stride for sampling
        num_workers: Number of workers
        test_size: Validation fraction
        random_seed: Random seed

    Returns:
        train_loader, val_loader, class_weights
    """
    # Find all matching triplets (video, annotation, DLC)
    video_files = sorted(Path(video_dir).glob("*.mp4"))

    video_paths = []
    annotation_paths = []
    dlc_paths = []

    for video_file in video_files:
        # Find matching annotation
        anno_pattern = f"{video_file.stem}*.csv"
        anno_files = list(Path(annotation_dir).glob(anno_pattern))

        if not anno_files:
            print(f"Warning: No annotation found for {video_file.name}")
            continue

        # Find matching DLC file (should be in dlc_dir with similar naming)
        dlc_pattern = f"{video_file.stem}*.csv"
        dlc_files = list(Path(dlc_dir).glob(dlc_pattern))

        if not dlc_files:
            print(f"Warning: No DLC file found for {video_file.name}")
            continue

        video_paths.append(str(video_file))
        annotation_paths.append(str(anno_files[0]))
        dlc_paths.append(str(dlc_files[0]))

    if not video_paths:
        raise RuntimeError(f"No matching triplets found in {video_dir}, {annotation_dir}, {dlc_dir}")

    print(f"Found {len(video_paths)} video-annotation-DLC triplets")

    # Split by video
    video_indices = np.arange(len(video_paths))
    train_indices, val_indices = train_test_split(
        video_indices,
        test_size=test_size,
        random_state=random_seed
    )

    train_videos = [video_paths[i] for i in train_indices]
    train_annos = [annotation_paths[i] for i in train_indices]
    train_dlcs = [dlc_paths[i] for i in train_indices]

    val_videos = [video_paths[i] for i in val_indices]
    val_annos = [annotation_paths[i] for i in val_indices]
    val_dlcs = [dlc_paths[i] for i in val_indices]

    print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")

    # Create datasets
    train_dataset = MultimodalMouseActionDataset(
        train_videos, train_annos, train_dlcs,
        clip_length=clip_length,
        stride=stride,
        normalize_visual=True,
        normalize_pose=True
    )

    val_dataset = MultimodalMouseActionDataset(
        val_videos, val_annos, val_dlcs,
        clip_length=clip_length,
        stride=stride,
        normalize_visual=True,
        normalize_pose=True
    )

    # Get class weights
    class_weights = MultimodalMouseActionDataset.get_class_weights(train_dataset)
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
    # Test
    video_dir = "./Videos"
    annotation_dir = "./Annotations"
    dlc_dir = "./DLC"  # Adjust to your DLC directory

    try:
        train_loader, val_loader, class_weights = create_multimodal_data_loaders(
            video_dir, annotation_dir, dlc_dir,
            batch_size=4,
            clip_length=16,
            stride=2
        )

        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        # Test one batch
        for batch_idx, (visual, pose, labels) in enumerate(train_loader):
            print(f"Visual shape: {visual.shape}")
            print(f"Pose shape: {pose.shape}")
            print(f"Labels shape: {labels.shape}")
            break

    except Exception as e:
        print(f"Error: {e}")
