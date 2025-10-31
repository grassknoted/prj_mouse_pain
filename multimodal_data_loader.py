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
from torch.utils.data.distributed import DistributedSampler
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
        normalize_pose: bool = True,
        use_extracted_frames: bool = True,
        extracted_frames_dir: str = None
    ):
        assert len(video_paths) == len(annotation_paths) == len(dlc_paths), \
            "Number of videos, annotations, and DLC files must match"
        assert clip_length % 2 == 1, \
            f"clip_length must be odd for centering (got {clip_length}). Use 15, 17, 19, etc."

        self.video_paths = video_paths
        self.annotation_paths = annotation_paths
        self.dlc_paths = dlc_paths
        self.clip_length = clip_length
        self.stride = stride
        self.normalize_visual = normalize_visual
        self.normalize_pose = normalize_pose
        self.half_clip = clip_length // 2
        self.use_extracted_frames = use_extracted_frames
        self.extracted_frames_dir = extracted_frames_dir

        # Data storage
        self.video_data = []
        self.clip_indices = []
        self.pose_stats = None  # For normalization
        self._frames_cache = {}  # Cache loaded .npy files (works with multiprocessing)

        self._preprocess_data()

        if normalize_pose:
            self._compute_pose_statistics()

    def _preprocess_data(self):
        """Load all data and validate consistency for multi-trial structure with variable lengths."""
        print(f"[Multimodal Loader] Starting preprocessing of {len(self.video_paths)} video-annotation-DLC triplets...")

        # First pass: group trials by video
        video_trial_info = {}

        for idx, (video_path, anno_path, dlc_path) in enumerate(zip(self.video_paths, self.annotation_paths, self.dlc_paths)):
            if idx % 10 == 0:
                print(f"[Multimodal Loader] Processing annotation {idx+1}/{len(self.video_paths)}: {Path(anno_path).name}")

            # Load annotation
            df_anno = pd.read_csv(anno_path)
            labels = df_anno['Action'].values.astype(np.int64)
            labels = self._merge_action_classes(labels)

            # Get trial length
            trial_length = len(labels)

            # Validate minimum length
            if trial_length < self.clip_length:
                print(f"[Multimodal Loader] Warning: Skipping trial {anno_path} - has {trial_length} frames, need at least {self.clip_length}")
                continue

            # Extract trial number
            anno_filename = Path(anno_path).stem
            trial_number = int(anno_filename.split('_')[-1])

            # Group by video
            if video_path not in video_trial_info:
                video_trial_info[video_path] = []

            video_trial_info[video_path].append({
                'anno_path': anno_path,
                'dlc_path': dlc_path,
                'trial_number': trial_number,
                'labels': labels,
                'trial_length': trial_length
            })

        print(f"[Multimodal Loader] Grouped trials into {len(video_trial_info)} unique videos")

        # Second pass: calculate cumulative offsets and load data
        trial_idx = 0

        print(f"[Multimodal Loader] Second pass: loading video metadata and DLC coordinates...")
        for vid_idx, (video_path, trials) in enumerate(video_trial_info.items()):
            print(f"[Multimodal Loader] Processing video {vid_idx+1}/{len(video_trial_info)}: {Path(video_path).name}")

            # Sort trials by trial number
            trials_sorted = sorted(trials, key=lambda x: x['trial_number'])

            # Get video metadata
            print(f"[Multimodal Loader]   Opening video to get metadata...")
            video = cv2.VideoCapture(video_path)
            video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            video.release()
            print(f"[Multimodal Loader]   Video has {video_frame_count} frames, {height}x{width} resolution")

            # Validate trial structure
            expected_num_trials = video_frame_count / 360
            if video_frame_count % 360 != 0:
                print(f"[Multimodal Loader]   WARNING: Video has {video_frame_count} frames, not a multiple of 360!")
                print(f"[Multimodal Loader]   Expected whole trials, got {expected_num_trials:.2f} trials")
            else:
                expected_num_trials = int(expected_num_trials)
                print(f"[Multimodal Loader]   Expected {expected_num_trials} trials (360 frames each)")

                if len(trials_sorted) != expected_num_trials:
                    print(f"[Multimodal Loader]   WARNING: Found {len(trials_sorted)} annotation files, expected {expected_num_trials}!")

            # Load DLC coordinates for entire video (once per video)
            dlc_path = trials_sorted[0]['dlc_path']  # All trials share same DLC file
            print(f"[Multimodal Loader]   Loading DLC coordinates from {Path(dlc_path).name}...")
            dlc_coords = self._load_dlc_coordinates(dlc_path)
            print(f"[Multimodal Loader]   DLC loaded: {dlc_coords.shape[0]} frames, {dlc_coords.shape[1]} keypoints")

            # Validate DLC has correct number of frames
            if dlc_coords.shape[0] != video_frame_count:
                print(f"[Multimodal Loader]   WARNING: DLC has {dlc_coords.shape[0]} frames, but video has {video_frame_count} frames!")

            # Calculate cumulative offsets
            cumulative_offset = 0

            print(f"[Multimodal Loader]   Processing {len(trials_sorted)} trials for this video...")
            for trial_info in trials_sorted:
                trial_length = trial_info['trial_length']

                # Check if video has enough frames
                if cumulative_offset + trial_length > video_frame_count:
                    print(f"[Multimodal Loader]   Warning: Skipping trial {trial_info['anno_path']} - "
                          f"video has {video_frame_count} frames, but trial needs frames "
                          f"{cumulative_offset} to {cumulative_offset + trial_length - 1}")
                    cumulative_offset += trial_length
                    continue

                # Check if DLC has enough frames
                if cumulative_offset + trial_length > dlc_coords.shape[0]:
                    print(f"[Multimodal Loader]   Warning: Skipping trial {trial_info['anno_path']} - "
                          f"DLC has {dlc_coords.shape[0]} frames, but trial needs frames "
                          f"{cumulative_offset} to {cumulative_offset + trial_length - 1}")
                    cumulative_offset += trial_length
                    continue

                # Extract DLC coordinates for this trial
                trial_dlc_coords = dlc_coords[cumulative_offset:cumulative_offset + trial_length]

                # Compute pose features
                print(f"[Multimodal Loader]   Computing pose features for trial {trial_info['trial_number']}...")
                pose_features = self._compute_pose_features(trial_dlc_coords)

                self.video_data.append({
                    'video_path': video_path,
                    'labels': trial_info['labels'],
                    'pose_features': pose_features,
                    'frame_offset': cumulative_offset,
                    'trial_length': trial_length,
                    'height': height,
                    'width': width,
                    'trial_number': trial_info['trial_number'],
                    'anno_path': trial_info['anno_path']
                })

                # Create clip indices
                for center_frame in range(self.half_clip, trial_length - self.half_clip, self.stride):
                    self.clip_indices.append((trial_idx, center_frame))

                trial_idx += 1
                cumulative_offset += trial_length

    def _load_dlc_coordinates(self, dlc_path: str) -> np.ndarray:
        """
        Load DLC coordinates from CSV.

        Expected format: CSV with columns like:
        bodypart1_x, bodypart1_y, bodypart1_likelihood,
        bodypart2_x, bodypart2_y, bodypart2_likelihood, ...

        Returns:
            (N_frames, N_keypoints, 2) array of (x, y) coordinates
        """
        print(f"\n[DLC Loader] ========================================")
        print(f"[DLC Loader] Loading DLC file: {Path(dlc_path).name}")

        # Try to read with multi-level header first (DLC default)
        try:
            print(f"[DLC Loader] Attempting multi-level header read (header=[1, 2])...")
            df = pd.read_csv(dlc_path, header=[1, 2])

            # Check if this is actually multi-level
            if isinstance(df.columns, pd.MultiIndex):
                print(f"[DLC Loader] ✓ Multi-level header detected!")
                print(f"[DLC Loader] Column structure: MultiIndex with {len(df.columns.levels)} levels")

                # Show level 0 (bodyparts)
                bodyparts = df.columns.get_level_values(0).unique()
                print(f"[DLC Loader] Level 0 (bodyparts): {list(bodyparts)[:10]}{'...' if len(bodyparts) > 10 else ''}")

                # Show level 1 (coords)
                coords_labels = df.columns.get_level_values(1).unique()
                print(f"[DLC Loader] Level 1 (coords): {list(coords_labels)}")

                coords_list = []
                valid_bodyparts = []

                for bodypart in bodyparts:
                    # Skip header labels
                    if bodypart in ['scorer', 'bodyparts', 'coords']:
                        print(f"[DLC Loader]   Skipping header label: {bodypart}")
                        continue
                    try:
                        x = df[(bodypart, 'x')].values
                        y = df[(bodypart, 'y')].values
                        coords_list.append(np.stack([x, y], axis=1))
                        valid_bodyparts.append(bodypart)
                        print(f"[DLC Loader]   ✓ Loaded bodypart '{bodypart}': {len(x)} frames")
                    except KeyError as e:
                        print(f"[DLC Loader]   ✗ Failed to load bodypart '{bodypart}': {e}")
                        continue

                if coords_list:
                    coords = np.stack(coords_list, axis=1)
                    print(f"[DLC Loader] ✓ Multi-level loading successful!")
                    print(f"[DLC Loader] Coordinates shape: {coords.shape} (frames, keypoints, xy)")
                    print(f"[DLC Loader] Valid bodyparts: {valid_bodyparts}")
                    print(f"[DLC Loader] Sample coordinates (frame 0):")
                    for i, bp in enumerate(valid_bodyparts[:6]):
                        print(f"[DLC Loader]   {bp}: ({coords[0, i, 0]:.2f}, {coords[0, i, 1]:.2f})")
                    print(f"[DLC Loader] ========================================\n")
                    return coords
        except Exception as e:
            print(f"[DLC Loader] ✗ Multi-level header failed: {e}")

        # Fallback: try single header with column patterns
        print(f"[DLC Loader] Attempting single-level header read...")
        df = pd.read_csv(dlc_path)

        print(f"[DLC Loader] Dataframe shape: {df.shape}")
        print(f"[DLC Loader] First 10 column names: {list(df.columns[:10])}")

        # Find all x columns (ending with _x)
        x_cols = sorted([col for col in df.columns if str(col).endswith('_x')])
        y_cols = sorted([col for col in df.columns if str(col).endswith('_y')])

        print(f"[DLC Loader] Found {len(x_cols)} x columns: {x_cols[:5]}{'...' if len(x_cols) > 5 else ''}")
        print(f"[DLC Loader] Found {len(y_cols)} y columns: {y_cols[:5]}{'...' if len(y_cols) > 5 else ''}")

        if len(x_cols) != len(y_cols):
            raise ValueError(f"Mismatch in x/y columns in {dlc_path}: {len(x_cols)} x cols, {len(y_cols)} y cols")

        coords_list = []
        for x_col, y_col in zip(x_cols, y_cols):
            x = df[x_col].values
            y = df[y_col].values
            coords_list.append(np.stack([x, y], axis=1))

        # Stack: (N_frames, N_keypoints, 2)
        coords = np.stack(coords_list, axis=1)

        print(f"[DLC Loader] ✓ Single-level loading successful!")
        print(f"[DLC Loader] Coordinates shape: {coords.shape} (frames, keypoints, xy)")
        print(f"[DLC Loader] Sample coordinates (frame 0):")
        for i, (x_col, y_col) in enumerate(zip(x_cols[:6], y_cols[:6])):
            print(f"[DLC Loader]   {x_col.replace('_x', '')}: ({coords[0, i, 0]:.2f}, {coords[0, i, 1]:.2f})")
        print(f"[DLC Loader] ========================================\n")

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
        n_keypoints = dlc_coords.shape[1]

        print(f"[Pose Features] Computing pose features for {n_frames} frames with {n_keypoints} keypoints")

        # Check if we have the expected 6 keypoints for PoseGraph
        if n_keypoints != 6:
            print(f"[Pose Features] WARNING: PoseGraph expects 6 keypoints, but got {n_keypoints}")
            print(f"[Pose Features] Will use first 6 keypoints if available, or pad with zeros")

        pose_features = np.zeros((n_frames, 18))
        errors = 0

        for frame_idx in range(n_frames):
            keypoints = dlc_coords[frame_idx]  # (N_keypoints, 2)

            # Ensure we have exactly 6 keypoints
            if n_keypoints < 6:
                # Pad with zeros if not enough keypoints
                keypoints_padded = np.zeros((6, 2))
                keypoints_padded[:n_keypoints] = keypoints
                keypoints = keypoints_padded
            elif n_keypoints > 6:
                # Use only first 6 keypoints
                keypoints = keypoints[:6]

            # Create PoseGraph and extract features
            try:
                pose_graph = PoseGraph(keypoints)
                features = pose_graph.construct_graph()
                pose_features[frame_idx] = np.array(features)

                # Show sample features for first frame
                if frame_idx == 0:
                    print(f"[Pose Features] Sample features (frame 0):")
                    print(f"[Pose Features]   Edge lengths (8): {features[:8]}")
                    print(f"[Pose Features]   Angles (10): {features[8:]}")

            except Exception as e:
                # If pose extraction fails, use zeros
                pose_features[frame_idx] = 0.0
                errors += 1
                if errors == 1:
                    print(f"[Pose Features] ERROR in frame {frame_idx}: {e}")

        if errors > 0:
            print(f"[Pose Features] WARNING: {errors}/{n_frames} frames had errors, using zero features")

        print(f"[Pose Features] ✓ Pose features computed: shape {pose_features.shape}")
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
        trial_idx, center_frame = self.clip_indices[idx]
        trial_info = self.video_data[trial_idx]

        # Extract frame range relative to trial
        start_frame = center_frame - self.half_clip
        end_frame = center_frame + self.half_clip + 1

        # Convert to absolute frame indices in the video
        start_frame_abs = trial_info['frame_offset'] + start_frame
        end_frame_abs = trial_info['frame_offset'] + end_frame

        # Load visual clip using absolute frame indices
        visual_clip = self._load_visual_clip(
            trial_info['video_path'], start_frame_abs, end_frame_abs
        )

        # Extract pose clip (relative to trial, since pose_features is already trial-specific)
        pose_clip = trial_info['pose_features'][start_frame:end_frame]

        # Normalize pose features
        if self.normalize_pose and self.pose_stats is not None:
            pose_clip = (pose_clip - self.pose_stats['mean']) / self.pose_stats['std']

        # Normalize visual
        if self.normalize_visual:
            visual_clip = visual_clip.float() / 255.0

        # Get label (relative to trial)
        label = int(trial_info['labels'][center_frame])

        return visual_clip, torch.FloatTensor(pose_clip), label

    def _load_visual_clip(self, video_path: str, start_frame: int, end_frame: int) -> torch.Tensor:
        """Load consecutive frames from extracted .npy file or video."""

        if self.use_extracted_frames:
            # Use pre-extracted frames (FAST)
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
                # Load all frames at once (cached for subsequent clips from same video)
                self._frames_cache[video_path] = np.load(frames_path, mmap_mode='r')

            all_frames = self._frames_cache[video_path]

            # Extract clip
            clip = all_frames[start_frame:end_frame]
            return torch.from_numpy(np.array(clip)).to(torch.uint8)

        else:
            # Fallback to reading from video (SLOW)
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

                # Convert to grayscale
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                frames.append(frame)

            video.release()

            # Stack frames: (T, H, W)
            clip = np.stack(frames, axis=0)
            return torch.from_numpy(clip).to(torch.uint8)

    @staticmethod
    def get_class_weights(dataset: 'MultimodalMouseActionDataset') -> torch.Tensor:
        """Compute class weights for handling imbalance."""
        all_labels = []
        for trial_info in dataset.video_data:
            all_labels.extend(trial_info['labels'])

        all_labels = np.array(all_labels)
        unique, counts = np.unique(all_labels, return_counts=True)

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


def create_multimodal_data_loaders(
    video_dir: str,
    annotation_dir: str,
    dlc_dir: str,
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
    Create train and validation dataloaders with multimodal data for multi-trial structure.

    Args:
        video_dir: Directory with video files
        annotation_dir: Directory with annotation CSVs (one per trial)
        dlc_dir: Directory with DLC coordinate CSVs (one per video)
        batch_size: Batch size
        clip_length: Temporal clip length
        stride: Stride for sampling
        num_workers: Number of workers
        test_size: Validation fraction
        random_seed: Random seed

    Returns:
        train_loader, val_loader, class_weights
    """
    print(f"\n[create_multimodal_data_loaders] Starting...")
    print(f"[create_multimodal_data_loaders] video_dir: {video_dir}")
    print(f"[create_multimodal_data_loaders] annotation_dir: {annotation_dir}")
    print(f"[create_multimodal_data_loaders] dlc_dir: {dlc_dir}")

    # Find all annotation files (each represents one trial)
    print(f"[create_multimodal_data_loaders] Searching for annotation files...")
    anno_files = sorted(Path(annotation_dir).glob("prediction_*.csv"))
    print(f"[create_multimodal_data_loaders] Found {len(anno_files)} annotation files")

    if not anno_files:
        raise RuntimeError(f"No annotation files found in {annotation_dir}")

    # Group annotations by video and find matching DLC files
    print(f"[create_multimodal_data_loaders] Matching annotations to videos and DLC files...")
    all_video_paths = []
    all_anno_paths = []
    all_dlc_paths = []

    video_to_dlc = {}  # Cache video -> DLC mapping

    for idx, anno_file in enumerate(anno_files):
        if idx % 20 == 0:
            print(f"[create_multimodal_data_loaders] Processing annotation {idx+1}/{len(anno_files)}")
        # Extract video name from annotation filename
        # Format: prediction_<video_name>_<trial_number>.csv
        # Note: video_name may already include .mp4 extension
        anno_stem = anno_file.stem
        parts = anno_stem.split('_')

        # Reconstruct video name (remove 'prediction' prefix and trial number suffix)
        video_name_parts = parts[1:-1]
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

        # Find matching DLC file (one DLC file per video, not per trial)
        if video_path not in video_to_dlc:
            # DLC files should be named similar to video (without .mp4 extension in pattern)
            video_stem = Path(video_name).stem  # Remove .mp4 if present
            dlc_pattern = f"{video_stem}*DLC*.csv"
            dlc_files = list(Path(dlc_dir).glob(dlc_pattern))

            if not dlc_files:
                # Try searching in video_dir if not in dlc_dir
                dlc_files = list(Path(video_dir).glob(dlc_pattern))

            if not dlc_files:
                # Try a more general pattern
                dlc_files = list(Path(video_dir).glob(f"{video_stem}*.csv"))
                # Filter to only DLC files (contain 'DLC' in name)
                dlc_files = [f for f in dlc_files if 'DLC' in f.name or 'dlc' in f.name]

            if not dlc_files:
                print(f"Warning: No DLC file found for {video_name}")
                continue

            # Use the first matching DLC file
            video_to_dlc[video_path] = str(dlc_files[0])

        all_video_paths.append(video_path)
        all_anno_paths.append(str(anno_file))
        all_dlc_paths.append(video_to_dlc[video_path])

    if not all_video_paths:
        raise RuntimeError(f"No matching triplets found in {video_dir}, {annotation_dir}, {dlc_dir}")

    print(f"Found {len(set(all_video_paths))} unique videos with {len(all_anno_paths)} total trials")

    # Split by TRIAL (not by video) to treat each trial independently
    trial_indices = np.arange(len(all_anno_paths))
    train_indices, val_indices = train_test_split(
        trial_indices,
        test_size=test_size,
        random_state=random_seed
    )

    train_videos = [all_video_paths[i] for i in train_indices]
    train_annos = [all_anno_paths[i] for i in train_indices]
    train_dlcs = [all_dlc_paths[i] for i in train_indices]

    val_videos = [all_video_paths[i] for i in val_indices]
    val_annos = [all_anno_paths[i] for i in val_indices]
    val_dlcs = [all_dlc_paths[i] for i in val_indices]

    print(f"Train trials: {len(train_videos)}, Val trials: {len(val_videos)}")

    # Set extracted frames directory if not provided
    if use_extracted_frames and extracted_frames_dir is None:
        extracted_frames_dir = str(Path(video_dir) / "../extracted_frames")

    # Create datasets
    train_dataset = MultimodalMouseActionDataset(
        train_videos, train_annos, train_dlcs,
        clip_length=clip_length,
        stride=stride,
        normalize_visual=True,
        normalize_pose=True,
        use_extracted_frames=use_extracted_frames,
        extracted_frames_dir=extracted_frames_dir
    )

    val_dataset = MultimodalMouseActionDataset(
        val_videos, val_annos, val_dlcs,
        clip_length=clip_length,
        stride=stride,
        normalize_visual=True,
        normalize_pose=True,
        use_extracted_frames=use_extracted_frames,
        extracted_frames_dir=extracted_frames_dir
    )

    # Get class weights
    class_weights = MultimodalMouseActionDataset.get_class_weights(train_dataset)
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
