#!/usr/bin/env python3
"""
Inference Script for Mouse Action Recognition (ActionOnlyModel)

Usage:
    python inference.py /path/to/video_folder

Expects:
    - Folder containing:
        - Video files (*.mp4, *.avi, etc.)
        - DLC annotation CSVs matching pattern: {video_stem}DLC_resnet50_pawtracking_*.csv
    - Trained checkpoint: best_model_multitask.pt in current working directory

Outputs:
    - Creates folder: predictions_{input_folder_name}/
    - For each video, saves CSV with columns:
        - Frame: Frame number (0-indexed)
        - Action: Predicted action label
        - {class}_prob: Probability for each action class
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import timm
except ImportError:
    print("[error] timm not installed. Run: pip install timm>=0.9.16")
    sys.exit(1)

try:
    from transformers import VideoMAEModel
except ImportError:
    print("[warn] transformers not installed. VideoMAE (3D) will not be available.")
    print("[warn] Install with: pip install transformers")
    VideoMAEModel = None

warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# POSE GRAPH COMPUTATION
# ============================================================================

class PoseGraph:
    """
    Compute geometric features from 6 keypoints.

    Features:
    - 8 edge lengths between keypoint pairs
    - 10 angles from keypoint triplets
    Total: 18 features per frame

    Keypoint order: [mouth, tail_base, L_frontpaw, R_frontpaw, L_hindpaw, R_hindpaw]
    """

    def __init__(self):
        # Define edges (pairs of keypoint indices)
        self.edges = [
            (0, 1),  # mouth - tail_base
            (0, 2),  # mouth - L_frontpaw
            (0, 3),  # mouth - R_frontpaw
            (1, 4),  # tail_base - L_hindpaw
            (1, 5),  # tail_base - R_hindpaw
            (2, 4),  # L_frontpaw - L_hindpaw
            (3, 5),  # R_frontpaw - R_hindpaw
            (2, 3),  # L_frontpaw - R_frontpaw
        ]

        # Define angle triplets (center, point1, point2)
        self.angle_triplets = [
            (1, 0, 2),  # tail_base - mouth - L_frontpaw
            (1, 0, 3),  # tail_base - mouth - R_frontpaw
            (0, 1, 4),  # mouth - tail_base - L_hindpaw
            (0, 1, 5),  # mouth - tail_base - R_hindpaw
            (0, 2, 4),  # mouth - L_frontpaw - L_hindpaw
            (0, 3, 5),  # mouth - R_frontpaw - R_hindpaw
            (1, 2, 4),  # tail_base - L_frontpaw - L_hindpaw
            (1, 3, 5),  # tail_base - R_frontpaw - R_hindpaw
            (2, 0, 3),  # L_frontpaw - mouth - R_frontpaw
            (4, 1, 5),  # L_hindpaw - tail_base - R_hindpaw
        ]

    def compute_features(self, coords: np.ndarray, visibility: np.ndarray) -> np.ndarray:
        """
        Compute pose graph features from keypoint coordinates.

        Args:
            coords: (T, K, 2) normalized keypoint coordinates
            visibility: (T, K) visibility mask

        Returns:
            features: (T, 18) pose graph features
        """
        T, K = coords.shape[0], coords.shape[1]
        features = np.zeros((T, 18), dtype=np.float32)

        for t in range(T):
            # Compute edge lengths (8 features)
            for i, (p1, p2) in enumerate(self.edges):
                if visibility[t, p1] > 0 and visibility[t, p2] > 0:
                    delta = coords[t, p2] - coords[t, p1]
                    length = np.sqrt(np.sum(delta ** 2))
                    features[t, i] = length
                else:
                    features[t, i] = 0.0

            # Compute angles (10 features)
            for i, (center, p1, p2) in enumerate(self.angle_triplets):
                if visibility[t, center] > 0 and visibility[t, p1] > 0 and visibility[t, p2] > 0:
                    v1 = coords[t, p1] - coords[t, center]
                    v2 = coords[t, p2] - coords[t, center]
                    len_v1 = np.sqrt(np.sum(v1 ** 2))
                    len_v2 = np.sqrt(np.sum(v2 ** 2))

                    if len_v1 > 1e-6 and len_v2 > 1e-6:
                        cos_angle = np.dot(v1, v2) / (len_v1 * len_v2)
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle = np.arccos(cos_angle)  # Radians
                        features[t, 8 + i] = angle
                    else:
                        features[t, 8 + i] = 0.0
                else:
                    features[t, 8 + i] = 0.0

        return features


# ============================================================================
# DLC CSV PARSING
# ============================================================================

def normalize_keypoint_name(name: str) -> str:
    """Normalize keypoint name to canonical form."""
    name = name.lower().strip().replace(' ', '_')

    # Handle common aliases
    aliases = {
        'snout': 'mouth',
        'nose': 'mouth',
        'rf': 'r_frontpaw',
        'rfp': 'r_frontpaw',
        'rightfrontpaw': 'r_frontpaw',
        'lf': 'l_frontpaw',
        'lfp': 'l_frontpaw',
        'leftfrontpaw': 'l_frontpaw',
        'rh': 'r_hindpaw',
        'rhp': 'r_hindpaw',
        'righthindpaw': 'r_hindpaw',
        'lh': 'l_hindpaw',
        'lhp': 'l_hindpaw',
        'lefthindpaw': 'l_hindpaw',
        'tail': 'tail_base',
        'tailbase': 'tail_base',
    }

    return aliases.get(name, name)


def parse_dlc_csv(
    csv_path: Path,
    video_width: int,
    video_height: int,
    kp_names: List[str],
    likelihood_thr: float = 0.90
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse DLC CSV file to extract keypoint coordinates and visibility.

    Args:
        csv_path: Path to DLC CSV file
        video_width: Video frame width
        video_height: Video frame height
        kp_names: List of keypoint names (6 keypoints)
        likelihood_thr: Likelihood threshold for visibility

    Returns:
        coords: (T, 6, 2) normalized keypoint coordinates
        visibility: (T, 6) visibility mask
    """
    try:
        # Try multi-level header (standard DLC format)
        df = pd.read_csv(csv_path, header=[0, 1, 2])
        # Flatten column names
        df.columns = ['_'.join(str(c).strip() for c in col).lower() for col in df.columns.values]
    except:
        # Fallback to single header
        df = pd.read_csv(csv_path)
        df.columns = [str(c).lower().strip() for c in df.columns]

    num_frames = len(df)
    coords = np.zeros((num_frames, 6, 2), dtype=np.float32)
    likelihoods = np.zeros((num_frames, 6), dtype=np.float32)

    # Extract each keypoint
    for i, kp_name in enumerate(kp_names):
        kp_norm = normalize_keypoint_name(kp_name)

        # Try multiple search patterns
        search_patterns = [
            kp_norm,
            kp_norm.replace('_', ''),
            kp_norm.replace('_', ' '),
            kp_name,  # Original name
        ]

        found = False
        for pattern in search_patterns:
            # Find x, y, likelihood columns
            x_cols = [c for c in df.columns if pattern in c and ('_x' in c or c.endswith('x'))]
            y_cols = [c for c in df.columns if pattern in c and ('_y' in c or c.endswith('y'))]
            like_cols = [c for c in df.columns if pattern in c and ('likelihood' in c or 'confidence' in c)]

            if x_cols and y_cols:
                # Normalize coordinates by video dimensions
                coords[:, i, 0] = pd.to_numeric(df[x_cols[0]], errors='coerce').fillna(0).values / video_width
                coords[:, i, 1] = pd.to_numeric(df[y_cols[0]], errors='coerce').fillna(0).values / video_height

                if like_cols:
                    likelihoods[:, i] = pd.to_numeric(df[like_cols[0]], errors='coerce').fillna(0).values
                else:
                    likelihoods[:, i] = 1.0  # Assume visible if no likelihood column

                found = True
                break

        if not found:
            logger.warning(f"Keypoint '{kp_name}' not found in DLC CSV: {csv_path}")
            likelihoods[:, i] = 0.0  # Mark as invisible

    # Apply likelihood threshold
    visibility = (likelihoods >= likelihood_thr).astype(np.float32)

    return coords, visibility


def find_dlc_csv(video_path: Path) -> Optional[Path]:
    """
    Find matching DLC CSV file for a video.

    Pattern: {video_stem}DLC_resnet50_pawtracking_*.csv
    """
    video_dir = video_path.parent
    video_stem = video_path.stem

    # Search for DLC CSV in same directory
    dlc_pattern = f"{video_stem}DLC_resnet50_*.csv"
    matches = list(video_dir.glob(dlc_pattern))

    if matches:
        return matches[0]

    # Fallback: any CSV containing video stem (but not prediction files)
    fallback_matches = [
        p for p in video_dir.glob(f"*{video_stem}*.csv")
        if 'DLC' in p.name and 'prediction' not in p.name.lower()
    ]

    if fallback_matches:
        return fallback_matches[0]

    return None


# ============================================================================
# VIDEO LOADING
# ============================================================================

def load_video_frames(video_path: Path, img_size: int = 224) -> np.ndarray:
    """
    Load all frames from a video file.

    Args:
        video_path: Path to video file
        img_size: Target image size (width and height)

    Returns:
        frames: (T, 3, H, W) normalized frames in [0, 1]
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to target size
        frame = cv2.resize(frame, (img_size, img_size))

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")

    # Stack and normalize
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)
    frames = frames.astype(np.float32) / 255.0  # Normalize to [0, 1]
    frames = frames.transpose(0, 3, 1, 2)  # (T, 3, H, W)

    return frames


def get_video_dimensions(video_path: Path) -> Tuple[int, int]:
    """Get video width and height."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return width, height


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network with dilated convolutions."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, num_blocks: int = 3, dropout: float = 0.3):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.projections = nn.ModuleList()  # For residual connections when dims don't match
        dilations = [2 ** i for i in range(num_blocks)]  # [1, 2, 4]

        for i, dilation in enumerate(dilations):
            in_dim = input_dim if i == 0 else hidden_dim

            block = nn.Sequential(
                nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.blocks.append(block)

            # Add projection if dimensions don't match
            if in_dim != hidden_dim:
                self.projections.append(nn.Conv1d(in_dim, hidden_dim, kernel_size=1))
            else:
                self.projections.append(None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) input features
        Returns:
            out: (B, T, hidden_dim) output features
        """
        # Convert to (B, D, T) for Conv1d
        x = x.transpose(1, 2)  # (B, D, T)

        for i, (block, projection) in enumerate(zip(self.blocks, self.projections)):
            identity = x

            # Apply projection to identity if needed
            if projection is not None:
                identity = projection(identity)

            # Apply block and add residual
            x = block(x) + identity

        # Convert back to (B, T, D)
        x = x.transpose(1, 2)  # (B, T, hidden_dim)

        return x


class ActionOnlyModel(nn.Module):
    """
    Action-only model using visual frames + pose graph features.

    Architecture:
    - Backbone: VideoMAE (3D) or ViT (2D with temporal pooling)
    - Pose projection: 18 -> 128 dimensions
    - Temporal head: TCN with dilated convolutions
    - Action head: Frame-wise classification
    """

    def __init__(
        self,
        num_classes: int,
        num_pose_features: int = 18,
        img_size: int = 224,
        freeze_backbone_epochs: int = 3,
        dropout: float = 0.3,
        head_dropout: float = 0.2,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_pose_features = num_pose_features
        self.img_size = img_size
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.current_epoch = 0

        # Try to load VideoMAE (3D) backbone
        self.is_3d = False
        if VideoMAEModel is not None:
            try:
                self.backbone_3d = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
                self.backbone_dim = 768  # VideoMAE base dimension
                self.is_3d = True
                logger.info("[info] Using VideoMAE (3D) backbone")
            except Exception as e:
                logger.warning(f"[warn] Failed to load VideoMAE: {e}")
                self.is_3d = False

        # Fallback to 2D ViT backbone
        if not self.is_3d:
            try:
                # Try DINOv2 first
                self.backbone_2d = timm.create_model(
                    'vit_small_patch14_dinov2.lvd142m',
                    pretrained=True,
                    num_classes=0,  # Remove classification head
                    img_size=img_size
                )
                self.backbone_dim = self.backbone_2d.num_features
                logger.info(f"[info] Using DINOv2 ViT (2D) backbone, dim={self.backbone_dim}")
            except Exception as e:
                logger.warning(f"[warn] Failed to load DINOv2: {e}")
                # Fallback to standard ViT
                self.backbone_2d = timm.create_model(
                    'vit_base_patch16_224.augreg_in21k',
                    pretrained=True,
                    num_classes=0,
                    img_size=img_size
                )
                self.backbone_dim = self.backbone_2d.num_features
                logger.info(f"[info] Using ViT-Base (2D) backbone, dim={self.backbone_dim}")

        # Pose feature projection
        self.pose_proj = nn.Sequential(
            nn.Linear(num_pose_features, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # Temporal head
        combined_dim = self.backbone_dim + 128
        hidden_dim = 512
        self.temporal_head = TemporalConvNet(
            input_dim=combined_dim,
            hidden_dim=hidden_dim,
            num_blocks=3,
            dropout=dropout
        )

        # Action classification head
        self.head_act = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # Freeze backbone initially
        self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        if self.is_3d:
            for param in self.backbone_3d.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone_2d.parameters():
                param.requires_grad = False

    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        if self.is_3d:
            for param in self.backbone_3d.parameters():
                param.requires_grad = True
        else:
            for param in self.backbone_2d.parameters():
                param.requires_grad = True

    def set_epoch(self, epoch: int):
        """Update current epoch and unfreeze backbone if needed."""
        self.current_epoch = epoch
        if epoch >= self.freeze_backbone_epochs:
            self._unfreeze_backbone()

    def forward(self, x: torch.Tensor, pose_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, T, 3, H, W) video frames in [0, 1]
            pose_features: (B, T, 18) pose graph features

        Returns:
            logits_act: (B, T, num_classes) action logits
        """
        B, T, C, H, W = x.shape

        # Extract visual features
        if self.is_3d:
            # VideoMAE expects 16 frames
            if T >= 16:
                # Sample 16 frames uniformly
                indices = torch.linspace(0, T - 1, 16).long().to(x.device)
                x_sampled = x[:, indices]
            else:
                # Pad if less than 16 frames
                padding = 16 - T
                x_sampled = torch.cat([x, x[:, -1:].repeat(1, padding, 1, 1, 1)], dim=1)

            outputs = self.backbone_3d(pixel_values=x_sampled)
            video_feats = outputs.last_hidden_state.mean(dim=1)  # (B, D)
            visual_feats = video_feats.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)
        else:
            # 2D ViT: process each frame independently
            x = x.view(B * T, C, H, W)
            visual_feats = self.backbone_2d(x)  # (B*T, D)
            visual_feats = visual_feats.view(B, T, -1)  # (B, T, D)

        # Project pose features
        pose_feats = self.pose_proj(pose_features)  # (B, T, 128)

        # Concatenate visual and pose features
        combined_feats = torch.cat([visual_feats, pose_feats], dim=-1)  # (B, T, D+128)

        # Temporal modeling
        feats = self.temporal_head(combined_feats)  # (B, T, 512)

        # Action classification
        logits_act = self.head_act(feats)  # (B, T, num_classes)

        return logits_act


# ============================================================================
# GROUND TRUTH LOADING
# ============================================================================

def find_ground_truth_csv(video_path: Path, annotations_dir: Optional[Path]) -> Optional[Path]:
    """
    Find ground truth annotation CSV for a video.

    Args:
        video_path: Path to video file
        annotations_dir: Directory containing annotation CSVs

    Returns:
        Path to annotation CSV or None if not found
    """
    if annotations_dir is None:
        return None

    video_stem = video_path.stem

    # Try multiple naming patterns
    patterns = [
        f"prediction_{video_stem}_*.csv",  # prediction_videoname_trial.csv
        f"{video_stem}.csv",                # videoname.csv
        f"*{video_stem}*.csv"               # any CSV containing video name
    ]

    for pattern in patterns:
        matches = list(annotations_dir.glob(pattern))
        # Filter out DLC files and prediction output files
        matches = [m for m in matches if 'DLC' not in m.name and 'predictions' not in m.parent.name]
        if matches:
            return matches[0]

    return None


def load_ground_truth_labels(csv_path: Path, num_frames: int, class_names: List[str]) -> Optional[np.ndarray]:
    """
    Load ground truth action labels from CSV.

    Args:
        csv_path: Path to annotation CSV
        num_frames: Expected number of frames
        class_names: List of class names

    Returns:
        Array of action indices (num_frames,) or None if failed
    """
    try:
        df = pd.read_csv(csv_path)

        # Check if 'Action' column exists
        if 'Action' not in df.columns:
            logger.warning(f"No 'Action' column in {csv_path}")
            return None

        actions = df['Action'].values

        # Ensure correct length (pad or truncate)
        if len(actions) < num_frames:
            # Pad with last action
            padding = np.full(num_frames - len(actions), actions[-1] if len(actions) > 0 else 0)
            actions = np.concatenate([actions, padding])
        elif len(actions) > num_frames:
            # Truncate
            actions = actions[:num_frames]

        return actions.astype(int)

    except Exception as e:
        logger.warning(f"Failed to load ground truth from {csv_path}: {e}")
        return None


# ============================================================================
# ETHOGRAM VISUALIZATION
# ============================================================================

def plot_ethogram(
    gt_labels: Optional[np.ndarray],
    pred_labels: np.ndarray,
    class_names: List[str],
    output_path: Path,
    video_name: str
):
    """
    Plot behavioral ethogram comparing ground truth and predictions.

    Args:
        gt_labels: Ground truth action indices (num_frames,) or None
        pred_labels: Predicted action indices (num_frames,)
        class_names: List of class names
        output_path: Path to save plot
        video_name: Name of video for title
    """
    num_frames = len(pred_labels)

    # Define colors for each behavior class (RGB normalized to 0-1)
    colors_rgb = [
        [0.12, 0.47, 0.71],  # rest - blue
        [1.00, 0.50, 0.05],  # paw_withdraw - orange
        [0.17, 0.63, 0.17],  # paw_lick - green
        [0.84, 0.15, 0.16],  # paw_shake - red
        [0.58, 0.40, 0.74],  # walk - purple
        [0.55, 0.34, 0.29],  # active - brown
    ]

    # Ensure we have enough colors
    while len(colors_rgb) < len(class_names):
        colors_rgb.append([np.random.random(), np.random.random(), np.random.random()])

    colors_rgb = np.array(colors_rgb)

    # Create color image arrays (much faster than adding patches)
    def labels_to_image(labels):
        """Convert action labels to RGB image."""
        img = np.zeros((1, num_frames, 3))
        for i in range(num_frames):
            action_idx = int(labels[i])
            if 0 <= action_idx < len(class_names):
                img[0, i, :] = colors_rgb[action_idx]
        return img

    # Create figure with two subplots (or one if no GT)
    num_plots = 2 if gt_labels is not None else 1
    fig, axes = plt.subplots(num_plots, 1, figsize=(16, 2 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]  # Make it iterable

    # Plot ground truth if available
    if gt_labels is not None:
        ax = axes[0]
        ax.set_title(f'Ground Truth - {video_name}', fontsize=14, fontweight='bold')

        # Use imshow for fast rendering
        gt_img = labels_to_image(gt_labels)
        ax.imshow(gt_img, aspect='auto', extent=[0, num_frames, 0, 1], interpolation='nearest')

        ax.set_ylabel('Ground Truth', fontsize=12)
        ax.set_yticks([])

        # Calculate accuracy
        accuracy = np.mean(gt_labels == pred_labels) * 100
        ax.text(0.02, 0.95, f'Accuracy: {accuracy:.1f}%',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot predictions
    pred_ax = axes[1] if gt_labels is not None else axes[0]
    pred_ax.set_title(f'Predictions - {video_name}', fontsize=14, fontweight='bold')

    # Use imshow for fast rendering
    pred_img = labels_to_image(pred_labels)
    pred_ax.imshow(pred_img, aspect='auto', extent=[0, num_frames, 0, 1], interpolation='nearest')

    pred_ax.set_ylabel('Predictions', fontsize=12)
    pred_ax.set_yticks([])
    pred_ax.set_xlabel('Frame Number', fontsize=12)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_rgb[i], label=class_names[i])
                       for i in range(min(len(class_names), len(colors_rgb)))]

    pred_ax.legend(handles=legend_elements, loc='upper right', ncol=len(class_names),
                   framealpha=0.9, fontsize=10)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved ethogram to: {output_path}")


# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(
    model: nn.Module,
    video_frames: np.ndarray,
    pose_features: np.ndarray,
    device: torch.device,
    batch_size: int = 1
) -> np.ndarray:
    """
    Run inference on video frames and pose features.

    Args:
        model: Trained ActionOnlyModel
        video_frames: (T, 3, H, W) normalized frames
        pose_features: (T, 18) pose graph features
        device: torch device
        batch_size: Batch size (always 1 for single video)

    Returns:
        probs: (T, num_classes) action probabilities
    """
    model.eval()

    with torch.no_grad():
        # Convert to tensors and add batch dimension
        frames_tensor = torch.from_numpy(video_frames).unsqueeze(0).to(device)  # (1, T, 3, H, W)
        pose_tensor = torch.from_numpy(pose_features).unsqueeze(0).to(device)  # (1, T, 18)

        # Forward pass
        logits = model(frames_tensor, pose_tensor)  # (1, T, num_classes)

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)  # (1, T, num_classes)

        # Remove batch dimension
        probs = probs.squeeze(0).cpu().numpy()  # (T, num_classes)

    return probs


def save_predictions(
    output_path: Path,
    probs: np.ndarray,
    class_names: List[str]
):
    """
    Save predictions to CSV file.

    Args:
        output_path: Path to output CSV
        probs: (T, num_classes) action probabilities
        class_names: List of class names
    """
    num_frames = probs.shape[0]

    # Get predicted class indices and labels
    pred_indices = np.argmax(probs, axis=1)
    pred_labels = [class_names[i] for i in pred_indices]

    # Create DataFrame
    data = {
        'Frame': np.arange(num_frames),
        'Action': pred_labels
    }

    # Add probability columns for each class
    for i, class_name in enumerate(class_names):
        data[f'{class_name}_prob'] = probs[:, i]

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved predictions to: {output_path}")


def process_video(
    video_path: Path,
    output_dir: Path,
    model: nn.Module,
    pose_graph: PoseGraph,
    class_names: List[str],
    kp_names: List[str],
    img_size: int,
    likelihood_thr: float,
    device: torch.device,
    annotations_dir: Optional[Path] = None,
    plot_ethogram_flag: bool = False
):
    """
    Process a single video: load, run inference, save predictions, and optionally plot ethogram.

    Args:
        video_path: Path to video file
        output_dir: Output directory for predictions
        model: Trained model
        pose_graph: PoseGraph instance
        class_names: List of class names
        kp_names: List of keypoint names
        img_size: Image size for model input
        likelihood_thr: Likelihood threshold for DLC keypoints
        device: Torch device
        annotations_dir: Directory containing ground truth annotations (optional)
        plot_ethogram_flag: Whether to generate ethogram plots
    """
    logger.info(f"\nProcessing: {video_path.name}")

    try:
        # Find corresponding DLC CSV
        dlc_csv_path = find_dlc_csv(video_path)
        if dlc_csv_path is None:
            logger.error(f"  [skip] No DLC CSV found for video: {video_path.name}")
            return

        logger.info(f"  Found DLC CSV: {dlc_csv_path.name}")

        # Get video dimensions
        video_width, video_height = get_video_dimensions(video_path)

        # Load video frames
        logger.info("  Loading video frames...")
        video_frames = load_video_frames(video_path, img_size=img_size)
        num_frames = video_frames.shape[0]
        logger.info(f"  Loaded {num_frames} frames")

        # Parse DLC CSV
        logger.info("  Parsing DLC keypoints...")
        coords, visibility = parse_dlc_csv(
            dlc_csv_path,
            video_width,
            video_height,
            kp_names,
            likelihood_thr
        )

        # Ensure DLC has same number of frames as video
        if coords.shape[0] != num_frames:
            logger.warning(f"  [warn] Frame mismatch: video={num_frames}, DLC={coords.shape[0]}")
            # Truncate or pad to match
            min_frames = min(num_frames, coords.shape[0])
            video_frames = video_frames[:min_frames]
            coords = coords[:min_frames]
            visibility = visibility[:min_frames]
            num_frames = min_frames

        # Compute pose graph features
        logger.info("  Computing pose graph features...")
        pose_features = pose_graph.compute_features(coords, visibility)

        # Run inference
        logger.info("  Running inference...")
        probs = run_inference(model, video_frames, pose_features, device)

        # Save predictions
        output_filename = f"{video_path.stem}_predictions.csv"
        output_path = output_dir / output_filename
        save_predictions(output_path, probs, class_names)

        # Generate ethogram if requested
        if plot_ethogram_flag:
            logger.info("  Generating ethogram...")

            # Get predicted labels
            pred_labels = np.argmax(probs, axis=1)

            # Try to load ground truth
            gt_labels = None
            if annotations_dir is not None:
                gt_csv_path = find_ground_truth_csv(video_path, annotations_dir)
                if gt_csv_path is not None:
                    logger.info(f"  Found ground truth: {gt_csv_path.name}")
                    gt_labels = load_ground_truth_labels(gt_csv_path, num_frames, class_names)
                else:
                    logger.info("  No ground truth found, plotting predictions only")

            # Plot ethogram
            ethogram_path = output_dir / f"{video_path.stem}_ethogram.png"
            plot_ethogram(gt_labels, pred_labels, class_names, ethogram_path, video_path.name)

        logger.info(f"  [success] Processed {video_path.name}")

    except Exception as e:
        logger.error(f"  [error] Failed to process {video_path.name}: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Inference script for mouse action recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python inference.py /path/to/video_folder

Expected folder structure:
    video_folder/
        video1.mp4
        video1DLC_resnet50_pawtracking_....csv
        video2.mp4
        video2DLC_resnet50_pawtracking_....csv
        ...

Output:
    predictions_video_folder/
        video1_predictions.csv
        video2_predictions.csv
        ...
        """
    )

    parser.add_argument(
        'input_folder',
        type=str,
        help='Path to folder containing videos and DLC CSV files'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='best_model_multitask.pt',
        help='Path to model checkpoint (default: best_model_multitask.pt in current directory)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for inference (default: auto-detect)'
    )

    parser.add_argument(
        '--ethogram',
        action='store_true',
        help='Generate ethogram plots comparing ground truth and predictions'
    )

    parser.add_argument(
        '--annotations_dir',
        type=str,
        default=None,
        help='Path to folder containing ground truth annotation CSVs (required for ethogram with ground truth)'
    )

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Make sure best_model_multitask.pt is in the current directory")
        sys.exit(1)

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract metadata from checkpoint
    train_args = checkpoint['args']
    class_names = checkpoint['classes']
    num_classes = len(class_names)

    # Handle kp_names - can be either string or list depending on how checkpoint was saved
    kp_names_raw = checkpoint['kp_names']
    if isinstance(kp_names_raw, str):
        kp_names = kp_names_raw.split(',')
    else:
        kp_names = kp_names_raw  # Already a list

    logger.info(f"Model trained with {num_classes} classes: {class_names}")
    logger.info(f"Keypoints: {kp_names}")

    # Detect which backbone was used during training
    state_dict_keys = checkpoint['model_state_dict'].keys()
    uses_3d_backbone = any('backbone_3d' in key for key in state_dict_keys)
    uses_2d_backbone = any('backbone_2d' in key for key in state_dict_keys)

    logger.info(f"Checkpoint uses 3D backbone: {uses_3d_backbone}")
    logger.info(f"Checkpoint uses 2D backbone: {uses_2d_backbone}")

    # Initialize model
    logger.info("Initializing model...")
    model = ActionOnlyModel(
        num_classes=num_classes,
        num_pose_features=18,
        img_size=train_args.get('img_size', 224),
        freeze_backbone_epochs=train_args.get('freeze_backbone_epochs', 3),
        dropout=train_args.get('dropout', 0.3),
        head_dropout=train_args.get('head_dropout', 0.2)
    ).to(device)

    # Verify backbone compatibility
    if uses_3d_backbone and not model.is_3d:
        logger.error("Checkpoint was trained with VideoMAE (3D) but VideoMAE is not available!")
        logger.error("Install transformers: pip install transformers")
        sys.exit(1)

    if uses_2d_backbone and model.is_3d:
        logger.warning("Checkpoint was trained with 2D backbone but 3D backbone initialized")
        logger.warning("This shouldn't happen - reinitializing with 2D backbone")
        # Force 2D backbone by removing 3D
        model.is_3d = False

    # Load model weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info("Model loaded successfully")
    except RuntimeError as e:
        logger.error("Failed to load model weights!")
        logger.error(f"Error: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Ensure the checkpoint matches the model architecture")
        logger.error("2. Check if you need to install transformers for VideoMAE support")
        sys.exit(1)

    # Initialize PoseGraph
    pose_graph = PoseGraph()

    # Setup input/output directories
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        logger.error(f"Input folder not found: {input_folder}")
        sys.exit(1)

    output_folder = input_folder.parent / f"predictions_{input_folder.name}"
    output_folder.mkdir(exist_ok=True)
    logger.info(f"Output folder: {output_folder}")

    # Setup annotations directory for ethogram
    annotations_dir = None
    if args.annotations_dir is not None:
        annotations_dir = Path(args.annotations_dir)
        if not annotations_dir.exists():
            logger.warning(f"Annotations directory not found: {annotations_dir}")
            logger.warning("Ethograms will be generated without ground truth")
            annotations_dir = None
        else:
            logger.info(f"Annotations folder: {annotations_dir}")

    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_folder.glob(f'*{ext}'))
        video_files.extend(input_folder.glob(f'*{ext.upper()}'))

    video_files = sorted(set(video_files))

    if len(video_files) == 0:
        logger.error(f"No video files found in: {input_folder}")
        sys.exit(1)

    logger.info(f"Found {len(video_files)} video(s)")

    # Process each video
    logger.info("\n" + "="*80)
    logger.info("Starting inference")
    logger.info("="*80)

    for video_path in tqdm(video_files, desc="Processing videos"):
        process_video(
            video_path=video_path,
            output_dir=output_folder,
            model=model,
            pose_graph=pose_graph,
            class_names=class_names,
            kp_names=kp_names,
            img_size=train_args.get('img_size', 224),
            likelihood_thr=train_args.get('kp_likelihood_thr', 0.90),
            device=device,
            annotations_dir=annotations_dir,
            plot_ethogram_flag=args.ethogram
        )

    logger.info("\n" + "="*80)
    logger.info("Inference complete!")
    logger.info(f"Results saved to: {output_folder}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
