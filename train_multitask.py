#!/usr/bin/env python3
"""
Multi-Task Training Script for Mouse Bottom-View Action + Keypoint Prediction

Trains a temporal model with:
- Backbone: VideoMAE2 (3D) or ViT (2D with mean pooling)
- Temporal head: TCN (dilated convolutions)
- Pose features: 18 geometric features (8 edge lengths + 10 angles)
- Task heads: frame-wise action classification + keypoint regression

Two model types:
1. Action-only model (--model_type action_only):
   - Predicts actions using visual frames + pose graph features
   - No keypoint prediction head
   - Faster training, focused on action recognition

2. Multi-task model (--model_type multitask, default):
   - Predicts actions using visual frames + pose graph features
   - Also predicts keypoint coordinates as auxiliary task
   - Joint training for better feature learning

Features:
- Pose graph features for geometric relationships (translation/scale/rotation invariant)
- Keypoint coordinate jittering augmentation
- 345-frame annotation support (automatic offset alignment)
- Comprehensive data augmentation
- Rare-class boosting and focal loss
- Early stopping and adaptive LR scheduling
- Handles messy real-world data with robust error handling
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

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

try:
    import wandb
except ImportError:
    print("[warn] wandb not installed. Logging to W&B will not be available.")
    print("[warn] Install with: pip install wandb")
    wandb = None

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

        # Define angle triplets (center keypoint, two endpoints)
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
        Compute 18 pose graph features from keypoint coordinates.

        Args:
            coords: (T, K, 2) normalized keypoint coordinates
            visibility: (T, K) visibility mask

        Returns:
            features: (T, 18) pose graph features
        """
        T, K = coords.shape[0], coords.shape[1]
        features = np.zeros((T, 18), dtype=np.float32)

        for t in range(T):
            # Extract edge lengths (8 features)
            for i, (p1, p2) in enumerate(self.edges):
                if visibility[t, p1] > 0 and visibility[t, p2] > 0:
                    delta = coords[t, p2] - coords[t, p1]
                    length = np.sqrt(np.sum(delta ** 2))
                    features[t, i] = length
                else:
                    features[t, i] = 0.0  # Missing keypoint

            # Extract angles (10 features)
            for i, (center, p1, p2) in enumerate(self.angle_triplets):
                if visibility[t, center] > 0 and visibility[t, p1] > 0 and visibility[t, p2] > 0:
                    # Vectors from center to endpoints
                    v1 = coords[t, p1] - coords[t, center]
                    v2 = coords[t, p2] - coords[t, center]

                    # Compute angle using law of cosines
                    len_v1 = np.sqrt(np.sum(v1 ** 2))
                    len_v2 = np.sqrt(np.sum(v2 ** 2))

                    if len_v1 > 1e-6 and len_v2 > 1e-6:
                        cos_angle = np.dot(v1, v2) / (len_v1 * len_v2)
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle = np.arccos(cos_angle)
                        features[t, 8 + i] = angle  # Radians
                    else:
                        features[t, 8 + i] = 0.0
                else:
                    features[t, 8 + i] = 0.0  # Missing keypoint

        return features


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_merge_mapping(
    orig_classes: List[str],
    merge_idxs: List[int] = [1, 3, 5],
    target_idx: int = 1,
    merged_name: str = "1"
) -> Tuple[List[str], Dict[int, int]]:
    """
    Build class merge mapping.

    Args:
        orig_classes: Original class names (e.g., ['rest', 'paw_withdraw', 'paw_lick', ...])
        merge_idxs: Original indices to merge (e.g., [1, 3, 5])
        target_idx: Target index to merge into (e.g., 1)
        merged_name: Name for merged class (e.g., "1")

    Returns:
        new_classes: List of merged class names
        col_map: Mapping from old index to new index
    """
    # Create new class list
    new_classes = []
    col_map = {}
    new_idx = 0

    for i, cls in enumerate(orig_classes):
        if i in merge_idxs:
            if target_idx not in col_map.values():
                new_classes.append(merged_name)
                col_map[i] = len(new_classes) - 1
            else:
                # Already mapped
                col_map[i] = [v for k, v in col_map.items() if k in merge_idxs][0]
        else:
            new_classes.append(cls)
            col_map[i] = len(new_classes) - 1

    # Deduplicate new_classes
    seen = set()
    dedup_classes = []
    final_col_map = {}
    for i, cls in enumerate(orig_classes):
        if i in merge_idxs:
            if merged_name not in seen:
                dedup_classes.append(merged_name)
                seen.add(merged_name)
            final_col_map[i] = dedup_classes.index(merged_name)
        else:
            if cls not in seen:
                dedup_classes.append(cls)
                seen.add(cls)
            final_col_map[i] = dedup_classes.index(cls)

    return dedup_classes, final_col_map


def apply_merge_to_onehot(y: np.ndarray, col_map: Dict[int, int], new_C: int) -> np.ndarray:
    """
    Apply class merge to one-hot encoded labels.

    Args:
        y: One-hot labels (T, old_C)
        col_map: Mapping from old index to new index
        new_C: Number of new classes

    Returns:
        y_new: Merged one-hot labels (T, new_C)
    """
    T, old_C = y.shape
    y_new = np.zeros((T, new_C), dtype=y.dtype)

    for old_idx in range(old_C):
        if old_idx in col_map:
            new_idx = col_map[old_idx]
            y_new[:, new_idx] += y[:, old_idx]

    # Clip to [0, 1] in case of multiple merges
    y_new = np.clip(y_new, 0, 1)
    return y_new


def normalize_keypoint_name(name: str) -> str:
    """Normalize keypoint names to canonical form."""
    name = name.lower().strip().replace(' ', '_').replace('-', '_')

    # Aliases
    if name in ['snout', 'mouth', 'nose']:
        return 'mouth'  # Use 'mouth' as canonical (matches DLC CSVs)
    elif name in ['rf', 'rfp', 'right_front', 'rightfront', 'right_front_paw', 'r_frontpaw']:
        return 'r_frontpaw'  # Match DLC CSV format
    elif name in ['lf', 'lfp', 'left_front', 'leftfront', 'left_front_paw', 'l_frontpaw']:
        return 'l_frontpaw'  # Match DLC CSV format
    elif name in ['rh', 'rhp', 'right_hind', 'righthind', 'right_back', 'rightback', 'right_hind_paw', 'r_hindpaw']:
        return 'r_hindpaw'  # Match DLC CSV format
    elif name in ['lh', 'lhp', 'left_hind', 'lefthind', 'left_back', 'leftback', 'left_hind_paw', 'l_hindpaw']:
        return 'l_hindpaw'  # Match DLC CSV format
    elif 'tail' in name and 'base' in name:
        return 'tail_base'
    elif 'tail' in name:
        return 'tail_base'

    return name


def parse_dlc_csv(
    csv_path: Path,
    kp_names: List[str],
    video_width: int,
    video_height: int,
    likelihood_thr: float = 0.90
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse DeepLabCut CSV and extract keypoints.

    Args:
        csv_path: Path to DLC CSV
        kp_names: List of canonical keypoint names to extract
        video_width: Video width for normalization
        video_height: Video height for normalization
        likelihood_thr: Likelihood threshold for visibility

    Returns:
        coords: (T, K, 2) normalized coordinates
        visibility: (T, K) boolean mask
    """
    try:
        # Try multi-level header (standard DLC format)
        df = pd.read_csv(csv_path, header=[0, 1, 2])
        # Flatten column names: level1_level2_level3
        df.columns = ['_'.join(str(c).strip() for c in col).lower() for col in df.columns.values]
    except:
        try:
            # Try single-level header
            df = pd.read_csv(csv_path)
            df.columns = [col.lower().strip() for col in df.columns]
        except Exception as e:
            logger.warning(f"Failed to parse DLC CSV {csv_path}: {e}")
            return None, None

    # Find keypoint columns
    T = len(df)
    K = len(kp_names)
    coords = np.zeros((T, K, 2), dtype=np.float32)
    likelihoods = np.zeros((T, K), dtype=np.float32)

    for i, kp_name in enumerate(kp_names):
        # Search for columns matching this keypoint
        found = False

        # Try various patterns
        search_patterns = [
            kp_name,
            kp_name.replace('_', ''),
            kp_name.replace('_', ' '),
        ]

        for pattern in search_patterns:
            # Look for x, y, likelihood columns containing the pattern
            x_cols = [c for c in df.columns if pattern in c and ('_x' in c or c.endswith('_x') or c.endswith('x'))]
            y_cols = [c for c in df.columns if pattern in c and ('_y' in c or c.endswith('_y') or c.endswith('y'))]
            like_cols = [c for c in df.columns if pattern in c and ('likelihood' in c or 'confidence' in c)]

            if x_cols and y_cols:
                x_col = x_cols[0]
                y_col = y_cols[0]
                like_col = like_cols[0] if like_cols else None

                try:
                    coords[:, i, 0] = pd.to_numeric(df[x_col], errors='coerce').fillna(0).values / video_width
                    coords[:, i, 1] = pd.to_numeric(df[y_col], errors='coerce').fillna(0).values / video_height

                    if like_col:
                        likelihoods[:, i] = pd.to_numeric(df[like_col], errors='coerce').fillna(0).values
                    else:
                        likelihoods[:, i] = 1.0  # Assume visible if no likelihood

                    found = True
                    break
                except Exception as e:
                    logger.warning(f"Error parsing keypoint '{kp_name}' from {csv_path}: {e}")
                    continue

        if not found:
            logger.warning(f"Keypoint '{kp_name}' not found in DLC CSV {csv_path}")
            coords[:, i, :] = 0.0
            likelihoods[:, i] = 0.0

    # Create visibility mask
    visibility = (likelihoods >= likelihood_thr).astype(np.float32)

    return coords, visibility


def find_dlc_csv(video_path: Path, annotations_dir: Path) -> Optional[Path]:
    """
    Find DLC CSV for a video.

    Search order:
    1. <video_stem>.csv in same directory as video
    2. Any CSV starting with video_stem in same directory (e.g., <stem>DLC_resnet50_...csv)
    3. Any non-action CSV in video directory containing video stem
    """
    video_stem = video_path.stem

    # Try exact match in same directory
    dlc_path = video_path.parent / f"{video_stem}.csv"
    if dlc_path.exists():
        return dlc_path

    # Try CSVs starting with video_stem in same directory (e.g., <stem>DLC_resnet50_...csv)
    for csv_path in video_path.parent.glob(f"{video_stem}*.csv"):
        # Skip if it's an action prediction CSV
        if not csv_path.name.startswith("prediction_"):
            return csv_path

    # Try any CSV in same directory containing video stem (but not action CSVs)
    for csv_path in video_path.parent.glob("*.csv"):
        if video_stem in csv_path.name and not csv_path.name.startswith("prediction_"):
            return csv_path

    return None


def detect_stim_frame(action_csv_path: Path, default: int = 60) -> int:
    """
    Detect stimulus frame from action CSV.

    Looks for columns: StimFrame, stim_frame, StimStart, LaserOn, etc.
    """
    try:
        df = pd.read_csv(action_csv_path, nrows=1)
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ['stimframe', 'stim_frame', 'stimstart', 'laseron', 'stim']:
                val = df[col].values[0]
                if isinstance(val, (int, float)) and not np.isnan(val):
                    return int(val)
    except:
        pass

    return default


def compute_segment_f1(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    iou_threshold: float = 0.3
) -> float:
    """
    Compute segment F1 score.

    Segments are consecutive frames with the same label.
    Matching criterion: IoU >= iou_threshold

    Args:
        pred_labels: (T,) predicted class indices
        true_labels: (T,) ground truth class indices
        iou_threshold: IoU threshold for matching

    Returns:
        Segment F1 score
    """
    def labels_to_segments(labels):
        """Convert frame labels to segments [(start, end, class)]."""
        if len(labels) == 0:
            return []

        segments = []
        start = 0
        current_class = labels[0]

        for i in range(1, len(labels)):
            if labels[i] != current_class:
                segments.append((start, i - 1, current_class))
                start = i
                current_class = labels[i]

        # Add final segment
        segments.append((start, len(labels) - 1, current_class))
        return segments

    def segment_iou(seg1, seg2):
        """Compute IoU between two segments (start, end, class)."""
        if seg1[2] != seg2[2]:  # Different classes
            return 0.0

        start1, end1, _ = seg1
        start2, end2, _ = seg2

        intersection = max(0, min(end1, end2) - max(start1, start2) + 1)
        union = max(end1, end2) - min(start1, start2) + 1

        return intersection / union if union > 0 else 0.0

    pred_segments = labels_to_segments(pred_labels)
    true_segments = labels_to_segments(true_labels)

    if len(pred_segments) == 0 or len(true_segments) == 0:
        return 0.0

    # Match segments
    matched_true = set()
    matched_pred = set()

    for i, pred_seg in enumerate(pred_segments):
        for j, true_seg in enumerate(true_segments):
            if j in matched_true:
                continue

            iou = segment_iou(pred_seg, true_seg)
            if iou >= iou_threshold:
                matched_true.add(j)
                matched_pred.add(i)
                break

    # Compute F1
    tp = len(matched_true)
    fp = len(pred_segments) - len(matched_pred)
    fn = len(true_segments) - len(matched_true)

    if tp + fp + fn == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class TemporalConvNet(nn.Module):
    """TCN with dilated convolutions."""

    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int = 3, dropout: float = 0.3):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.projections = nn.ModuleList()  # For residual connections when dims don't match
        dilations = [2 ** i for i in range(num_blocks)]  # [1, 2, 4]

        for i, dilation in enumerate(dilations):
            in_dim = input_dim if i == 0 else hidden_dim

            block = nn.Sequential(
                nn.Conv1d(in_dim, hidden_dim,
                         kernel_size=3, padding=dilation, dilation=dilation),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)  # Increased from 0.1
            )
            self.blocks.append(block)

            # Add projection if dimensions don't match
            if in_dim != hidden_dim:
                self.projections.append(nn.Conv1d(in_dim, hidden_dim, kernel_size=1))
            else:
                self.projections.append(None)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) features
        Returns:
            (B, T, D) temporal features
        """
        x = x.transpose(1, 2)  # (B, D, T)

        for i, (block, projection) in enumerate(zip(self.blocks, self.projections)):
            identity = x

            # Apply projection to identity if needed
            if projection is not None:
                identity = projection(identity)

            # Apply block and add residual
            x = block(x) + identity

        x = x.transpose(1, 2)  # (B, T, D)
        return x


class DINOv3_Temporal_MultiTask_VideoMAE2Preferred(nn.Module):
    """
    Multi-task model with VideoMAE2 (3D) or ViT (2D) backbone.

    Uses pose graph features (18 geometric features) concatenated with visual
    features for action prediction, and predicts keypoint coordinates as an
    auxiliary task.

    Outputs:
        - Action logits: (B, T, num_classes)
        - Keypoint coords: (B, T, 2*K) in [0, 1]
    """

    def __init__(
        self,
        num_classes: int,
        num_keypoints: int,
        num_pose_features: int = 18,
        img_size: int = 224,
        freeze_backbone_epochs: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        head_dropout: float = 0.2
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.num_pose_features = num_pose_features
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.current_epoch = 0
        self.dropout = dropout
        self.head_dropout = head_dropout

        # Try VideoMAE (3D) first from Hugging Face
        self.is_3d = False
        self.backbone_3d = None
        self.backbone_2d = None

        if VideoMAEModel is not None:
            try:
                self.backbone_3d = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
                self.is_3d = True
                backbone_dim = 768  # VideoMAE-base hidden size
                logger.info("[info] Using VideoMAE (3D) backbone from Hugging Face")
                logger.info("[info] Model: MCG-NJU/videomae-base-finetuned-kinetics")
            except Exception as e:
                logger.warning(f"[warn] VideoMAE not available ({e}), falling back to 2D path")
                logger.warning("[warn] Falling back to 2D path: collapsing clip_T frames with mean pooling.")
        else:
            logger.warning("[warn] transformers not installed, falling back to 2D path")
            logger.warning("[warn] Falling back to 2D path: collapsing clip_T frames with mean pooling.")

            # Try 2D backbones in order
            for model_name in [
                "vit_small_patch14_dinov2.lvd142m",
                "vit_base_patch14_dinov2.lvd142m",
                "vit_small_patch16_224.augreg_in21k",
                "vit_base_patch16_224.augreg_in21k"
            ]:
                try:
                    self.backbone_2d = timm.create_model(
                        model_name,
                        pretrained=True,
                        num_classes=0,
                        img_size=img_size,
                        dynamic_img_size=True  # Allow flexible input sizes
                    )
                    backbone_dim = self.backbone_2d.num_features
                    logger.info(f"[info] Using 2D backbone: {model_name}")
                    break
                except Exception as e:
                    logger.warning(f"[warn] Failed to load {model_name}: {e}")
                    continue

            if self.backbone_2d is None:
                raise RuntimeError("No suitable backbone found!")

        # Freeze backbone initially
        self._freeze_backbone()

        # Pose feature projection (project 18 features to 128)
        self.pose_proj = nn.Sequential(
            nn.Linear(num_pose_features, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # Temporal head (TCN) - input is visual features + projected pose features
        self.temporal_head = TemporalConvNet(
            input_dim=backbone_dim + 128,  # Visual + pose features
            hidden_dim=hidden_dim,
            num_blocks=3,
            dropout=dropout
        )

        # Task heads with dropout
        self.head_act = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.head_kp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim // 2, 2 * num_keypoints),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        if self.backbone_3d is not None:
            for param in self.backbone_3d.parameters():
                param.requires_grad = False
        if self.backbone_2d is not None:
            for param in self.backbone_2d.parameters():
                param.requires_grad = False

    def _unfreeze_backbone(self):
        """Unfreeze last N blocks of backbone."""
        if self.backbone_3d is not None:
            # Unfreeze last 4 blocks
            for name, param in self.backbone_3d.named_parameters():
                if 'blocks.11' in name or 'blocks.10' in name or 'blocks.9' in name or 'blocks.8' in name:
                    param.requires_grad = True
        if self.backbone_2d is not None:
            # Unfreeze last 4 blocks
            for name, param in self.backbone_2d.named_parameters():
                if 'blocks.11' in name or 'blocks.10' in name or 'blocks.9' in name or 'blocks.8' in name:
                    param.requires_grad = True
                elif 'blocks.7' in name or 'blocks.6' in name or 'blocks.5' in name or 'blocks.4' in name:
                    param.requires_grad = True

    def set_epoch(self, epoch: int):
        """Update epoch for backbone unfreezing."""
        self.current_epoch = epoch
        if epoch >= self.freeze_backbone_epochs:
            self._unfreeze_backbone()

    def forward(self, x, pose_features, time_feats=None):
        """
        Args:
            x: (B, T, 3, H, W) video frames
            pose_features: (B, T, 18) pose graph features
            time_feats: (B, T, F) optional time features

        Returns:
            logits_act: (B, T, num_classes) action logits
            pred_kp: (B, T, 2*K) keypoint coords
        """
        B, T, C, H, W = x.shape

        # Extract visual features
        if self.is_3d:
            # VideoMAE (Hugging Face): expects exactly 16 frames
            # Sample 16 frames uniformly from our T frames
            num_frames = 16
            if T >= num_frames:
                # Uniformly sample 16 frames
                indices = torch.linspace(0, T - 1, num_frames).long()
                x_sampled = x[:, indices]  # (B, 16, C, H, W)
            else:
                # If we have fewer than 16 frames, repeat the last frame
                x_sampled = x
                while x_sampled.shape[1] < num_frames:
                    x_sampled = torch.cat([x_sampled, x[:, -1:]], dim=1)
                x_sampled = x_sampled[:, :num_frames]

            outputs = self.backbone_3d(pixel_values=x_sampled)
            # outputs.last_hidden_state is (B, num_patches, hidden_dim)
            # Take mean across patches to get (B, hidden_dim)
            video_feats = outputs.last_hidden_state.mean(dim=1)  # (B, D)
            # Expand to all frames
            visual_feats = video_feats.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)
        else:
            # 2D backbone: process each frame
            x = x.view(B * T, C, H, W)
            visual_feats = self.backbone_2d(x)  # (B*T, D)
            visual_feats = visual_feats.view(B, T, -1)  # (B, T, D)

        # Project pose features
        pose_feats = self.pose_proj(pose_features)  # (B, T, 128)

        # Concatenate visual and pose features
        combined_feats = torch.cat([visual_feats, pose_feats], dim=-1)  # (B, T, D+128)

        # Temporal head
        feats = self.temporal_head(combined_feats)  # (B, T, hidden_dim)

        # Task heads
        logits_act = self.head_act(feats)  # (B, T, num_classes)
        pred_kp = self.head_kp(feats)  # (B, T, 2*K)

        return logits_act, pred_kp


class ActionOnlyModel(nn.Module):
    """
    Action-only model using visual frames + pose graph features.

    Uses 18 pose graph features (edge lengths + angles) concatenated with
    visual features for action prediction only.

    Outputs:
        - Action logits: (B, T, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        num_pose_features: int = 18,
        img_size: int = 224,
        freeze_backbone_epochs: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        head_dropout: float = 0.2
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_pose_features = num_pose_features
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.current_epoch = 0
        self.dropout = dropout
        self.head_dropout = head_dropout

        # Try VideoMAE (3D) first from Hugging Face
        self.is_3d = False
        self.backbone_3d = None
        self.backbone_2d = None

        if VideoMAEModel is not None:
            try:
                self.backbone_3d = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
                self.is_3d = True
                backbone_dim = 768  # VideoMAE-base hidden size
                logger.info("[info] Using VideoMAE (3D) backbone for action-only model")
                logger.info("[info] Model: MCG-NJU/videomae-base-finetuned-kinetics")
            except Exception as e:
                logger.warning(f"[warn] VideoMAE not available ({e}), falling back to 2D path")
                logger.warning("[warn] Falling back to 2D path: collapsing clip_T frames with mean pooling.")
        else:
            logger.warning("[warn] transformers not installed, falling back to 2D path")
            logger.warning("[warn] Falling back to 2D path: collapsing clip_T frames with mean pooling.")

            # Try 2D backbones in order
            for model_name in [
                "vit_small_patch14_dinov2.lvd142m",
                "vit_base_patch14_dinov2.lvd142m",
                "vit_small_patch16_224.augreg_in21k",
                "vit_base_patch16_224.augreg_in21k"
            ]:
                try:
                    self.backbone_2d = timm.create_model(
                        model_name,
                        pretrained=True,
                        num_classes=0,
                        img_size=img_size,
                        dynamic_img_size=True
                    )
                    backbone_dim = self.backbone_2d.num_features
                    logger.info(f"[info] Using 2D backbone for action-only model: {model_name}")
                    break
                except Exception as e:
                    logger.warning(f"[warn] Failed to load {model_name}: {e}")
                    continue

            if self.backbone_2d is None:
                raise RuntimeError("No suitable backbone found!")

        # Freeze backbone initially
        self._freeze_backbone()

        # Pose feature projection (project 18 features to match visual feature dim)
        self.pose_proj = nn.Sequential(
            nn.Linear(num_pose_features, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # Temporal head (TCN) - input is visual features + projected pose features
        self.temporal_head = TemporalConvNet(
            input_dim=backbone_dim + 128,  # Visual + pose features
            hidden_dim=hidden_dim,
            num_blocks=3,
            dropout=dropout
        )

        # Action head only (no keypoint head)
        self.head_act = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        if self.backbone_3d is not None:
            for param in self.backbone_3d.parameters():
                param.requires_grad = False
        if self.backbone_2d is not None:
            for param in self.backbone_2d.parameters():
                param.requires_grad = False

    def _unfreeze_backbone(self):
        """Unfreeze last N blocks of backbone."""
        if self.backbone_3d is not None:
            for name, param in self.backbone_3d.named_parameters():
                if 'blocks.11' in name or 'blocks.10' in name or 'blocks.9' in name or 'blocks.8' in name:
                    param.requires_grad = True
        if self.backbone_2d is not None:
            for name, param in self.backbone_2d.named_parameters():
                if 'blocks.11' in name or 'blocks.10' in name or 'blocks.9' in name or 'blocks.8' in name:
                    param.requires_grad = True

    def set_epoch(self, epoch: int):
        """Update current epoch and unfreeze backbone if needed."""
        self.current_epoch = epoch
        if epoch >= self.freeze_backbone_epochs:
            self._unfreeze_backbone()

    def forward(self, x, pose_features, time_feats=None):
        """
        Args:
            x: (B, T, 3, H, W) video frames
            pose_features: (B, T, 18) pose graph features
            time_feats: (B, T, F) optional time features

        Returns:
            logits_act: (B, T, num_classes) action logits
        """
        B, T, C, H, W = x.shape

        # Extract visual features
        if self.is_3d:
            # VideoMAE (Hugging Face): expects exactly 16 frames
            # Sample 16 frames uniformly from our T frames
            num_frames = 16
            if T >= num_frames:
                # Uniformly sample 16 frames
                indices = torch.linspace(0, T - 1, num_frames).long()
                x_sampled = x[:, indices]  # (B, 16, C, H, W)
            else:
                # If we have fewer than 16 frames, repeat the last frame
                x_sampled = x
                while x_sampled.shape[1] < num_frames:
                    x_sampled = torch.cat([x_sampled, x[:, -1:]], dim=1)
                x_sampled = x_sampled[:, :num_frames]

            outputs = self.backbone_3d(pixel_values=x_sampled)
            # outputs.last_hidden_state is (B, num_patches, hidden_dim)
            # Take mean across patches to get (B, hidden_dim)
            video_feats = outputs.last_hidden_state.mean(dim=1)  # (B, D)
            # Expand to all frames
            visual_feats = video_feats.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)
        else:
            # 2D backbone: process each frame
            x = x.view(B * T, C, H, W)
            visual_feats = self.backbone_2d(x)  # (B*T, D)
            visual_feats = visual_feats.view(B, T, -1)  # (B, T, D)

        # Project pose features
        pose_feats = self.pose_proj(pose_features)  # (B, T, 128)

        # Concatenate visual and pose features
        combined_feats = torch.cat([visual_feats, pose_feats], dim=-1)  # (B, T, D+128)

        # Temporal head
        feats = self.temporal_head(combined_feats)  # (B, T, hidden_dim)

        # Action head only
        logits_act = self.head_act(feats)  # (B, T, num_classes)

        return logits_act


# ============================================================================
# DATASET
# ============================================================================

class MiceActionDatasetFromAnnotations(Dataset):
    """
    Multi-task dataset for action classification and keypoint prediction.

    Handles:
    - Multiple trials per video
    - Missing/invalid data
    - Class merging
    - Rare-class boosting
    """

    # Predefined action class mapping (matches existing data_loader.py)
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

    def __init__(
        self,
        annotations_dir: str,
        videos_dir: str,
        kp_names: List[str],
        split: str = 'train',
        train_T: int = 180,
        val_T: int = 240,
        img_size: int = 224,
        kp_likelihood_thr: float = 0.90,
        rare_threshold_share: float = 0.02,
        rare_boost_cap: float = 12.0,
        merge_idxs: List[int] = [1, 3, 5],
        min_frames: int = 345,  # Changed from 360 to support 345-frame annotations
        split_ratio: float = 0.9,  # 90/10 train/val split
        seed: int = 123,
        kp_jitter_std: float = 0.0,  # Keypoint jittering std (0 = disabled)
        aug_brightness: float = 0.2,  # Brightness jitter
        aug_contrast: float = 0.2,  # Contrast jitter
        aug_temporal_drop: float = 0.1,  # Frame dropout probability
        aug_hflip: float = 0.0  # Horizontal flip probability (disabled by default for left/right keypoints)
    ):
        super().__init__()

        self.annotations_dir = Path(annotations_dir)
        self.videos_dir = Path(videos_dir)
        self.kp_names = kp_names
        self.split = split
        self.window_T = train_T if split == 'train' else val_T
        self.img_size = img_size
        self.kp_likelihood_thr = kp_likelihood_thr
        self.rare_threshold_share = rare_threshold_share
        self.rare_boost_cap = rare_boost_cap
        self.merge_idxs = merge_idxs
        self.min_frames = min_frames
        self.kp_jitter_std = kp_jitter_std
        self.aug_brightness = aug_brightness
        self.aug_contrast = aug_contrast
        self.aug_temporal_drop = aug_temporal_drop
        self.aug_hflip = aug_hflip

        # Initialize pose graph computer
        self.pose_graph = PoseGraph()

        # Use predefined classes (numeric indices 0-7)
        self.orig_classes = [self.ACTION_CLASSES[i] for i in sorted(self.ACTION_CLASSES.keys())]
        logger.info(f"[info] Original classes: {self.orig_classes}")

        # Build merge mapping
        self.merged_classes, self.col_map = build_merge_mapping(
            self.orig_classes,
            merge_idxs=self.merge_idxs,
            target_idx=self.merge_idxs[0],
            merged_name=self.orig_classes[self.merge_idxs[0]]
        )
        logger.info(f"[info] Merged classes: {self.merged_classes}")
        logger.info(f"[info] Column mapping: {self.col_map}")

        # Discover videos and action CSVs
        logger.info(f"[info] Discovering video-trial pairs (video + action CSV + DLC CSV)...")
        self.samples = self._discover_samples()

        # Split train/val
        random.seed(seed)
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * split_ratio)
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

        logger.info(f"[info] {split} samples before boosting: {len(self.samples)}")

        # Print class distribution before boosting
        if split == 'train':
            self._print_class_distribution("BEFORE boosting")

        # Apply rare-class boosting (train only)
        if split == 'train' and rare_boost_cap > 1.0:
            self._apply_rare_class_boosting()
            self._print_class_distribution("AFTER boosting")

        logger.info(f"[info] {split} samples after boosting: {len(self.samples)}")

    def _discover_samples(self) -> List[Dict[str, Any]]:
        """Discover (video, action_csv, dlc_csv) tuples."""
        samples = []

        # Enumerate videos
        video_files = list(self.videos_dir.glob("*.mp4"))
        video_files = [v for v in video_files if not v.name.startswith("._")]  # Skip junk

        for video_path in tqdm(video_files, desc="Discovering video-trial pairs"):
            # Try to open video
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    logger.warning(f"[warn] Cannot open video: {video_path}")
                    continue

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                if frame_count < self.min_frames:
                    logger.warning(f"[warn] Video too short ({frame_count} frames): {video_path}")
                    continue

            except Exception as e:
                logger.warning(f"[warn] Failed to read video {video_path}: {e}")
                continue

            # Find action CSVs for this video
            video_name = video_path.name
            action_csvs = list(self.annotations_dir.glob(f"prediction_{video_name}_*.csv"))

            # Find DLC CSV
            dlc_csv = find_dlc_csv(video_path, self.annotations_dir)

            # If no action CSVs, create a keypoint-only sample
            if len(action_csvs) == 0:
                samples.append({
                    'video_path': video_path,
                    'action_csv': None,
                    'dlc_csv': dlc_csv,
                    'video_width': video_width,
                    'video_height': video_height,
                    'frame_count': frame_count,
                    'action_length': 360,  # Assume full video
                    'frame_offset': 0,  # No offset for keypoint-only
                    'stim_frame': 60,  # Default
                    'has_actions': False
                })
            else:
                # Create one sample per action CSV (trial)
                for action_csv in action_csvs:
                    # Validate action CSV
                    try:
                        df_action = pd.read_csv(action_csv)
                        action_length = len(df_action)

                        if action_length < self.min_frames:
                            logger.warning(f"[warn] Action CSV too short ({action_length} rows): {action_csv}")
                            continue

                        if 'Action' not in df_action.columns:
                            logger.warning(f"[warn] No 'Action' column in {action_csv}")
                            continue

                        # Determine frame offset based on annotation length
                        # 345 frames: annotations start at frame 15 (15-frame context window)
                        # 360 frames: annotations start at frame 0
                        if action_length == 345:
                            frame_offset = 15  # Use video frames [15:360]
                        else:
                            frame_offset = 0  # Use video frames [0:action_length]

                        # Detect stim frame (relative to annotation start)
                        stim_frame = detect_stim_frame(action_csv, default=60)

                        samples.append({
                            'video_path': video_path,
                            'action_csv': action_csv,
                            'dlc_csv': dlc_csv,
                            'video_width': video_width,
                            'video_height': video_height,
                            'frame_count': frame_count,
                            'action_length': action_length,
                            'frame_offset': frame_offset,  # Where to start in video
                            'stim_frame': stim_frame,
                            'has_actions': True
                        })

                    except Exception as e:
                        logger.warning(f"[warn] Failed to parse action CSV {action_csv}: {e}")
                        continue

        return samples

    def _print_class_distribution(self, title: str):
        """Print class distribution table."""
        class_counts = Counter()

        for sample in self.samples:
            if sample['action_csv'] is not None:
                try:
                    df = pd.read_csv(sample['action_csv'])
                    for action_idx in df['Action'].values:
                        # action_idx is a numeric index (0-7)
                        if isinstance(action_idx, (int, np.integer)) and action_idx < len(self.orig_classes):
                            merged_idx = self.col_map[action_idx]
                            merged_class = self.merged_classes[merged_idx]
                            class_counts[merged_class] += 1
                except:
                    pass

        logger.info(f"\n{'='*60}")
        logger.info(f"Class Distribution {title}")
        logger.info(f"{'='*60}")
        logger.info(f"{'Class':<20} {'Count':>10} {'Share':>10}")
        logger.info(f"{'-'*60}")

        total = sum(class_counts.values())
        for cls in self.merged_classes:
            count = class_counts.get(cls, 0)
            share = count / total if total > 0 else 0.0
            logger.info(f"{cls:<20} {count:>10} {share:>10.2%}")

        logger.info(f"{'-'*60}")
        logger.info(f"{'TOTAL':<20} {total:>10} {1.0:>10.2%}")
        logger.info(f"{'='*60}\n")

    def _apply_rare_class_boosting(self):
        """Upsample rare classes by replicating samples."""
        # Compute class frequencies
        class_counts = Counter()
        sample_classes = []

        for sample in self.samples:
            if sample['action_csv'] is not None:
                try:
                    df = pd.read_csv(sample['action_csv'])
                    # Determine dominant class in this sample
                    sample_class_counts = Counter()
                    for action_idx in df['Action'].values:
                        # action_idx is a numeric index (0-7)
                        if isinstance(action_idx, (int, np.integer)) and action_idx < len(self.orig_classes):
                            merged_idx = self.col_map[action_idx]
                            merged_class = self.merged_classes[merged_idx]
                            sample_class_counts[merged_class] += 1

                    # Most common class in this sample
                    if sample_class_counts:
                        dominant_class = sample_class_counts.most_common(1)[0][0]
                        sample_classes.append(dominant_class)
                        class_counts[dominant_class] += 1
                    else:
                        sample_classes.append(None)
                except:
                    sample_classes.append(None)
            else:
                sample_classes.append(None)

        # Identify rare classes
        total = sum(class_counts.values())
        rare_classes = []
        for cls, count in class_counts.items():
            share = count / total if total > 0 else 0.0
            if share < self.rare_threshold_share:
                rare_classes.append(cls)

        logger.info(f"[info] Rare classes (< {self.rare_threshold_share:.1%}): {rare_classes}")

        if not rare_classes:
            return

        # Replicate rare samples
        new_samples = []
        replication_counts = Counter()

        for sample, sample_class in zip(self.samples, sample_classes):
            new_samples.append(sample)

            if sample_class in rare_classes:
                # Compute replication factor
                class_share = class_counts[sample_class] / total
                target_share = self.rare_threshold_share
                replication_factor = min(target_share / class_share, self.rare_boost_cap)

                # Replicate
                num_replicas = int(replication_factor) - 1
                for _ in range(num_replicas):
                    new_samples.append(sample)
                    replication_counts[sample_class] += 1

        self.samples = new_samples

        logger.info(f"[info] Rare class replication totals:")
        for cls, count in replication_counts.items():
            logger.info(f"  {cls}: +{count} replicas")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load video window (with frame offset for 345-frame annotations)
        video_frames = self._load_video_window(
            sample['video_path'],
            sample['stim_frame'],
            sample['frame_count'],
            sample['frame_offset'],
            sample['action_length']
        )  # (T, 3, H, W)

        # Load actions
        if sample['has_actions']:
            actions, action_mask = self._load_actions(
                sample['action_csv'],
                sample['action_length']
            )  # (T, C), scalar
        else:
            actions = np.zeros((self.window_T, len(self.merged_classes)), dtype=np.float32)
            action_mask = 0.0

        # Load keypoints as (T, K, 2) for pose graph computation
        keypoints_2d, visibility = self._load_keypoints(
            sample['dlc_csv'],
            sample['video_width'],
            sample['video_height'],
            sample['frame_offset'],
            sample['action_length']
        )  # (T, K, 2), (T, K)

        # Apply keypoint jittering (train only, before pose graph computation)
        if self.split == 'train' and self.kp_jitter_std > 0:
            keypoints_2d = self._apply_keypoint_jitter(keypoints_2d, visibility)

        # Compute pose graph features (18 features) from jittered keypoints
        pose_features = self.pose_graph.compute_features(keypoints_2d, visibility)  # (T, 18)

        # Flatten keypoints for coordinate prediction head (T, 2K)
        keypoints_flat = keypoints_2d.reshape(self.window_T, -1)

        # Compute time features
        time_feats = self._compute_time_features(sample['stim_frame'])  # (T, 2)

        # Apply augmentations (ALL training samples, not just rare)
        # NOTE: We don't augment pose_features here since they're derived from keypoints
        if self.split == 'train':
            video_frames, keypoints_flat, actions = self._apply_augmentations(
                video_frames, keypoints_flat, actions, action_mask > 0
            )

        return {
            'video': torch.from_numpy(video_frames).float(),  # (T, 3, H, W)
            'actions': torch.from_numpy(actions).float(),  # (T, C)
            'action_mask': torch.tensor(action_mask, dtype=torch.float32),  # scalar
            'pose_features': torch.from_numpy(pose_features).float(),  # (T, 18) - NEW!
            'keypoints': torch.from_numpy(keypoints_flat).float(),  # (T, 2K) - for coordinate head
            'visibility': torch.from_numpy(visibility).float(),  # (T, K)
            'time_feats': torch.from_numpy(time_feats).float(),  # (T, 2)
        }

    def _load_video_window(
        self,
        video_path: Path,
        stim_frame: int,
        frame_count: int,
        frame_offset: int,
        action_length: int
    ) -> np.ndarray:
        """
        Load video window centered near stimulus.

        For 345-frame annotations: frame_offset=15, use video frames [15:360]
        For 360-frame annotations: frame_offset=0, use video frames [0:360]
        """
        # Compute available frame range (after offset)
        available_start = frame_offset
        available_end = min(frame_offset + action_length, frame_count)
        available_length = available_end - available_start

        # Compute window start (relative to available range)
        center_frame = stim_frame + self.window_T // 4  # Slightly after stim
        start_frame_rel = max(0, center_frame - self.window_T // 2)
        end_frame_rel = start_frame_rel + self.window_T

        # Adjust if exceeds available length
        if end_frame_rel > available_length:
            end_frame_rel = available_length
            start_frame_rel = max(0, end_frame_rel - self.window_T)

        # Convert to absolute video frames
        start_frame = available_start + start_frame_rel
        end_frame = available_start + end_frame_rel

        # Load frames
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(self.window_T):
            ret, frame = cap.read()
            if not ret:
                # Pad with last frame if video ends early
                if len(frames) > 0:
                    frame = frames[-1]
                else:
                    frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

            # Resize and normalize
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        # Stack and normalize
        frames = np.stack(frames, axis=0)  # (T, H, W, 3)
        frames = frames.astype(np.float32) / 255.0  # [0, 1]
        frames = frames.transpose(0, 3, 1, 2)  # (T, 3, H, W)

        return frames

    def _load_actions(self, action_csv: Path, action_length: int) -> Tuple[np.ndarray, float]:
        """Load and merge action labels."""
        df = pd.read_csv(action_csv)

        # Extract window (up to min of window_T or action_length)
        max_frames = min(self.window_T, action_length)
        action_indices = df['Action'].values[:max_frames]

        # Convert to one-hot (original classes)
        T = len(action_indices)
        C_orig = len(self.orig_classes)
        y_onehot = np.zeros((T, C_orig), dtype=np.float32)

        for t, action_idx in enumerate(action_indices):
            # action_idx is a numeric index (0-7)
            if isinstance(action_idx, (int, np.integer)) and 0 <= action_idx < C_orig:
                y_onehot[t, int(action_idx)] = 1.0

        # Apply merge
        y_merged = apply_merge_to_onehot(y_onehot, self.col_map, len(self.merged_classes))

        # Pad if needed
        if T < self.window_T:
            pad = np.zeros((self.window_T - T, len(self.merged_classes)), dtype=np.float32)
            if T > 0:
                pad[:] = y_merged[-1:, :]  # Repeat last frame
            y_merged = np.concatenate([y_merged, pad], axis=0)

        return y_merged, 1.0

    def _load_keypoints(
        self,
        dlc_csv: Optional[Path],
        video_width: int,
        video_height: int,
        frame_offset: int,
        action_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load keypoints and visibility with frame offset support.

        Returns:
            coords: (T, K, 2) keypoint coordinates
            visibility: (T, K) visibility mask
        """
        if dlc_csv is None or not dlc_csv.exists():
            logger.warning(f"[warn] DLC CSV not found, using zeros")
            coords = np.zeros((self.window_T, len(self.kp_names), 2), dtype=np.float32)
            visibility = np.zeros((self.window_T, len(self.kp_names)), dtype=np.float32)
        else:
            coords, visibility = parse_dlc_csv(
                dlc_csv,
                self.kp_names,
                video_width,
                video_height,
                self.kp_likelihood_thr
            )

            if coords is None:
                coords = np.zeros((self.window_T, len(self.kp_names), 2), dtype=np.float32)
                visibility = np.zeros((self.window_T, len(self.kp_names)), dtype=np.float32)
            else:
                # Apply frame offset (for 345-frame annotations, skip first 15 frames)
                max_len = min(action_length, len(coords))
                coords = coords[frame_offset:frame_offset + max_len]
                visibility = visibility[frame_offset:frame_offset + max_len]

                # Pad/trim to window_T
                if len(coords) < self.window_T:
                    pad_len = self.window_T - len(coords)
                    if len(coords) > 0:
                        coords = np.concatenate([coords, np.tile(coords[-1:], (pad_len, 1, 1))], axis=0)
                        visibility = np.concatenate([visibility, np.tile(visibility[-1:], (pad_len, 1))], axis=0)
                    else:
                        coords = np.zeros((self.window_T, len(self.kp_names), 2), dtype=np.float32)
                        visibility = np.zeros((self.window_T, len(self.kp_names)), dtype=np.float32)
                elif len(coords) > self.window_T:
                    coords = coords[:self.window_T]
                    visibility = visibility[:self.window_T]

        return coords, visibility  # Return (T, K, 2) not flattened

    def _compute_time_features(self, stim_frame: int) -> np.ndarray:
        """Compute time features: time_since_stim, post_stim."""
        time_feats = np.zeros((self.window_T, 2), dtype=np.float32)

        for t in range(self.window_T):
            time_since_stim = max(0, t - stim_frame) / self.window_T
            post_stim = 1.0 if t >= stim_frame else 0.0

            time_feats[t, 0] = time_since_stim
            time_feats[t, 1] = post_stim

        return time_feats

    def _apply_keypoint_jitter(
        self,
        keypoints: np.ndarray,
        visibility: np.ndarray
    ) -> np.ndarray:
        """
        Apply Gaussian jittering to keypoint coordinates.

        Args:
            keypoints: (T, K, 2) normalized coordinates
            visibility: (T, K) visibility mask (not modified)

        Returns:
            Jittered keypoints (T, K, 2)
        """
        if self.kp_jitter_std == 0:
            return keypoints

        # Add Gaussian noise to coordinates
        noise = np.random.randn(*keypoints.shape) * self.kp_jitter_std
        keypoints_jittered = keypoints + noise

        # Clip to [0, 1] range
        keypoints_jittered = np.clip(keypoints_jittered, 0, 1)

        return keypoints_jittered

    def _apply_augmentations(
        self,
        video_frames: np.ndarray,
        keypoints: np.ndarray,
        actions: np.ndarray,
        has_actions: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply comprehensive data augmentation (all training samples).

        Args:
            video_frames: (T, 3, H, W)
            keypoints: (T, 2K)
            actions: (T, C)
            has_actions: Whether this sample has action labels

        Returns:
            Augmented video_frames, keypoints, actions
        """
        # Brightness jitter
        if self.aug_brightness > 0 and random.random() < 0.5:
            brightness = random.uniform(1 - self.aug_brightness, 1 + self.aug_brightness)
            video_frames = np.clip(video_frames * brightness, 0, 1)

        # Contrast jitter
        if self.aug_contrast > 0 and random.random() < 0.5:
            contrast = random.uniform(1 - self.aug_contrast, 1 + self.aug_contrast)
            mean = video_frames.mean()
            video_frames = np.clip((video_frames - mean) * contrast + mean, 0, 1)

        # Temporal frame dropout (randomly drop frames and interpolate)
        if self.aug_temporal_drop > 0 and random.random() < 0.3:
            T = len(video_frames)
            num_drop = int(T * self.aug_temporal_drop)
            if num_drop > 0 and num_drop < T - 2:
                # Randomly select frames to drop
                drop_indices = np.random.choice(range(1, T - 1), num_drop, replace=False)
                for idx in drop_indices:
                    # Replace with previous frame
                    video_frames[idx] = video_frames[idx - 1]
                    # Don't modify keypoints/actions (just visual dropout)

        # Gaussian noise
        if random.random() < 0.3:
            noise = np.random.randn(*video_frames.shape) * 0.02
            video_frames = np.clip(video_frames + noise, 0, 1)

        # Temporal roll (circular shift) - careful with actions!
        if random.random() < 0.2:
            shift = random.randint(-5, 5)
            video_frames = np.roll(video_frames, shift, axis=0)
            keypoints = np.roll(keypoints, shift, axis=0)
            if has_actions:
                actions = np.roll(actions, shift, axis=0)

        return video_frames, keypoints, actions


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 1.5,
    label_smoothing: float = 0.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Focal loss for multi-class classification with label smoothing.

    Args:
        logits: (B, T, C) raw logits
        targets: (B, T, C) one-hot targets
        class_weights: (C,) class weights
        gamma: Focal loss gamma
        label_smoothing: Label smoothing factor (0.0 = no smoothing)
        reduction: 'mean' or 'sum'

    Returns:
        loss: scalar
    """
    # Apply label smoothing
    if label_smoothing > 0:
        C = targets.shape[-1]
        targets = targets * (1 - label_smoothing) + label_smoothing / C

    # Apply softmax
    probs = F.softmax(logits, dim=-1)  # (B, T, C)

    # Compute p_t
    p_t = (probs * targets).sum(dim=-1)  # (B, T)

    # Focal term
    focal_term = (1 - p_t) ** gamma

    # Cross-entropy term (numerically stable)
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, C)
    ce = -(log_probs * targets).sum(dim=-1)  # (B, T)

    # Apply class weights
    weights = (targets * class_weights.view(1, 1, -1)).sum(dim=-1)  # (B, T)

    # Combine
    loss = focal_term * ce * weights

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def temporal_smoothness_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Temporal smoothness loss on logit deltas.

    Args:
        logits: (B, T, C) raw logits

    Returns:
        loss: scalar
    """
    deltas = logits[:, 1:, :] - logits[:, :-1, :]  # (B, T-1, C)
    loss = torch.abs(deltas).mean()
    return loss


def masked_keypoint_loss(
    pred_kp: torch.Tensor,
    target_kp: torch.Tensor,
    visibility: torch.Tensor
) -> torch.Tensor:
    """
    Masked SmoothL1 loss for keypoints.

    Args:
        pred_kp: (B, T, 2K) predicted keypoints
        target_kp: (B, T, 2K) target keypoints
        visibility: (B, T, K) visibility mask

    Returns:
        loss: scalar
    """
    B, T, two_K = pred_kp.shape
    K = two_K // 2

    # Reshape for easier masking
    pred_kp = pred_kp.view(B, T, K, 2)  # (B, T, K, 2)
    target_kp = target_kp.view(B, T, K, 2)

    # Expand visibility to (B, T, K, 2)
    visibility = visibility.unsqueeze(-1).expand(-1, -1, -1, 2)

    # Compute loss only on visible keypoints
    diff = F.smooth_l1_loss(pred_kp, target_kp, reduction='none')  # (B, T, K, 2)
    masked_diff = diff * visibility

    # Average over visible elements
    num_visible = visibility.sum() + 1e-6
    loss = masked_diff.sum() / num_visible

    return loss


# ============================================================================
# METRICS
# ============================================================================

def compute_frame_metrics(
    all_logits: np.ndarray,
    all_targets: np.ndarray,
    class_names: List[str]
) -> Dict[str, float]:
    """
    Compute frame-wise metrics with per-class threshold optimization.

    Args:
        all_logits: (N, C) logits
        all_targets: (N, C) one-hot targets

    Returns:
        metrics: dict with macro_f1, macro_acc, best_thresholds
    """
    N, C = all_logits.shape

    # Convert to probabilities
    probs = torch.softmax(torch.from_numpy(all_logits), dim=-1).numpy()

    # For each class, find best threshold
    best_thresholds = []
    class_f1s = []

    for c in range(C):
        class_probs = probs[:, c]
        class_targets = all_targets[:, c]

        # Skip if no positive examples
        if class_targets.sum() == 0:
            best_thresholds.append(0.5)
            class_f1s.append(0.0)
            continue

        # Search for best threshold
        best_thr = 0.5
        best_f1 = 0.0

        for thr in np.linspace(0.1, 0.9, 17):
            preds = (class_probs >= thr).astype(int)
            tp = ((preds == 1) & (class_targets == 1)).sum()
            fp = ((preds == 1) & (class_targets == 0)).sum()
            fn = ((preds == 0) & (class_targets == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

        best_thresholds.append(best_thr)
        class_f1s.append(best_f1)

    # Macro F1
    macro_f1 = np.mean(class_f1s)

    # Macro accuracy (with best thresholds)
    preds_multi = np.zeros((N, C), dtype=int)
    for c in range(C):
        preds_multi[:, c] = (probs[:, c] >= best_thresholds[c]).astype(int)

    # Convert to single-label predictions (argmax)
    pred_labels = np.argmax(preds_multi, axis=-1)
    true_labels = np.argmax(all_targets, axis=-1)

    macro_acc = (pred_labels == true_labels).mean()

    return {
        'macro_f1': macro_f1,
        'macro_acc': macro_acc,
        'best_thresholds': best_thresholds,
        'class_f1s': class_f1s
    }


# ============================================================================
# TRAINING & VALIDATION
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    class_weights: torch.Tensor,
    args,
    device: torch.device,
    ema_model=None
):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_action_loss = 0.0
    total_kp_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for batch in pbar:
        video = batch['video'].to(device)  # (B, T, 3, H, W)
        actions = batch['actions'].to(device)  # (B, T, C)
        action_mask = batch['action_mask'].to(device)  # (B,)
        pose_features = batch['pose_features'].to(device)  # (B, T, 18)
        keypoints = batch['keypoints'].to(device)  # (B, T, 2K)
        visibility = batch['visibility'].to(device)  # (B, T, K)
        time_feats = batch['time_feats'].to(device)  # (B, T, 2)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=args.use_amp, dtype=torch.bfloat16 if args.use_amp else torch.float32):
            # Forward - handle both model types
            if args.model_type == 'action_only':
                logits_act = model(video, pose_features, time_feats)
                pred_kp = None
            else:  # multitask
                logits_act, pred_kp = model(video, pose_features, time_feats)

            # Action loss (only on samples with actions)
            if action_mask.sum() > 0:
                action_loss = focal_loss(
                    logits_act[action_mask > 0],
                    actions[action_mask > 0],
                    class_weights,
                    gamma=args.focal_gamma,
                    label_smoothing=args.label_smoothing
                )
                smooth_loss = temporal_smoothness_loss(logits_act[action_mask > 0])
                action_loss = action_loss + args.lambda_smooth * smooth_loss
            else:
                # Ensure gradient tracking even with zero loss
                action_loss = 0.0 * logits_act.sum()

            # Keypoint loss (only for multitask model)
            if args.model_type == 'multitask':
                kp_loss = masked_keypoint_loss(pred_kp, keypoints, visibility)
                loss = action_loss + args.lambda_kp * kp_loss
            else:
                # Ensure gradient tracking even with zero loss
                kp_loss = 0.0 * logits_act.sum()
                loss = action_loss

        # Backward (skip if loss doesn't require grad, e.g., all samples invalid)
        if loss.requires_grad:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Skip optimization step if no gradients
            logger.warning("[warn] Skipping optimization step: loss doesn't require grad")

        # Update EMA
        if ema_model is not None:
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(args.ema_decay).add_(param.data, alpha=1 - args.ema_decay)

        # Track metrics
        total_loss += loss.item()
        total_action_loss += action_loss.item()
        total_kp_loss += kp_loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / num_batches,
            'act_loss': total_action_loss / num_batches,
            'kp_loss': total_kp_loss / num_batches
        })

    return {
        'loss': total_loss / num_batches,
        'action_loss': total_action_loss / num_batches,
        'kp_loss': total_kp_loss / num_batches
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    class_weights: torch.Tensor,
    class_names: List[str],
    args,
    device: torch.device
):
    """Validate model."""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    # Collect predictions for metrics
    all_logits = []
    all_targets = []
    all_pred_labels = []
    all_true_labels = []
    all_pred_kp = []
    all_true_kp = []
    all_visibility = []

    pbar = tqdm(loader, desc="Validation", leave=False)

    for batch in pbar:
        video = batch['video'].to(device)
        actions = batch['actions'].to(device)
        action_mask = batch['action_mask'].to(device)
        pose_features = batch['pose_features'].to(device)
        keypoints = batch['keypoints'].to(device)
        visibility = batch['visibility'].to(device)
        time_feats = batch['time_feats'].to(device)

        with torch.amp.autocast('cuda', enabled=args.use_amp, dtype=torch.bfloat16 if args.use_amp else torch.float32):
            # Forward - handle both model types
            if args.model_type == 'action_only':
                logits_act = model(video, pose_features, time_feats)
                pred_kp = None
            else:  # multitask
                logits_act, pred_kp = model(video, pose_features, time_feats)

            if action_mask.sum() > 0:
                action_loss = focal_loss(
                    logits_act[action_mask > 0],
                    actions[action_mask > 0],
                    class_weights,
                    gamma=args.focal_gamma,
                    label_smoothing=args.label_smoothing
                )
                smooth_loss = temporal_smoothness_loss(logits_act[action_mask > 0])
                action_loss = action_loss + args.lambda_smooth * smooth_loss
            else:
                # Ensure gradient tracking even with zero loss
                action_loss = 0.0 * logits_act.sum()

            # Keypoint loss (only for multitask model)
            if args.model_type == 'multitask':
                kp_loss = masked_keypoint_loss(pred_kp, keypoints, visibility)
                loss = action_loss + args.lambda_kp * kp_loss
            else:
                # Ensure gradient tracking even with zero loss
                kp_loss = 0.0 * logits_act.sum()
                loss = action_loss

        total_loss += loss.item() if torch.is_tensor(loss) else 0.0
        num_batches += 1

        # Collect for metrics (only samples with actions)
        # Convert to float32 before numpy (bfloat16 not supported by numpy)
        if action_mask.sum() > 0:
            all_logits.append(logits_act[action_mask > 0].float().cpu().numpy())
            all_targets.append(actions[action_mask > 0].float().cpu().numpy())

            pred_labels = torch.argmax(logits_act[action_mask > 0], dim=-1).cpu().numpy()
            true_labels = torch.argmax(actions[action_mask > 0], dim=-1).cpu().numpy()
            all_pred_labels.append(pred_labels)
            all_true_labels.append(true_labels)

        # Collect keypoints (only for multitask model, convert to float32 before numpy)
        if args.model_type == 'multitask':
            all_pred_kp.append(pred_kp.float().cpu().numpy())
            all_true_kp.append(keypoints.float().cpu().numpy())
            all_visibility.append(visibility.float().cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / num_batches

    # Frame metrics
    if len(all_logits) > 0:
        all_logits = np.concatenate([x.reshape(-1, x.shape[-1]) for x in all_logits], axis=0)
        all_targets = np.concatenate([x.reshape(-1, x.shape[-1]) for x in all_targets], axis=0)
        frame_metrics = compute_frame_metrics(all_logits, all_targets, class_names)

        # Segment F1
        all_pred_labels = np.concatenate([x.flatten() for x in all_pred_labels], axis=0)
        all_true_labels = np.concatenate([x.flatten() for x in all_true_labels], axis=0)
        seg_f1 = compute_segment_f1(all_pred_labels, all_true_labels, iou_threshold=0.3)
    else:
        frame_metrics = {'macro_f1': 0.0, 'macro_acc': 0.0, 'best_thresholds': [], 'class_f1s': []}
        seg_f1 = 0.0

    # Keypoint MAE (only for multitask model)
    if args.model_type == 'multitask' and len(all_pred_kp) > 0:
        all_pred_kp = np.concatenate([x.reshape(-1, x.shape[-1]) for x in all_pred_kp], axis=0)
        all_true_kp = np.concatenate([x.reshape(-1, x.shape[-1]) for x in all_true_kp], axis=0)
        all_visibility = np.concatenate([x.reshape(-1, x.shape[-1]) for x in all_visibility], axis=0)

        # Expand visibility to match 2K
        K = all_visibility.shape[-1]
        visibility_2K = np.repeat(all_visibility, 2, axis=-1)  # (N, 2K)

        kp_mae = np.abs(all_pred_kp - all_true_kp) * visibility_2K
        kp_mae = kp_mae.sum() / (visibility_2K.sum() + 1e-6)
    else:
        kp_mae = 0.0

    return {
        'loss': avg_loss,
        'macro_f1': frame_metrics['macro_f1'],
        'macro_acc': frame_metrics['macro_acc'],
        'seg_f1_0.3': seg_f1,
        'kp_mae_norm': kp_mae,
        'best_thresholds': frame_metrics['best_thresholds'],
        'class_f1s': frame_metrics['class_f1s']
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-task training for mouse action + keypoint prediction")

    # Data
    parser.add_argument('--annotations', type=str, required=True, help='Path to action CSV folder')
    parser.add_argument('--videos', type=str, required=True, help='Path to videos folder')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')

    # Training
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate (decreased for better convergence)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='Weight decay (increased for better regularization)')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')

    # Model
    parser.add_argument('--model_type', type=str, default='multitask', choices=['action_only', 'multitask'],
                       help='Model type: action_only (actions from visual+pose) or multitask (actions+keypoints)')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=3, help='Epochs to freeze backbone')
    parser.add_argument('--focal_gamma', type=float, default=2.5, help='Focal loss gamma (increased for better focus on hard examples)')
    parser.add_argument('--lambda_smooth', type=float, default=5e-4, help='Temporal smoothness weight')
    parser.add_argument('--lambda_kp', type=float, default=1.0, help='Keypoint loss weight (only for multitask model)')
    parser.add_argument('--dropout', type=float, default=0.3, help='TCN dropout')
    parser.add_argument('--head_dropout', type=float, default=0.2, help='Task head dropout')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')

    # Temporal
    parser.add_argument('--train_T', type=int, default=180, help='Train window length')
    parser.add_argument('--val_T', type=int, default=240, help='Val window length')
    parser.add_argument('--clip_T', type=int, default=16, help='VideoMAE2 clip length')

    # Keypoints
    parser.add_argument('--kp_names', type=str,
                       default='mouth,tail_base,L_frontpaw,R_frontpaw,L_hindpaw,R_hindpaw',
                       help='Comma-separated keypoint names (use exact names from DLC CSV)')
    parser.add_argument('--kp_likelihood_thr', type=float, default=0.90, help='Keypoint likelihood threshold')

    # Dataset
    parser.add_argument('--min_frames', type=int, default=345, help='Minimum frames required (345 for 15-frame context window annotations)')
    parser.add_argument('--rare_threshold_share', type=float, default=0.02, help='Rare class threshold')
    parser.add_argument('--rare_boost_cap', type=float, default=30.0, help='Max rare class boost factor (increased from 12)')

    # Data Augmentation
    parser.add_argument('--kp_jitter_std', type=float, default=0.02, help='Keypoint jitter std (0=disabled)')
    parser.add_argument('--aug_brightness', type=float, default=0.2, help='Brightness jitter range')
    parser.add_argument('--aug_contrast', type=float, default=0.2, help='Contrast jitter range')
    parser.add_argument('--aug_temporal_drop', type=float, default=0.1, help='Temporal frame dropout probability')
    parser.add_argument('--aug_hflip', type=float, default=0.0, help='Horizontal flip probability (disabled for left/right keypoints)')

    # Scheduler
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs (increased from 3)')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum LR')
    parser.add_argument('--lr_schedule', type=str, default='cosine', choices=['cosine', 'plateau'], help='LR schedule type')
    parser.add_argument('--lr_patience', type=int, default=10, help='ReduceLROnPlateau patience (epochs)')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='ReduceLROnPlateau factor')

    # Early Stopping
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Early stopping patience (epochs, 0=disabled)')

    # EMA
    parser.add_argument('--use_ema', action='store_true', help='Use EMA')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay')

    # W&B Logging
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_entity', type=str, default='grassknoted', help='W&B entity name')
    parser.add_argument('--wandb_project', type=str, default='prj_mouse_pain', help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name (auto-generated if not provided)')

    args = parser.parse_args()

    # Derived arguments
    args.kp_names = [normalize_keypoint_name(name) for name in args.kp_names.split(',')]
    args.use_amp = torch.cuda.is_available()

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[info] Using device: {device}")

    # Initialize W&B
    if args.use_wandb and wandb is not None:
        # Auto-generate run name if not provided
        if args.wandb_run_name is None:
            # Determine encoder model name (will be set later after model creation)
            encoder_name = "videomae" if VideoMAEModel is not None else "vit2d"
            run_name = (
                f"{encoder_name}_{args.model_type}_"
                f"lr{args.lr:.0e}_bs{args.batch_size}_"
                f"drop{args.dropout}_T{args.train_T}"
            )
        else:
            run_name = args.wandb_run_name

        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=[args.model_type, "pose_graph"]
        )
        logger.info(f"[info] W&B logging enabled: {args.wandb_entity}/{args.wandb_project}/{run_name}")
    elif args.use_wandb and wandb is None:
        logger.warning("[warn] --use_wandb specified but wandb not installed. Skipping W&B logging.")
        logger.warning("[warn] Install with: pip install wandb")

    # Create datasets
    logger.info("[info] Creating datasets...")
    train_dataset = MiceActionDatasetFromAnnotations(
        annotations_dir=args.annotations,
        videos_dir=args.videos,
        kp_names=args.kp_names,
        split='train',
        train_T=args.train_T,
        val_T=args.val_T,
        img_size=args.img_size,
        kp_likelihood_thr=args.kp_likelihood_thr,
        rare_threshold_share=args.rare_threshold_share,
        rare_boost_cap=args.rare_boost_cap,
        min_frames=args.min_frames,
        seed=args.seed,
        kp_jitter_std=args.kp_jitter_std,
        aug_brightness=args.aug_brightness,
        aug_contrast=args.aug_contrast,
        aug_temporal_drop=args.aug_temporal_drop,
        aug_hflip=args.aug_hflip
    )

    val_dataset = MiceActionDatasetFromAnnotations(
        annotations_dir=args.annotations,
        videos_dir=args.videos,
        kp_names=args.kp_names,
        split='val',
        train_T=args.train_T,
        val_T=args.val_T,
        img_size=args.img_size,
        kp_likelihood_thr=args.kp_likelihood_thr,
        rare_threshold_share=args.rare_threshold_share,
        rare_boost_cap=1.0,  # No boosting for val
        min_frames=args.min_frames,
        seed=args.seed,
        kp_jitter_std=0.0,  # No jittering for val
        aug_brightness=0.0,  # No augmentation for val
        aug_contrast=0.0,
        aug_temporal_drop=0.0,
        aug_hflip=0.0
    )

    # Print comprehensive statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE DATA STATISTICS")
    logger.info("=" * 80)
    logger.info("")

    # 1. Dataset overview
    logger.info("1. DATASET OVERVIEW")
    logger.info("-" * 80)
    logger.info(f"Total unique videos/trials discovered: {len(train_dataset.samples) + len(val_dataset.samples)} "
                f"(before boosting)")
    logger.info(f"Training samples (after boosting): {len(train_dataset.samples)}")
    logger.info(f"Validation samples: {len(val_dataset.samples)}")
    logger.info(f"Train/Val split: {len(train_dataset.samples)}/{len(val_dataset.samples)} "
                f"({len(train_dataset.samples)/(len(train_dataset.samples)+len(val_dataset.samples))*100:.1f}% / "
                f"{len(val_dataset.samples)/(len(train_dataset.samples)+len(val_dataset.samples))*100:.1f}%)")
    logger.info("")

    # 2. Class information
    logger.info("2. CLASS INFORMATION")
    logger.info("-" * 80)
    logger.info(f"Original classes (before merging): {len(train_dataset.orig_classes)}")
    for i, cls in enumerate(train_dataset.orig_classes):
        logger.info(f"  {i}: {cls}")
    logger.info(f"\nMerged classes (after merging): {len(train_dataset.merged_classes)}")
    for i, cls in enumerate(train_dataset.merged_classes):
        logger.info(f"  {i}: {cls}")
    logger.info(f"\nMerge mapping: {train_dataset.col_map}")
    logger.info(f"Merged indices: {train_dataset.merge_idxs}")
    logger.info("")

    # 3. Sample distribution (by dominant class)
    logger.info("3. SAMPLE DISTRIBUTION (by dominant class in window)")
    logger.info("-" * 80)
    train_sample_classes = []
    for sample in train_dataset.samples:
        if sample['action_csv'] is not None:
            try:
                df = pd.read_csv(sample['action_csv'])
                sample_class_counts = Counter()
                for action_idx in df['Action'].values:
                    if isinstance(action_idx, (int, np.integer)) and action_idx < len(train_dataset.orig_classes):
                        merged_idx = train_dataset.col_map[action_idx]
                        merged_class = train_dataset.merged_classes[merged_idx]
                        sample_class_counts[merged_class] += 1
                if sample_class_counts:
                    dominant_class = sample_class_counts.most_common(1)[0][0]
                    train_sample_classes.append(dominant_class)
            except:
                pass

    sample_counts = Counter(train_sample_classes)
    total_samples = len(train_sample_classes)
    logger.info(f"{'Class':<20} {'Samples':<10} {'Share':<10}")
    logger.info("-" * 80)
    for cls in train_dataset.merged_classes:
        count = sample_counts.get(cls, 0)
        share = count / total_samples * 100 if total_samples > 0 else 0.0
        logger.info(f"{cls:<20} {count:<10} {share:>6.2f}%")
    logger.info("-" * 80)
    logger.info(f"{'TOTAL':<20} {total_samples:<10} {'100.00%':>10}")
    logger.info("")

    # 4. Temporal information
    logger.info("4. TEMPORAL INFORMATION")
    logger.info("-" * 80)
    logger.info(f"Training window length (train_T): {args.train_T} frames")
    logger.info(f"Validation window length (val_T): {args.val_T} frames")
    logger.info(f"Minimum frames required: {args.min_frames} frames")
    logger.info(f"VideoMAE2 clip length (if used): {args.clip_T} frames")
    logger.info("")

    # 5. Keypoint information
    logger.info("5. KEYPOINT INFORMATION")
    logger.info("-" * 80)
    kp_names_list = args.kp_names if isinstance(args.kp_names, list) else args.kp_names.split(',')
    logger.info(f"Number of keypoints: {len(kp_names_list)}")
    logger.info(f"Keypoint names: {kp_names_list}")
    logger.info(f"Keypoint likelihood threshold: {args.kp_likelihood_thr}")
    logger.info(f"Pose graph features: 18 (8 edge lengths + 10 angles)")
    logger.info("")

    # 6. Augmentation settings
    logger.info("6. AUGMENTATION SETTINGS (training only)")
    logger.info("-" * 80)
    logger.info(f"Keypoint jittering std: {args.kp_jitter_std}")
    logger.info(f"Brightness jitter: ±{args.aug_brightness}")
    logger.info(f"Contrast jitter: ±{args.aug_contrast}")
    logger.info(f"Temporal dropout probability: {args.aug_temporal_drop}")
    logger.info(f"Horizontal flip probability: {args.aug_hflip}")
    logger.info("")

    # 7. Class balancing settings
    logger.info("7. CLASS BALANCING SETTINGS")
    logger.info("-" * 80)
    logger.info(f"Rare class threshold: <{args.rare_threshold_share*100:.1f}% of samples")
    logger.info(f"Rare class boost cap: up to {args.rare_boost_cap}x replication")
    logger.info(f"Focal loss gamma: {args.focal_gamma}")
    logger.info(f"Label smoothing: {args.label_smoothing}")
    logger.info("")

    logger.info("=" * 80)
    logger.info("")

    # Smoke test
    logger.info("")
    logger.info("=" * 80)
    logger.info("SMOKE TEST - Checking Data Loading")
    logger.info("=" * 80)
    sample = train_dataset[0]
    logger.info(f"  Video shape: {sample['video'].shape}")
    logger.info(f"  Actions shape: {sample['actions'].shape}")
    logger.info(f"  Pose features shape: {sample['pose_features'].shape}")
    logger.info(f"  Keypoints shape: {sample['keypoints'].shape}")
    logger.info(f"  Visibility shape: {sample['visibility'].shape}")
    logger.info(f"  Time feats shape: {sample['time_feats'].shape}")
    logger.info(f"  Action mask: {sample['action_mask']}")

    # Check pose features statistics
    pose_feats = sample['pose_features'].numpy()
    logger.info(f"\nPose feature statistics:")
    logger.info(f"  Min: {pose_feats.min():.4f}, Max: {pose_feats.max():.4f}")
    logger.info(f"  Mean: {pose_feats.mean():.4f}, Std: {pose_feats.std():.4f}")
    logger.info(f"  NaN count: {np.isnan(pose_feats).sum()}")
    logger.info("=" * 80)
    logger.info("")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Create model
    logger.info(f"[info] Creating model (type: {args.model_type})...")
    num_classes = len(train_dataset.merged_classes)
    num_keypoints = len(args.kp_names)

    if args.model_type == 'action_only':
        model = ActionOnlyModel(
            num_classes=num_classes,
            num_pose_features=18,  # 8 edge lengths + 10 angles
            img_size=args.img_size,
            freeze_backbone_epochs=args.freeze_backbone_epochs,
            dropout=args.dropout,
            head_dropout=args.head_dropout
        ).to(device)
    else:  # multitask
        model = DINOv3_Temporal_MultiTask_VideoMAE2Preferred(
            num_classes=num_classes,
            num_keypoints=num_keypoints,
            num_pose_features=18,  # 8 edge lengths + 10 angles
            img_size=args.img_size,
            freeze_backbone_epochs=args.freeze_backbone_epochs,
            dropout=args.dropout,
            head_dropout=args.head_dropout
        ).to(device)

    # Update W&B run name with actual encoder
    if args.use_wandb and wandb is not None and wandb.run is not None:
        # Determine actual encoder used
        if hasattr(model, 'is_3d') and model.is_3d:
            encoder_name = "videomae"
        elif hasattr(model, 'backbone_2d') and model.backbone_2d is not None:
            # Get the actual 2D model name
            model_class = model.backbone_2d.__class__.__name__
            if 'dinov2' in str(model.backbone_2d).lower():
                encoder_name = "dinov2"
            elif 'augreg' in str(model.backbone_2d).lower():
                encoder_name = "vit_augreg"
            else:
                encoder_name = "vit2d"
        else:
            encoder_name = "unknown"

        # Update run name if it was auto-generated
        if args.wandb_run_name is None:
            new_run_name = (
                f"{encoder_name}_{args.model_type}_"
                f"lr{args.lr:.0e}_bs{args.batch_size}_"
                f"drop{args.dropout}_T{args.train_T}"
            )
            wandb.run.name = new_run_name
            logger.info(f"[info] Updated W&B run name to: {new_run_name}")

    # EMA model
    ema_model = None
    if args.use_ema:
        logger.info("[info] Creating EMA model...")
        import copy
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.requires_grad = False

    # Compute class weights
    class_counts = Counter()
    for sample in train_dataset.samples:
        if sample['action_csv'] is not None:
            try:
                df = pd.read_csv(sample['action_csv'])
                for action_idx in df['Action'].values:
                    # action_idx is a numeric index (0-7)
                    if isinstance(action_idx, (int, np.integer)) and action_idx < len(train_dataset.orig_classes):
                        merged_idx = train_dataset.col_map[action_idx]
                        merged_class = train_dataset.merged_classes[merged_idx]
                        class_counts[merged_class] += 1
            except:
                pass

    class_weights = torch.ones(num_classes, device=device)
    total = sum(class_counts.values())
    for i, cls in enumerate(train_dataset.merged_classes):
        count = class_counts.get(cls, 1)
        class_weights[i] = total / (count * num_classes)

    logger.info(f"[info] Class weights: {class_weights}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Choose scheduler type
    if args.lr_schedule == 'cosine':
        # Warmup + cosine scheduler
        warmup_steps = args.warmup_epochs * len(train_loader)
        total_steps = args.epochs * len(train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        plateau_scheduler = None
    else:  # plateau
        # Warmup for first few epochs, then plateau
        scheduler = None
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=args.lr_factor,
            patience=args.lr_patience,
            verbose=True,
            min_lr=args.min_lr
        )

    # Scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # Training loop
    best_val_f1 = 0.0
    checkpoint_path = Path(args.annotations).parent / 'best_model_multitask.pt'
    epochs_without_improvement = 0

    logger.info(f"\n{'='*80}")
    logger.info("Starting training")
    logger.info(f"{'='*80}\n")

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'-'*80}")

        # Update model epoch for backbone unfreezing
        model.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, class_weights, args, device, ema_model
        )

        logger.info(f"Train loss: {train_metrics['loss']:.4f} "
                   f"(action: {train_metrics['action_loss']:.4f}, kp: {train_metrics['kp_loss']:.4f})")

        # Validate
        val_model = ema_model if args.use_ema else model
        val_metrics = validate(val_model, val_loader, class_weights, train_dataset.merged_classes, args, device)

        logger.info(f"Val loss: {val_metrics['loss']:.4f}")

        # Action metrics
        logger.info("")
        logger.info("=" * 60)
        logger.info("ACTION METRICS")
        logger.info("=" * 60)
        logger.info(f"Val macro F1: {val_metrics['macro_f1']:.4f}")
        logger.info(f"Val macro Acc: {val_metrics['macro_acc']:.4f}")
        logger.info(f"Val segment F1@0.3: {val_metrics['seg_f1_0.3']:.4f}")

        # Per-class F1
        if len(val_metrics['class_f1s']) > 0:
            logger.info("Per-class F1:")
            for cls, f1 in zip(train_dataset.merged_classes, val_metrics['class_f1s']):
                logger.info(f"  {cls}: {f1:.4f}")

        # Keypoint metrics (only for multitask model)
        if args.model_type == 'multitask':
            logger.info("")
            logger.info("=" * 60)
            logger.info("KEYPOINT METRICS")
            logger.info("=" * 60)
            logger.info(f"Val keypoint MAE (norm): {val_metrics['kp_mae_norm']:.4f}")

        logger.info("=" * 60)
        logger.info("")

        # Log to W&B
        if args.use_wandb and wandb is not None:
            # Training metrics
            wandb_log = {
                "epoch": epoch,
                "train/loss": train_metrics['loss'],
                "train/action_loss": train_metrics['action_loss'],
                "train/kp_loss": train_metrics['kp_loss'],
                # Validation metrics
                "val/loss": val_metrics['loss'],
                "val/macro_f1": val_metrics['macro_f1'],
                "val/macro_acc": val_metrics['macro_acc'],
                "val/segment_f1_0.3": val_metrics['seg_f1_0.3'],
                # Learning rate
                "lr": optimizer.param_groups[0]['lr']
            }

            # Per-class F1 scores
            if len(val_metrics['class_f1s']) > 0:
                for cls, f1 in zip(train_dataset.merged_classes, val_metrics['class_f1s']):
                    wandb_log[f"val/f1_{cls}"] = f1

            # Keypoint metrics (multitask only)
            if args.model_type == 'multitask':
                wandb_log["val/kp_mae_norm"] = val_metrics['kp_mae_norm']

            wandb.log(wandb_log)

        # Save best model
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            epochs_without_improvement = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'classes': train_dataset.merged_classes,
                'orig_classes': train_dataset.orig_classes,
                'col_map': train_dataset.col_map,
                'merge_idxs': train_dataset.merge_idxs,
                'best_thresholds': val_metrics['best_thresholds'],
                'kp_names': args.kp_names,
                'val_metrics': val_metrics,
                'args': vars(args)
            }

            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()

            if args.use_ema:
                checkpoint['ema_state_dict'] = ema_model.state_dict()

            torch.save(checkpoint, checkpoint_path)
            logger.info(f"[info] Saved best checkpoint to {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            logger.info(f"\n[info] Early stopping triggered after {epoch + 1} epochs")
            logger.info(f"[info] No improvement for {epochs_without_improvement} epochs")
            break

        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        elif plateau_scheduler is not None:
            plateau_scheduler.step(val_metrics['macro_f1'])

    logger.info(f"\n{'='*80}")
    logger.info(f"Training complete! Best val macro F1: {best_val_f1:.4f}")
    logger.info(f"Checkpoint saved to: {checkpoint_path}")
    logger.info(f"{'='*80}\n")

    # Finish W&B run
    if args.use_wandb and wandb is not None:
        wandb.finish()
        logger.info("[info] W&B run finished")


if __name__ == '__main__':
    main()
