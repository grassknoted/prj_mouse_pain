"""
Inference script for multimodal action recognition model.
Makes predictions using both visual frames and pose features.
"""

import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from multimodal_model import create_multimodal_model
from pose_graph import PoseGraph


CLASS_NAMES = [
    "rest",
    "paw_withdraw",
    "paw_lick",
    "paw_shake",
    "walk",
    "active"
]


class MultimodalActionRecognitionInference:
    """
    Inference wrapper for multimodal action recognition.
    Uses both visual frames and DLC pose features.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dlc_csv_path: Optional[str] = None,
        device: str = None,
        clip_length: int = 16
    ):
        """
        Initialize inference module.

        Args:
            checkpoint_path: Path to trained multimodal model checkpoint
            dlc_csv_path: Path to DLC coordinates CSV (can be set per video later)
            device: Device to use ("cuda" or "cpu")
            clip_length: Temporal clip length
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.clip_length = clip_length
        self.half_clip = clip_length // 2
        self.dlc_csv_path = dlc_csv_path

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create and load model
        self.model = create_multimodal_model(
            num_classes=len(CLASS_NAMES),
            model_type=checkpoint["config"]["model_type"],
            device=device
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        print(f"Loaded multimodal model from {checkpoint_path}")
        print(f"Using device: {device}")

    def predict_frame(
        self,
        video_path: str,
        frame_idx: int,
        dlc_csv_path: str
    ) -> Tuple[str, float]:
        """
        Predict action for a single frame.

        Args:
            video_path: Path to video file
            frame_idx: Frame index to predict
            dlc_csv_path: Path to DLC coordinates CSV

        Returns:
            (action_name, confidence)
        """
        visual_clip = self._extract_visual_clip(video_path, frame_idx)
        pose_clip = self._extract_pose_clip(dlc_csv_path, frame_idx)

        if visual_clip is None or pose_clip is None:
            return None, 0.0

        with torch.no_grad():
            visual = visual_clip.unsqueeze(0).to(self.device)  # (1, 1, T, H, W)
            pose = pose_clip.unsqueeze(0).to(self.device)      # (1, T, 18)

            logits = self.model(visual, pose)
            probs = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        return CLASS_NAMES[pred_class], confidence

    def predict_video(
        self,
        video_path: str,
        dlc_csv_path: str,
        stride: int = 1,
        return_probs: bool = False
    ) -> Dict:
        """
        Predict actions for all frames in a video.

        Args:
            video_path: Path to video file
            dlc_csv_path: Path to DLC coordinates CSV
            stride: Process every stride-th frame
            return_probs: Return class probabilities

        Returns:
            Dictionary with frame-level predictions
        """
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()

        predictions = {
            "video_path": video_path,
            "dlc_path": dlc_csv_path,
            "frame_count": frame_count,
            "predictions": [],
            "frame_indices": []
        }

        if return_probs:
            predictions["probabilities"] = []

        # Predict frames with sufficient context
        for frame_idx in range(self.half_clip, frame_count - self.half_clip, stride):
            action, confidence = self.predict_frame(video_path, frame_idx, dlc_csv_path)

            if action is not None:
                predictions["frame_indices"].append(frame_idx)
                predictions["predictions"].append({
                    "frame": frame_idx,
                    "action": action,
                    "confidence": float(confidence)
                })

                if return_probs:
                    probs = self._get_probabilities(video_path, frame_idx, dlc_csv_path)
                    predictions["probabilities"].append(probs)

        return predictions

    def _extract_visual_clip(
        self,
        video_path: str,
        center_frame: int
    ) -> Optional[torch.Tensor]:
        """Extract visual temporal clip."""
        start_frame = center_frame - self.half_clip
        end_frame = center_frame + self.half_clip + 1

        if start_frame < 0:
            return None

        frames = []
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame > frame_count:
            video.release()
            return None

        for frame_idx in range(start_frame, end_frame):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video.read()

            if not ret:
                video.release()
                return None

            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames.append(frame)

        video.release()

        clip = np.stack(frames, axis=0)
        clip = torch.from_numpy(clip).float() / 255.0
        return clip  # (T, H, W)

    def _extract_pose_clip(
        self,
        dlc_csv_path: str,
        center_frame: int
    ) -> Optional[torch.Tensor]:
        """Extract pose features temporal clip."""
        start_frame = center_frame - self.half_clip
        end_frame = center_frame + self.half_clip + 1

        if start_frame < 0:
            return None

        # Load DLC coordinates
        dlc_coords = self._load_dlc_coordinates(dlc_csv_path)

        if dlc_coords is None or end_frame > dlc_coords.shape[0]:
            return None

        # Extract pose clip
        pose_clip = np.zeros((end_frame - start_frame, 18))

        for i, frame_idx in enumerate(range(start_frame, end_frame)):
            try:
                keypoints = dlc_coords[frame_idx]
                pose_graph = PoseGraph(keypoints)
                features = pose_graph.construct_graph()
                pose_clip[i] = np.array(features)
            except Exception as e:
                pose_clip[i] = 0.0

        return torch.FloatTensor(pose_clip)  # (T, 18)

    def _load_dlc_coordinates(self, dlc_csv_path: str) -> Optional[np.ndarray]:
        """
        Load DLC coordinates from CSV.

        Returns:
            (N_frames, N_keypoints, 2) array or None if error
        """
        try:
            df = pd.read_csv(dlc_csv_path, header=[1, 2])

            coords_list = []
            bodyparts = df.columns.get_level_values(0).unique()

            for bodypart in bodyparts:
                x = df[(bodypart, 'x')].values
                y = df[(bodypart, 'y')].values
                coords_list.append(np.stack([x, y], axis=1))

            coords = np.stack(coords_list, axis=1)
            return coords

        except Exception as e:
            print(f"Error loading DLC coordinates from {dlc_csv_path}: {e}")
            return None

    def _get_probabilities(
        self,
        video_path: str,
        frame_idx: int,
        dlc_csv_path: str
    ) -> List[float]:
        """Get class probabilities for a frame."""
        visual_clip = self._extract_visual_clip(video_path, frame_idx)
        pose_clip = self._extract_pose_clip(dlc_csv_path, frame_idx)

        if visual_clip is None or pose_clip is None:
            return None

        with torch.no_grad():
            visual = visual_clip.unsqueeze(0).to(self.device)
            pose = pose_clip.unsqueeze(0).to(self.device)

            logits = self.model(visual, pose)
            probs = torch.softmax(logits, dim=1)

        return probs[0].cpu().numpy().tolist()

    def save_predictions(self, predictions: Dict, output_path: str):
        """Save predictions to JSON file."""
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved predictions to {output_path}")


def predict_multiple_videos_multimodal(
    checkpoint_path: str,
    video_paths: List[str],
    dlc_paths: List[str],
    output_dir: str = "./predictions_multimodal",
    device: str = None,
    stride: int = 2
):
    """
    Batch prediction on multiple videos using multimodal model.

    Args:
        checkpoint_path: Path to trained model
        video_paths: List of video file paths
        dlc_paths: List of DLC coordinate CSV paths
        output_dir: Directory to save predictions
        device: Device to use
        stride: Frame stride for predictions
    """
    assert len(video_paths) == len(dlc_paths), \
        "Number of videos must match number of DLC files"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize inference
    inference = MultimodalActionRecognitionInference(
        checkpoint_path, device=device
    )

    # Predict on all videos
    for video_path, dlc_path in zip(video_paths, dlc_paths):
        print(f"\nProcessing {video_path}...")

        predictions = inference.predict_video(
            video_path,
            dlc_path,
            stride=stride,
            return_probs=True
        )

        # Save predictions
        video_name = Path(video_path).stem
        output_path = output_dir / f"{video_name}_predictions.json"
        inference.save_predictions(predictions, str(output_path))

        num_frames = len(predictions["frame_indices"])
        print(f"  Predicted {num_frames} frames")


if __name__ == "__main__":
    # Example usage
    checkpoint_path = "./checkpoints_multimodal/run_20240101_120000/best_model_epoch50.pt"
    video_path = "./Videos/sample_video.mp4"
    dlc_path = "./DLC/sample_video_dlc.csv"

    # Single video prediction
    try:
        inference = MultimodalActionRecognitionInference(checkpoint_path)
        predictions = inference.predict_video(video_path, dlc_path, stride=5, return_probs=True)

        # Print first 10 predictions
        print("Sample predictions:")
        for pred in predictions["predictions"][:10]:
            print(f"  Frame {pred['frame']}: {pred['action']:15s} ({pred['confidence']:.4f})")

    except Exception as e:
        print(f"Error: {e}")
