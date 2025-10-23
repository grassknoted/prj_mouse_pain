"""
Inference script for action recognition model.
Supports both single videos and batch predictions.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import json
from model import create_model


CLASS_NAMES = [
    "rest",
    "paw_withdraw",
    "paw_lick",
    "paw_shake",
    "walk",
    "active"
]


class ActionRecognitionInference:
    """
    Inference wrapper for action recognition model.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        clip_length: int = 16
    ):
        """
        Initialize inference module.

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to use ("cuda" or "cpu")
            clip_length: Temporal clip length used during training
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.clip_length = clip_length
        self.half_clip = clip_length // 2

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model and load weights
        self.model = create_model(
            num_classes=len(CLASS_NAMES),
            clip_length=clip_length,
            model_type=checkpoint["config"]["model_type"],
            device=device
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        print(f"Loaded model from {checkpoint_path}")
        print(f"Using device: {device}")

    def predict_frame(
        self,
        video_path: str,
        frame_idx: int
    ) -> Tuple[str, float]:
        """
        Predict action for a single frame.

        Args:
            video_path: Path to video file
            frame_idx: Frame index to predict

        Returns:
            (action_name, confidence)
        """
        frames = self._extract_clip(video_path, frame_idx)

        if frames is None:
            return None, 0.0

        with torch.no_grad():
            frames = frames.unsqueeze(0).to(self.device)  # Add batch dimension
            logits = self.model(frames)
            probs = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        return CLASS_NAMES[pred_class], confidence

    def predict_video(
        self,
        video_path: str,
        stride: int = 1,
        return_probs: bool = False
    ) -> Dict:
        """
        Predict actions for all frames in a video.

        Args:
            video_path: Path to video file
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
            "frame_count": frame_count,
            "predictions": [],
            "frame_indices": []
        }

        if return_probs:
            predictions["probabilities"] = []

        # Only predict frames that have enough context
        for frame_idx in range(self.half_clip, frame_count - self.half_clip, stride):
            action, confidence = self.predict_frame(video_path, frame_idx)

            if action is not None:
                predictions["frame_indices"].append(frame_idx)
                predictions["predictions"].append({
                    "frame": frame_idx,
                    "action": action,
                    "confidence": confidence
                })

                if return_probs:
                    probs = self._get_probabilities(video_path, frame_idx)
                    predictions["probabilities"].append(probs)

        return predictions

    def _extract_clip(
        self,
        video_path: str,
        center_frame: int
    ) -> torch.Tensor:
        """Extract temporal clip around center frame."""
        start_frame = center_frame - self.half_clip
        end_frame = center_frame + self.half_clip + 1

        if start_frame < 0:
            return None

        frames = []
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame > frame_count:
            return None

        for frame_idx in range(start_frame, end_frame):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video.read()

            if not ret:
                video.release()
                return None

            # Convert to grayscale
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames.append(frame)

        video.release()

        clip = np.stack(frames, axis=0)
        clip = torch.from_numpy(clip).float() / 255.0
        return clip

    def _get_probabilities(
        self,
        video_path: str,
        frame_idx: int
    ) -> List[float]:
        """Get class probabilities for a frame."""
        frames = self._extract_clip(video_path, frame_idx)

        if frames is None:
            return None

        with torch.no_grad():
            frames = frames.unsqueeze(0).to(self.device)
            logits = self.model(frames)
            probs = torch.softmax(logits, dim=1)

        return probs[0].cpu().numpy().tolist()

    def save_predictions(self, predictions: Dict, output_path: str):
        """Save predictions to JSON file."""
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved predictions to {output_path}")


def predict_multiple_videos(
    checkpoint_path: str,
    video_paths: List[str],
    output_dir: str = "./predictions",
    device: str = None,
    stride: int = 2
):
    """
    Batch prediction on multiple videos.

    Args:
        checkpoint_path: Path to trained model
        video_paths: List of video file paths
        output_dir: Directory to save predictions
        device: Device to use
        stride: Frame stride for predictions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize inference
    inference = ActionRecognitionInference(checkpoint_path, device=device)

    # Predict on all videos
    for video_path in video_paths:
        print(f"\nProcessing {video_path}...")

        predictions = inference.predict_video(
            video_path,
            stride=stride,
            return_probs=True
        )

        # Save predictions
        video_name = Path(video_path).stem
        output_path = output_dir / f"{video_name}_predictions.json"
        inference.save_predictions(predictions, str(output_path))

        # Print summary
        num_frames = len(predictions["frame_indices"])
        print(f"  Predicted {num_frames} frames")


if __name__ == "__main__":
    # Example usage
    checkpoint_path = "./checkpoints/run_20240101_120000/best_model_epoch50.pt"
    video_path = "/Users/anagara8/Documents/prj_mouse_pain/Videos/shortened_2023-10-25_CFA_010_267M_tracking.mp4"

    # Single video prediction
    inference = ActionRecognitionInference(checkpoint_path)
    predictions = inference.predict_video(video_path, stride=5, return_probs=True)

    # Print first 10 predictions
    print("Sample predictions:")
    for pred in predictions["predictions"][:10]:
        print(f"  Frame {pred['frame']}: {pred['action']} ({pred['confidence']:.4f})")
