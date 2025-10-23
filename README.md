# Mouse Pain Action Recognition

A deep learning pipeline for recognizing pain-related behaviors in bottom-view video recordings of mice using 3D CNNs.

## Overview

This project implements a temporal action recognition system to detect and classify 6 distinct mouse behaviors from grayscale video:

- **rest**: Mouse at rest
- **paw_withdraw**: Acute paw withdrawal response (includes paw guard and flinch)
- **paw_lick**: Mouse licking its paw
- **paw_shake**: Rhythmic paw shaking
- **walk**: Walking behavior
- **active**: General active behavior

The model uses **3D CNNs** to capture spatiotemporal features from 16-frame temporal clips (about 0.5 seconds at 30 FPS), which is crucial for detecting short pain responses (5-10 frames).

## Project Structure

```
.
├── data_loader.py         # Data loading, preprocessing, and train/val split
├── model.py              # 3D CNN architecture definitions
├── train.py              # Training script with early stopping and checkpointing
├── evaluation.py         # Metrics computation (F1, precision, recall, confusion matrix)
├── inference.py          # Inference on new videos
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Installation

### Requirements
- Python 3.8+
- GPU recommended (NVIDIA GPU with CUDA support)

### Setup

```bash
# Clone or navigate to project directory
cd /path/to/prj_mouse_pain

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Format

### Video Files
- Format: MP4 (or any format supported by OpenCV)
- Resolution: Any size (will be processed as-is)
- Color space: Grayscale (RGB automatically converted if needed)
- Frame rate: 30 FPS (expected for correct temporal understanding)
- Duration: 12 seconds per video (360 frames)
- Location: `Videos/` directory

### Annotation Files
- Format: CSV with two columns: `Frame,Action`
- Action: Integer 0-7 (mapped to class indices)
- One row per frame (360 rows per video)
- Location: `Annotations/` directory
- Naming convention: Should match video filename

**Example annotation file:**
```
Frame,Action
0,0
1,0
2,0
3,1
...
```

## Quick Start

### 1. Organize Your Data

```
prj_mouse_pain/
├── Videos/
│   ├── video_1.mp4
│   ├── video_2.mp4
│   └── ...
├── Annotations/
│   ├── video_1_*.csv
│   ├── video_2_*.csv
│   └── ...
└── [training scripts]
```

### 2. Train Model

```python
from train import train_model

run_dir = train_model(
    video_dir="./Videos",
    annotation_dir="./Annotations",
    output_dir="./checkpoints",
    num_epochs=100,
    batch_size=32,
    clip_length=16,
    learning_rate=1e-3,
    model_type="standard",  # or "light" for lower memory usage
    use_amp=True  # Mixed precision for faster training
)
```

Or run from command line:

```bash
python train.py
```

### 3. Monitor Training

```bash
tensorboard --logdir=./checkpoints
```

Open browser to `http://localhost:6006` to view real-time metrics.

### 4. Evaluate Model

```python
from inference import ActionRecognitionInference
from evaluation import evaluate_and_save_results

# Load best model
checkpoint_path = "./checkpoints/run_20240101_120000/best_model_epoch50.pt"
inference = ActionRecognitionInference(checkpoint_path)

# Get predictions
predictions = inference.predict_video(
    video_path="./Videos/video_1.mp4",
    stride=2,
    return_probs=True
)

# Print results
for pred in predictions["predictions"][:10]:
    print(f"Frame {pred['frame']}: {pred['action']} ({pred['confidence']:.2%})")
```

### 5. Batch Prediction

```python
from inference import predict_multiple_videos

predict_multiple_videos(
    checkpoint_path="./checkpoints/run_20240101_120000/best_model_epoch50.pt",
    video_paths=["./Videos/video_1.mp4", "./Videos/video_2.mp4"],
    output_dir="./predictions",
    stride=2
)
```

## Model Architecture

### 3D CNN (Standard)
- **Input**: (B, 1, 16, H, W) - Grayscale 16-frame clips
- **Architecture**:
  - 4 3D convolutional blocks with increasing channels (32 → 64 → 128 → 256)
  - Batch normalization and ReLU after each conv layer
  - Max pooling for spatial and temporal downsampling
  - Global average pooling to create feature vector
  - 2 fully connected layers (256 → 512 → num_classes)
  - Dropout (0.5) for regularization
- **Parameters**: ~3M

### 3D CNN (Light)
- Lighter version with fewer channels (32 → 64 → 128)
- **Parameters**: ~600K
- Recommended for limited memory

## Training Details

### Class Weighting
- Automatically computed as inverse class frequency
- Critical for handling severe class imbalance (pain responses are rare)
- Applied in the loss function (CrossEntropyLoss with weight parameter)

### Optimization
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Loss**: CrossEntropyLoss with class weights
- **LR Scheduler**: Cosine Annealing with Warm Restarts (T_0=10)
- **Mixed Precision**: Uses automatic casting for faster training on compatible GPUs
- **Gradient Clipping**: Max norm = 1.0 for stability
- **Early Stopping**: Patience = 15 epochs

### Hyperparameters (Recommended)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 32 | Adjust based on GPU memory |
| Learning rate | 1e-3 | Initial; scheduler will reduce |
| Clip length | 16 frames | 0.53 seconds at 30 FPS |
| Weight decay | 1e-5 | Mild L2 regularization |
| Dropout | 0.5 | In FC layers |
| Epochs | 100 | With early stopping |
| Stride | 1 | Sample every frame during training |

## Evaluation Metrics

The evaluation module computes:

- **Per-frame accuracy**: Percentage of correctly classified frames
- **F1 Score (macro)**: Unweighted mean of per-class F1 scores (best for imbalanced data)
- **F1 Score (weighted)**: Class-weighted mean F1
- **Per-class metrics**: Precision, recall, F1 for each action class
- **Confusion matrix**: Shows which classes are confused
- **Classification report**: Detailed breakdown per class

### Interpretation for Publication

Focus on:
1. **F1 (macro)** - Primary metric for imbalanced data
2. **Per-class F1** - Especially important for pain response ("paw_withdraw")
3. **Recall for pain behaviors** - Don't want to miss pain events
4. **Specificity** - Need good "rest" classification to avoid false positives

## Temporal Clip Strategy

Why 16-frame clips centered on target frame?

1. **Pain responses are SHORT**: 5-10 frames, need context
2. **Temporal dynamics matter**: Model sees before/after behavior
3. **Frame rate consideration**: 16 frames ≈ 0.5 seconds at 30 FPS
4. **Trade-off**: Balance between context and computational cost

The model predicts the action at the **center frame** based on 8 frames before and 8 frames after.

## Tips for Best Results

### 1. Data Quality
- Ensure annotations are accurate (especially pain responses)
- Check that video frame counts match annotation CSV rows
- Verify consistent video format and resolution

### 2. Training
- Monitor training curves in TensorBoard
- If overfitting appears, increase dropout or regularization
- If underfitting, increase model capacity or reduce dropout
- Save the best model by F1 score, not just accuracy

### 3. Inference
- Use stride > 1 for faster inference if full frame-by-frame prediction not needed
- Return probabilities to assess model confidence
- Post-process predictions to filter low-confidence frames if needed

### 4. Class Imbalance
- The model automatically weights classes by inverse frequency
- For very rare classes, consider focal loss or data augmentation
- Temporal augmentation (frame jittering) could help

## Advanced Usage

### Custom Data Augmentation

Modify `data_loader.py` to add augmentations:

```python
from torchvision.transforms import Compose

transform = Compose([
    # Add custom transforms here
])

dataset = MouseActionDataset(..., transform=transform)
```

### Multi-GPU Training

```python
model = nn.DataParallel(model)  # Wrap model for multi-GPU
```

### Custom Loss Functions

Replace CrossEntropyLoss in `train.py`:

```python
criterion = FocalLoss(alpha=class_weights, gamma=2.0)
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size (e.g., 16 or 8)
- Use "light" model type
- Increase num_workers or reduce clip_length

### Poor Pain Response Detection
- Check annotation quality
- Increase class weight for pain classes
- Use focal loss instead of weighted cross-entropy
- Ensure sufficient negative (rest) samples in training

### Slow Inference
- Increase stride parameter (e.g., stride=5)
- Use "light" model architecture
- Reduce video resolution if possible

## Key Implementation Details

### Data Loader (`data_loader.py:data_loader.py`)
- Loads videos on-the-fly (doesn't require storing all frames in memory)
- Creates temporal clips with configurable length and stride
- Handles frame count validation against annotations
- Automatically merges action classes (paw_guard, flinch → paw_withdraw)
- Computes class weights for handling severe imbalance

### Model (`model.py:model.py`)
- 3D CNN with 4 convolutional blocks
- Spatiotemporal feature learning via 3D kernels
- Global average pooling for robustness
- Two variants: standard (3M params) and light (600K params)

### Training (`train.py:train.py`)
- Weighted loss to handle class imbalance
- Automatic mixed precision for faster training
- Gradient clipping for stability
- Early stopping and best model checkpointing
- TensorBoard integration for monitoring

### Evaluation (`evaluation.py:evaluation.py`)
- Comprehensive metrics (F1, precision, recall)
- Confusion matrix visualization
- Per-class performance breakdown
- Publication-ready plots

## References

- **3D CNNs for Video**: Tran et al. "Learning Spatiotemporal Features with 3D Convolutional Networks" (ICCV 2015)
- **Class Weighting**: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
- **PyTorch Docs**: https://pytorch.org/docs/
