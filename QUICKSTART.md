# Quick Start Guide

Get your model training in 5 minutes.

## Prerequisites
- Python 3.8+
- NVIDIA GPU (CUDA) recommended
- 1000 videos in `Videos/` folder
- 1000 annotation CSVs in `Annotations/` folder

## Step 1: Install Dependencies (2 minutes)

```bash
cd /path/to/prj_mouse_pain
pip install -r requirements.txt
```

## Step 2: Verify Setup (1 minute)

```bash
python test_pipeline.py
```

You should see:
```
✓ PASS for all 4 tests
```

If any test fails, fix the issue before proceeding.

## Step 3: Train Model (hours, depends on GPU)

### Default (Recommended for Publication Quality)
```bash
python train.py
```

### Or from Python
```python
from train import train_model

train_model(
    video_dir="./Videos",
    annotation_dir="./Annotations",
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    model_type="standard"
)
```

### Watch Training Progress
In another terminal:
```bash
tensorboard --logdir=./checkpoints
```
Then open `http://localhost:6006` in your browser.

## Step 4: Get Predictions (seconds to minutes)

After training completes:

```python
from inference import ActionRecognitionInference

# Load best trained model
checkpoint_path = "./checkpoints/run_YYYYMMDD_HHMMSS/best_model_epochN.pt"
inference = ActionRecognitionInference(checkpoint_path)

# Predict on one video
predictions = inference.predict_video(
    "./Videos/my_video.mp4",
    stride=2,
    return_probs=True
)

# Print first 10 predictions
for pred in predictions["predictions"][:10]:
    print(f"Frame {pred['frame']:3d}: {pred['action']:15s} ({pred['confidence']:.2%})")
```

## Step 5: Evaluate Results

```python
from inference import ActionRecognitionInference
from evaluation import evaluate_and_save_results
from data_loader import create_data_loaders
import torch

# Load model
checkpoint_path = "./checkpoints/run_YYYYMMDD_HHMMSS/best_model_epochN.pt"
checkpoint = torch.load(checkpoint_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

from model import create_model
model = create_model(
    num_classes=7,
    clip_length=16,
    model_type=checkpoint["config"]["model_type"],
    device=device
)
model.load_state_dict(checkpoint["model_state"])

# Load validation data
_, val_loader, _ = create_data_loaders(
    "./Videos",
    "./Annotations",
    batch_size=32,
    test_size=0.2
)

# Evaluate
evaluate_and_save_results(model, val_loader, device, "./results")
```

This will create:
- `results/metrics.txt` - detailed metrics
- `results/confusion_matrix.png` - confusion matrix
- `results/per_class_metrics.png` - F1, precision, recall per class

## Common Adjustments

### Running Out of GPU Memory?
```python
train_model(
    video_dir="./Videos",
    annotation_dir="./Annotations",
    batch_size=16,              # Reduce from 32
    model_type="light"          # Use lightweight model
)
```

### Want Faster Training?
```python
train_model(
    video_dir="./Videos",
    annotation_dir="./Annotations",
    batch_size=64,              # Increase batch size
    clip_length=8               # Shorter clips
)
```

### Want Better Results?
```python
train_model(
    video_dir="./Videos",
    annotation_dir="./Annotations",
    num_epochs=200,             # More epochs
    learning_rate=5e-4,         # Smaller LR
    batch_size=32
)
```

### Predicting on All Videos
```python
from inference import predict_multiple_videos
from pathlib import Path

video_paths = list(Path("./Videos").glob("*.mp4"))

predict_multiple_videos(
    checkpoint_path="./checkpoints/run_YYYYMMDD_HHMMSS/best_model_epochN.pt",
    video_paths=video_paths,
    output_dir="./predictions",
    stride=2
)
```

## Expected Output

### During Training
```
Epoch 1/100
  Batch 0/500, Loss: 1.8234, Acc: 45.32%
  Batch 20/500, Loss: 1.5123, Acc: 58.22%
  ...
Train Loss: 1.2345, Acc: 65.43%
Val Loss: 1.1234, Acc: 68.21%, F1 (macro): 0.6234
  Saved checkpoint: ./checkpoints/run_20240101_120000/best_model_epoch0.pt
```

### After Training
```
Checkpoint saved to: ./checkpoints/run_20240101_120000
- best_model_epoch50.pt (best F1 score)
- config.json (hyperparameters used)
- tensorboard logs
```

## Key Metrics to Watch

In TensorBoard:
- **Loss (train & val)**: Should decrease smoothly
- **Accuracy (train & val)**: Should increase
- **F1 (macro)**: Primary metric - should improve

In results files:
- **F1 (macro)**: 0.75-0.85 is good
- **paw_withdraw F1**: Critical - should be 0.70+ for publication
- **Confusion matrix**: Check which classes are confused

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce `batch_size` or use `model_type="light"` |
| No training progress | Check data with `test_pipeline.py` |
| NaN loss | Lower `learning_rate` to 5e-4 |
| Poor pain detection | Verify annotation quality; increase pain class weight |
| Slow training | Increase `batch_size` or use GPU with more memory |

## Files Generated

After training:
```
checkpoints/
└── run_20240101_120000/
    ├── best_model_epoch50.pt  ← Use this for inference
    ├── best_model_epoch45.pt  ← Earlier checkpoints (if saved)
    ├── config.json            ← Hyperparameters used
    ├── events.out.tfevents.*  ← TensorBoard logs
    └── ...

predictions/
├── video_001_predictions.json
├── video_002_predictions.json
└── ...

results/
├── metrics.txt
├── confusion_matrix.png
├── per_class_metrics.png
└── class_distribution.png
```

## What to Report in Your Paper

1. **Data**: 1000 videos, 80/20 train/val split
2. **Model**: 3D CNN, 3M parameters, 16-frame clips
3. **Training**: Adam optimizer, weighted loss, early stopping
4. **Results**:
   - Overall accuracy: X%
   - **F1 (macro): X.XX** ← Most important
   - Per-class F1 (especially paw_withdraw)
   - Confusion matrix

## Next: Advanced

Read `IMPLEMENTATION_SUMMARY.md` for detailed explanations.
Read `README.md` for comprehensive documentation.

Good luck! 🚀
