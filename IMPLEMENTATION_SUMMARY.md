# Implementation Summary: Mouse Pain Action Recognition

## What Was Built

A complete, publication-quality deep learning pipeline for action recognition in mouse behavioral videos. The system detects and classifies 6 distinct mouse behaviors, with special focus on pain responses (paw withdrawal, licking, shaking, etc.).

## Why This Approach?

### 3D CNN Architecture
- **Why**: Pain responses are very short (5-10 frames), so we need to capture temporal context
- **What we built**: 4-layer 3D CNN that learns spatiotemporal features from 16-frame clips
- **How it works**:
  - Each frame is treated as part of a temporal sequence (not independently)
  - 3D convolutions slide over both space AND time simultaneously
  - This lets the model understand "before, during, and after" behavior

### Class Weighting
- **Why**: Pain responses are rare (much fewer frames than "rest" behavior)
- **What we built**: Automatic inverse-frequency weighting for the loss function
- **How it works**:
  - Common classes (rest) get lower weight (less penalty for errors)
  - Rare classes (pain) get higher weight (high penalty for misses)
  - This forces the model to focus on detecting the important behaviors

### Temporal Clipping Strategy
- **Why**: Need good temporal context without excessive computation
- **What we built**: 16-frame center-aligned clips (8 frames before + center + 8 frames after)
- **How it works**:
  - At 30 FPS, this is ~0.5 seconds of context
  - Perfect for capturing the dynamics of pain responses
  - Model predicts the action at the center frame based on full context

## Key Files Explained

### `data_loader.py`
Handles all data pipeline complexity:
- Loads videos on-the-fly (doesn't require storing all frames in RAM)
- Extracts temporal clips efficiently
- Merges action classes (paw_guard + flinch → paw_withdraw)
- Computes class weights automatically
- Handles train/val split at video level (prevents data leakage)

**Key class**: `MouseActionDataset` - loads video clips and returns (temporal_clip, label)

### `model.py`
Two model variants for different memory constraints:
- **Standard (3M params)**: Full model with 4 conv blocks for best performance
- **Light (600K params)**: Reduced channels for GPU memory constraints

Both use:
- 3D convolutions (spatiotemporal learning)
- Batch normalization (training stability)
- ReLU activation (non-linearity)
- Global average pooling (robustness to spatial shifts)
- 2 FC layers with dropout (regularization)

### `train.py`
Complete training pipeline:
- Weighted loss function (handles class imbalance)
- Adam optimizer with cosine annealing + warm restarts
- Mixed precision training (2x faster on NVIDIA GPUs)
- Gradient clipping (prevents exploding gradients)
- Early stopping (prevents overfitting)
- Automatic model checkpointing (saves best model by F1 score)
- TensorBoard logging (visualize training in real-time)

### `evaluation.py`
Comprehensive evaluation toolkit:
- Per-class F1, precision, recall
- Macro and weighted F1 (F1 macro is primary metric for imbalanced data)
- Confusion matrix visualization
- Per-class metrics plots
- Classification report

### `inference.py`
Make predictions on new videos:
- Single frame prediction: `predict_frame(video_path, frame_idx)`
- Full video prediction: `predict_video(video_path, stride=2)`
- Batch prediction on multiple videos
- Returns per-frame probabilities for confidence assessment

## How to Use

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Test Pipeline (optional but recommended)
```bash
python test_pipeline.py
```

### 3. Train
```bash
python train.py
```
Or programmatically:
```python
from train import train_model
train_model("./Videos", "./Annotations")
```

### 4. Monitor Training
```bash
tensorboard --logdir=./checkpoints
```

### 5. Evaluate & Get Predictions
```python
from inference import ActionRecognitionInference
inference = ActionRecognitionInference("./checkpoints/run_xxx/best_model_epoch50.pt")
predictions = inference.predict_video("./Videos/video_1.mp4", stride=2)
```

## Design Decisions & Rationale

### Why 16-frame clips?
- 5-10 frame pain responses need context
- 16 frames = 0.5 seconds at 30 FPS = good balance
- Odd length (16+1 center) allows center-aligned prediction

### Why global average pooling?
- Makes model robust to small spatial shifts
- Better than fully connected layers after conv
- Reduces parameters and overfitting

### Why cosine annealing with restarts?
- Standard learning rate decay sometimes gets stuck
- Cosine annealing gradually reduces LR smoothly
- Warm restarts escape local minima by periodically increasing LR

### Why early stopping?
- Prevents overfitting
- Saves computation (stops unnecessary epochs)
- Uses patience=15 to give learning some margin

### Why save by F1 not accuracy?
- F1 is more important for imbalanced data
- Macro F1 treats all classes equally (best for publication)
- Accuracy can be misleading when one class dominates

## Performance Expectations

For publication-quality work with 1000 videos:

**Expected Results** (ballpark estimates):
- Overall accuracy: 80-90% (but not the main metric)
- **F1 macro: 0.75-0.85** ← Primary metric
- Precision (pain behaviors): 0.80-0.90
- Recall (pain behaviors): 0.70-0.85
- Inference: 30-60 FPS on modern GPU

**Factors affecting performance:**
- Annotation quality (most important!)
- Class distribution
- Video quality and consistency
- Hyperparameter tuning

## Advanced Modifications

### For Better Performance on Pain Responses
1. Increase class weight for paw_withdraw manually
2. Use focal loss instead of weighted cross-entropy
3. Add temporal data augmentation (frame jittering)
4. Collect more pain response examples if possible

### For Faster Training
1. Reduce clip_length (e.g., 8 instead of 16)
2. Increase stride during training (skip frames)
3. Use batch size 16 or 8 if memory allows
4. Use "light" model variant

### For Better Generalization
1. Add spatial augmentation (random crops, flips)
2. Temporal augmentation (frame jittering)
3. Mixup or CutMix data augmentation
4. Ensemble predictions from multiple models

## File Organization

Your project structure should be:
```
prj_mouse_pain/
├── Videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ... (1000 videos)
├── Annotations/
│   ├── video_001_*.csv
│   ├── video_002_*.csv
│   └── ... (1000 CSVs)
├── checkpoints/
│   └── run_YYYYMMDD_HHMMSS/
│       ├── best_model_epochN.pt
│       └── config.json
├── predictions/
│   ├── video_001_predictions.json
│   └── ...
├── data_loader.py
├── model.py
├── train.py
├── evaluation.py
├── inference.py
├── test_pipeline.py
├── requirements.txt
└── README.md
```

## Common Issues & Solutions

### CUDA Out of Memory
- **Reduce batch_size**: 32 → 16 → 8
- **Use light model**: model_type="light"
- **Reduce clip_length**: 16 → 8 (less temporal context)

### Poor Pain Detection (Low F1 on paw_withdraw)
- **Check annotations**: Are pain responses correctly labeled?
- **Increase weighting**: Manually boost paw_withdraw weight
- **Add more samples**: Collect more pain response examples
- **Use focal loss**: Better for rare classes

### Slow Inference
- **Increase stride**: stride=1 → stride=5
- **Use light model**: ~5x faster
- **Batch process**: Process multiple videos in parallel

### Training diverges (loss becomes NaN)
- **Lower learning rate**: 1e-3 → 5e-4
- **Reduce batch size**: 32 → 16
- **Check data**: Invalid frames or labels?

## Next Steps After Training

1. **Validate Results**
   - Check confusion matrix
   - Review per-class F1 scores
   - Visualize predictions on representative frames

2. **Iterate if Needed**
   - Adjust hyperparameters
   - Add data augmentation
   - Collect more training data for weak classes

3. **Prepare for Publication**
   - Save best model checkpoint
   - Document hyperparameters used
   - Create results tables/figures
   - Write methods section

4. **Deploy (Optional)**
   - Use `inference.py` for batch prediction
   - Create REST API wrapper if needed
   - Implement real-time prediction pipeline

## Contact & Support

If you encounter issues or have questions, check:
1. README.md - comprehensive usage guide
2. Code comments - implementation details
3. test_pipeline.py - verify your setup works

Good luck with your research!
