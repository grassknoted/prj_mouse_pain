# Mouse Pain Action Recognition - System Overview

## Architecture Diagram

```
                        TRAINING PIPELINE
                        ═══════════════════

    ┌─────────────────────────────────────────────────────────────┐
    │                     RAW DATA                                 │
    │  (1000 videos @ 30fps, 12s each + frame-level annotations)  │
    └────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              DATA LOADER (data_loader.py)                    │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │ • Load videos on-the-fly (memory efficient)              │ │
    │  │ • Extract 16-frame temporal clips                        │ │
    │  │ • Center-align clips around target frame                 │ │
    │  │ • Merge action classes (guard+flinch→withdraw)          │ │
    │  │ • Compute class weights (inverse frequency)              │ │
    │  │ • 80/20 train/val split at video level                  │ │
    │  └─────────────────────────────────────────────────────────┘ │
    └────────────┬────────────────────────────────────────────────┘
                 │
         ┌───────┴────────┐
         │                │
         ▼                ▼
    TRAIN SET        VAL SET
    (800 videos)     (200 videos)
         │                │
         │                │
         ▼                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              MODEL (model.py)                                │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │ 3D CNN ARCHITECTURE                                      │ │
    │  │                                                           │ │
    │  │  Input: (B, 1, 16, H, W) - grayscale 16-frame clips     │ │
    │  │    │                                                      │ │
    │  │    ├─→ Conv3D(1→32)  + BatchNorm + ReLU + MaxPool       │ │
    │  │    │   Output: (B, 32, 16, H/2, W/2)                    │ │
    │  │    │                                                      │ │
    │  │    ├─→ Conv3D(32→64) + BatchNorm + ReLU + MaxPool       │ │
    │  │    │   Output: (B, 64, 8, H/4, W/4)                     │ │
    │  │    │                                                      │ │
    │  │    ├─→ Conv3D(64→128) + BatchNorm + ReLU + MaxPool      │ │
    │  │    │   Output: (B, 128, 4, H/8, W/8)                    │ │
    │  │    │                                                      │ │
    │  │    ├─→ Conv3D(128→256) + BatchNorm + ReLU + MaxPool     │ │
    │  │    │   Output: (B, 256, 2, H/16, W/16)                  │ │
    │  │    │                                                      │ │
    │  │    ├─→ GlobalAvgPool3D                                  │ │
    │  │    │   Output: (B, 256)                                 │ │
    │  │    │                                                      │ │
    │  │    ├─→ FC(256→512) + ReLU + Dropout(0.5)                │ │
    │  │    │   Output: (B, 512)                                 │ │
    │  │    │                                                      │ │
    │  │    ├─→ FC(512→256) + ReLU + Dropout(0.5)                │ │
    │  │    │   Output: (B, 256)                                 │ │
    │  │    │                                                      │ │
    │  │    └─→ FC(256→7) [classification head]                  │ │
    │  │        Output: (B, 7) logits                            │ │
    │  │                                                           │ │
    │  │  3M parameters total                                     │ │
    │  │  Lightweight variant: 600K parameters                    │ │
    │  └─────────────────────────────────────────────────────────┘ │
    └────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │        TRAINING LOOP (train.py)                              │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │ For each epoch:                                          │ │
    │  │   1. Forward pass: logits = model(clips)                │ │
    │  │   2. Weighted loss: CE_loss(logits, labels, weights)    │ │
    │  │   3. Backward pass: loss.backward()                     │ │
    │  │   4. Gradient clipping: norm ≤ 1.0                      │ │
    │  │   5. Adam optimizer step                                │ │
    │  │   6. LR scheduler: Cosine annealing w/ restarts         │ │
    │  │   7. Validation & checkpointing                         │ │
    │  │   8. Early stopping: patience=15 epochs                 │ │
    │  │                                                           │ │
    │  │ Mixed precision training for 2x speedup                 │ │
    │  │ TensorBoard logging for monitoring                      │ │
    │  └─────────────────────────────────────────────────────────┘ │
    └────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │        EVALUATION (evaluation.py)                            │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │ Compute Metrics:                                         │ │
    │  │   • F1 score (macro, weighted, micro)      [KEY METRIC] │ │
    │  │   • Precision & Recall per class                        │ │
    │  │   • Confusion matrix visualization                      │ │
    │  │   • Per-class performance plots                         │ │
    │  │   • Classification report                              │ │
    │  │                                                           │ │
    │  │ Best Model Selection: Save by F1 (macro)                │ │
    │  └─────────────────────────────────────────────────────────┘ │
    └────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                  TRAINED MODEL                              │
    │  (saved checkpoint with best F1 score)                      │
    └────────────┬────────────────────────────────────────────────┘
                 │
                 ▼

                     INFERENCE PIPELINE
                     ═══════════════════

    ┌─────────────────────────────────────────────────────────────┐
    │              INFERENCE (inference.py)                        │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │ Load trained model checkpoint                           │ │
    │  │ For each frame in video:                                │ │
    │  │   1. Extract 16-frame clip around frame                │ │
    │  │   2. Normalize to [0,1]                                 │ │
    │  │   3. Forward pass through model                         │ │
    │  │   4. Get softmax probabilities                          │ │
    │  │   5. Return argmax class & confidence                   │ │
    │  │                                                           │ │
    │  │ Output: Frame-level predictions with confidence         │ │
    │  └─────────────────────────────────────────────────────────┘ │
    └────────────┬────────────────────────────────────────────────┘
                 │
         ┌───────┴────────┬─────────────┐
         │                │             │
         ▼                ▼             ▼
    Single Frame  Batch Videos    All Videos
    Prediction    Prediction      Prediction
         │                │             │
         │                │             │
         └────────────────┴─────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                PREDICTIONS (JSON)                           │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │ {                                                        │ │
    │  │   "video": "video_001.mp4",                             │ │
    │  │   "predictions": [                                      │ │
    │  │     {                                                   │ │
    │  │       "frame": 100,                                     │ │
    │  │       "action": "paw_withdraw",                         │ │
    │  │       "confidence": 0.87                                │ │
    │  │     },                                                  │ │
    │  │     ...                                                 │ │
    │  │   ]                                                     │ │
    │  │ }                                                        │ │
    │  └─────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. **Temporal Learning via 3D CNNs**
- Standard CNNs treat each frame independently
- 3D CNNs learn spatiotemporal features (video understanding)
- Critical for detecting short, dynamic behaviors like pain responses

### 2. **Class Weighting for Imbalance**
- Pain responses << Rest frames (~5% vs 50%)
- Inverse frequency weighting forces model to pay attention to rare classes
- Automatic computation from training data distribution

### 3. **Center-Aligned Temporal Clips**
- Predicts action at center frame using full context
- 8 frames before + center + 8 frames after = 0.5 seconds
- Balances temporal context vs computational cost

### 4. **Robust Training Practices**
- Mixed precision (2x speedup without quality loss)
- Gradient clipping (prevents exploding gradients)
- Batch normalization (training stability)
- Early stopping (prevents overfitting)
- Cosine annealing with restarts (escapes local minima)

### 5. **Publication-Quality Metrics**
- F1 macro (primary metric for imbalanced data)
- Per-class metrics (especially important for pain behaviors)
- Confusion matrix (debug model errors)

## Data Flow Example

### Single Sample Through Training
```
Video frame 100 (from video_001.mp4)
     ↓
[frames 92-108] ← 16-frame clip (center at 100)
     ↓
Normalize to [0,1], shape (1, 16, 256, 256)
     ↓
3D CNN Forward Pass
     ↓
Output logits: [0.1, 2.3, 0.4, ...]
     ↓
Label from annotation: "paw_withdraw" (class 1)
     ↓
Weighted CrossEntropy Loss (paw_withdraw gets higher weight)
     ↓
Backward pass → gradient update
```

## What Makes This Publication-Quality

1. **Solid Architecture**: 3D CNN is proven for action recognition
2. **Proper Handling of Class Imbalance**: Automatic class weighting
3. **Rigorous Evaluation**: Comprehensive metrics beyond accuracy
4. **Good Practices**: Early stopping, best model selection, gradient clipping
5. **Reproducibility**: Fixed seeds, config logging, documented hyperparameters
6. **Monitoring**: TensorBoard integration for real-time visualization

## Common Pitfalls Avoided

❌ **NOT DOING**: Frame-level predictions without temporal context
✓ **DOING**: 16-frame clips for proper temporal understanding

❌ **NOT DOING**: Simple accuracy metric for imbalanced data
✓ **DOING**: F1 macro and per-class metrics

❌ **NOT DOING**: Naive train/test split (data leakage)
✓ **DOING**: Video-level split (no data leakage)

❌ **NOT DOING**: Fixed learning rate throughout training
✓ **DOING**: Cosine annealing with warm restarts

❌ **NOT DOING**: Saving best model by accuracy
✓ **DOING**: Saving best model by F1 score

## Expected Performance Ranges

After training on 800 videos with good quality annotations:

| Metric | Expected Range | Notes |
|--------|-----------------|-------|
| Overall Accuracy | 80-90% | Not primary metric |
| **F1 (macro)** | **0.75-0.85** | **Most important** |
| paw_withdraw Recall | 0.70-0.85 | Don't miss pain events |
| paw_withdraw Precision | 0.80-0.90 | Avoid false positives |
| Training Time | 2-8 hours | Depends on GPU |
| Inference Speed | 30-60 FPS | Per-frame prediction |

## Customization Points

If results are not satisfactory, adjust:

| Issue | Solution |
|-------|----------|
| Low pain detection | Increase paw_withdraw class weight |
| Overfitting | Increase dropout, use light model, add augmentation |
| Slow training | Increase batch size, reduce clip_length |
| Memory issues | Reduce batch_size or use light model |
| Poor generalization | Add data augmentation, increase L2 regularization |

See `IMPLEMENTATION_SUMMARY.md` for more details.
