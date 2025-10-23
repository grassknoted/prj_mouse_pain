# Files Guide - What Each File Does

## Core Pipeline Files

### `data_loader.py` - Data Loading & Preprocessing
**Purpose**: Loads videos and annotations, creates temporal training clips

**Key Classes**:
- `MouseActionDataset`: Loads video clips and returns (temporal_clip, label) pairs
- Functions: `create_data_loaders()` to create train/val dataloaders

**What it handles**:
- ✓ Loads videos frame-by-frame on demand (memory efficient)
- ✓ Extracts 16-frame temporal clips centered on target frame
- ✓ Merges action classes (paw_guard + flinch → paw_withdraw)
- ✓ Validates frame counts against annotations
- ✓ Computes inverse-frequency class weights
- ✓ Creates 80/20 train/val split at video level
- ✓ Normalizes frames to [0,1]

**Use when**:
- Initializing training
- Creating dataloaders in custom scripts
- Debugging data issues

**Example**:
```python
from data_loader import create_data_loaders
train_loader, val_loader, class_weights = create_data_loaders(
    "./Videos", "./Annotations", batch_size=32
)
```

---

### `model.py` - Neural Network Architecture
**Purpose**: Defines 3D CNN models for action recognition

**Key Classes**:
- `Conv3DBlock`: Reusable 3D conv + batch norm + activation
- `Mouse3DCNN`: Full 4-layer 3D CNN (3M parameters)
- `Mouse3DCNNLight`: Lightweight variant (600K parameters)
- Function: `create_model()` factory function

**Architecture Highlights**:
- 4 blocks of 3D convolutions (32→64→128→256 channels)
- Batch normalization and ReLU activations
- Max pooling for spatial/temporal downsampling
- Global average pooling for robustness
- 2 fully connected layers with dropout

**Use when**:
- Training a model
- Loading a saved checkpoint for inference
- Experimenting with architecture changes

**Example**:
```python
from model import create_model
model = create_model(num_classes=7, clip_length=16, model_type="standard")
```

---

### `train.py` - Training Script
**Purpose**: Complete training pipeline with all best practices

**Key Features**:
- ✓ Weighted CrossEntropy loss (handles class imbalance)
- ✓ Adam optimizer with cosine annealing + warm restarts
- ✓ Automatic mixed precision (AMP) for 2x speedup
- ✓ Gradient clipping (max norm 1.0)
- ✓ Early stopping (patience=15)
- ✓ Model checkpointing (saves best by F1 score)
- ✓ TensorBoard logging for monitoring
- ✓ Configuration saving (reproducibility)

**Use when**:
- Training from scratch
- Fine-tuning an existing model
- Running the complete training pipeline

**Run from command line**:
```bash
python train.py
```

**Or programmatically**:
```python
from train import train_model
run_dir = train_model(
    video_dir="./Videos",
    annotation_dir="./Annotations",
    num_epochs=100,
    batch_size=32
)
```

---

### `evaluation.py` - Evaluation & Metrics
**Purpose**: Compute comprehensive evaluation metrics and create visualizations

**Key Functions**:
- `evaluate()`: Compute all metrics on validation set
- `compute_metrics()`: Calculate F1, precision, recall per class
- `print_detailed_report()`: Sklearn classification report
- `plot_confusion_matrix()`: Confusion matrix heatmap
- `plot_per_class_metrics()`: Bar plots of precision/recall/F1
- `plot_class_distribution()`: Class distribution histogram
- `evaluate_and_save_results()`: Full evaluation pipeline

**Metrics Computed**:
- F1 (macro, weighted, micro)
- Precision & recall per class
- Confusion matrix
- Per-class F1 scores
- Classification report

**Use when**:
- Evaluating trained models
- Creating publication-quality figures
- Analyzing per-class performance
- Debugging model confusion patterns

**Example**:
```python
from evaluation import evaluate_and_save_results
evaluate_and_save_results(model, val_loader, device, "./results")
```

---

### `inference.py` - Make Predictions
**Purpose**: Load trained models and make predictions on new videos

**Key Classes**:
- `ActionRecognitionInference`: Main inference wrapper

**Key Methods**:
- `predict_frame()`: Single frame prediction
- `predict_video()`: Full video frame-by-frame predictions
- `save_predictions()`: Save results to JSON

**Function**:
- `predict_multiple_videos()`: Batch prediction on many videos

**Use when**:
- Making predictions after training
- Analyzing specific videos
- Running inference pipeline on test set
- Getting confidence scores for predictions

**Example**:
```python
from inference import ActionRecognitionInference
inference = ActionRecognitionInference("./checkpoints/run_xxx/best_model.pt")
predictions = inference.predict_video("./Videos/video_1.mp4", stride=2)
```

---

## Documentation Files

### `README.md` - Comprehensive Documentation
**Read this for**: Complete guide covering everything

**Sections**:
- Overview of the project
- Installation instructions
- Data format specification
- Quick start examples
- Model architecture details
- Training procedure
- Evaluation metrics
- Temporal clip strategy explanation
- Tips for best results
- Troubleshooting guide
- Advanced usage examples

**When to read**: Before starting, or when you have specific questions

---

### `QUICKSTART.md` - 5-Minute Start Guide
**Read this for**: Getting running as fast as possible

**Sections**:
- Prerequisites checklist
- Install (2 minutes)
- Verify setup (1 minute)
- Train model
- Get predictions
- Evaluate results
- Common adjustments
- Expected output
- Troubleshooting

**When to read**: When you just want to run the code

---

### `IMPLEMENTATION_SUMMARY.md` - Design Rationale
**Read this for**: Understanding WHY things are designed this way

**Sections**:
- Why 3D CNN instead of 2D
- Why class weighting matters
- Why 16-frame clips
- Why temporal clipping strategy
- Why global average pooling
- Why cosine annealing
- Design decisions & tradeoffs
- Performance expectations
- Advanced modifications

**When to read**: When you want to understand the thinking behind the implementation

---

### `SYSTEM_OVERVIEW.md` - Visual Architecture
**Read this for**: Seeing how all components fit together

**Sections**:
- Full architecture diagram (training + inference)
- Key design principles
- Data flow example
- What makes it publication-quality
- Common pitfalls avoided
- Expected performance ranges
- Customization points

**When to read**: Getting a bird's-eye view of the system

---

### `FILES_GUIDE.md` - This File
**Read this for**: Understanding what each file does

---

## Testing & Debugging

### `test_pipeline.py` - System Verification
**Purpose**: Quick tests to verify your setup works

**Tests**:
1. Data loading (can load videos/annotations)
2. Model creation (model builds without errors)
3. Metrics computation (metrics work correctly)
4. Light model variant (memory-efficient model works)

**Use when**:
- Setting up for the first time
- Debugging data issues
- Verifying dependencies installed correctly

**Run**:
```bash
python test_pipeline.py
```

**Output example**:
```
✓ Data loading test PASSED
✓ Model architecture test PASSED
✓ Metrics test PASSED
✓ Lightweight Model test PASSED
✓ All tests passed! Ready to train.
```

---

## Configuration & Dependencies

### `requirements.txt` - Python Dependencies
**Purpose**: Lists all required packages and versions

**Packages**:
- `torch`, `torchvision`: Deep learning framework
- `opencv-python`: Video reading
- `numpy`, `pandas`: Data manipulation
- `scikit-learn`: Metrics & preprocessing
- `matplotlib`, `seaborn`: Visualization
- `tensorboard`: Training monitoring

**Use when**:
```bash
pip install -r requirements.txt
```

---

## File Dependencies Graph

```
┌──────────────────────────────────────────┐
│         Data Files                       │
│  (Videos/ and Annotations/)              │
└───────────────────────┬──────────────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │   data_loader.py       │
           │  (loads data)          │
           └─────────┬──────────────┘
                     │
         ┌───────────┴──────────────┐
         │                          │
         ▼                          ▼
    ┌─────────────┐         ┌──────────────┐
    │ train.py    │         │ inference.py │
    │(training)   │         │(predictions) │
    └──────┬──────┘         └──────┬───────┘
           │                       │
           └───────────┬───────────┘
                       │
         ┌─────────────▼─────────────┐
         │  model.py                 │
         │  (neural network)         │
         └─────────────┬─────────────┘
                       │
         ┌─────────────▼─────────────┐
         │  evaluation.py            │
         │  (metrics & plots)        │
         └───────────────────────────┘
```

---

## Typical Workflow

### First Time Setup
```
1. Read QUICKSTART.md
2. pip install -r requirements.txt
3. Run test_pipeline.py
4. Run train.py
```

### During Training
```
1. Monitor with: tensorboard --logdir=./checkpoints
2. Watch loss decrease and F1 increase
3. Training auto-stops with early stopping
```

### After Training
```
1. Load best model from checkpoints/
2. Run inference.py on validation videos
3. Use evaluation.py to get metrics & plots
```

### For Understanding
```
1. Read SYSTEM_OVERVIEW.md for architecture
2. Read IMPLEMENTATION_SUMMARY.md for why decisions
3. Read README.md for comprehensive details
4. Look at code comments for implementation details
```

---

## Which File Should I Look At?

| Question | File |
|----------|------|
| How do I get started? | `QUICKSTART.md` |
| How does the system work? | `SYSTEM_OVERVIEW.md` |
| Why were things designed this way? | `IMPLEMENTATION_SUMMARY.md` |
| What's the full documentation? | `README.md` |
| How do I train a model? | `train.py` |
| How do I make predictions? | `inference.py` |
| How do I evaluate results? | `evaluation.py` |
| How is data loaded? | `data_loader.py` |
| What's the model architecture? | `model.py` |
| Is my setup correct? | `test_pipeline.py` |
| What packages do I need? | `requirements.txt` |

---

## File Sizes & Complexity

| File | Lines | Purpose | Complexity |
|------|-------|---------|------------|
| `data_loader.py` | ~300 | Data pipeline | ⭐⭐⭐ |
| `model.py` | ~250 | Neural network | ⭐⭐ |
| `train.py` | ~300 | Training loop | ⭐⭐⭐ |
| `evaluation.py` | ~400 | Metrics & plots | ⭐⭐ |
| `inference.py` | ~250 | Predictions | ⭐⭐ |
| `test_pipeline.py` | ~200 | Testing | ⭐ |

Total: ~1700 lines of well-documented production code.

---

## Generated Output Files

After running, you'll get:

```
checkpoints/
├── run_20240101_120000/          ← One per training run
│   ├── best_model_epoch50.pt     ← Load this for inference
│   ├── config.json               ← Hyperparameters used
│   └── events.out.tfevents.*     ← TensorBoard logs

predictions/
├── video_001_predictions.json    ← Frame-level predictions
├── video_002_predictions.json
└── ...

results/
├── metrics.txt                   ← Detailed metrics
├── confusion_matrix.png          ← Confusion visualization
├── per_class_metrics.png         ← F1/precision/recall plots
└── class_distribution.png        ← Class histogram
```

---

## Best Practices for File Organization

```
prj_mouse_pain/
├── Videos/                  ← Your 1000 videos here
├── Annotations/             ← Your 1000 CSV files here
├── checkpoints/             ← Generated during training
├── predictions/             ← Generated during inference
├── results/                 ← Generated during evaluation
│
├── data_loader.py           ← Core pipeline files
├── model.py
├── train.py
├── evaluation.py
├── inference.py
├── test_pipeline.py
│
├── requirements.txt         ← Dependencies
├── README.md                ← Main documentation
├── QUICKSTART.md            ← 5-minute guide
├── IMPLEMENTATION_SUMMARY.md ← Design rationale
├── SYSTEM_OVERVIEW.md       ← Architecture diagrams
└── FILES_GUIDE.md           ← This file
```

Good luck with your research!
