# Files Guide - What Each File Does

Complete guide to understanding the repository structure and what each file is responsible for.

---

## Quick Navigation

- [Python Files](#python-files)
  - [Approach 1: Video-Only](#approach-1-video-only)
  - [Approach 2: Multimodal](#approach-2-multimodal)
  - [Approach 3: Multi-Task](#approach-3-multi-task-advanced)
  - [Testing & Utilities](#testing--utilities)
- [Shell Scripts](#shell-scripts)
- [Documentation](#documentation)
- [Configuration](#configuration)

---

## Python Files

### Approach 1: Video-Only

**Use Case**: Quick experiments, baseline 3D CNN approach

#### `train.py` (14KB)
**Purpose**: Training script for video-only 3D CNN model

**Features**:
- Weighted CrossEntropy loss
- Adam optimizer with cosine annealing
- Mixed precision training (AMP)
- Gradient clipping
- Early stopping
- TensorBoard logging
- Model checkpointing (saves best by F1)

**Run**:
```bash
python train.py
```

**Output**: `./checkpoints/run_YYYYMMDD_HHMMSS/best_model_epochN.pt`

---

#### `model.py` (14KB)
**Purpose**: 3D CNN architecture definitions

**Classes**:
- `Conv3DBlock`: Reusable 3D conv + batch norm + activation block
- `Mouse3DCNN`: Standard model (3M parameters, 4 conv blocks)
- `Mouse3DCNNLight`: Lightweight variant (600K parameters, 3 conv blocks)
- `create_model()`: Factory function

**Architecture**:
```
Input (B, 1, 16, H, W) → Conv3D(32→64→128→256) → GlobalAvgPool
  → FC(512→256) → FC(7) → Logits
```

**Use When**: Need model for training or inference

---

#### `data_loader.py` (19KB)
**Purpose**: Data loading and preprocessing pipeline

**Classes**:
- `MouseActionDataset`: Loads videos and creates temporal clips
- `create_data_loaders()`: Creates train/val dataloaders

**Features**:
- On-the-fly video loading (memory efficient)
- 16-frame temporal clips centered on target frame
- Action class merging (paw_guard+flinch → paw_withdraw)
- Inverse-frequency class weight computation
- 80/20 train/val split at video level
- Frame validation

**Use When**: Initializing data for training

---

#### `inference.py` (7.5KB)
**Purpose**: Make predictions on new videos

**Classes**:
- `ActionRecognitionInference`: Main inference wrapper

**Methods**:
- `predict_frame()`: Single frame prediction
- `predict_video()`: Full video frame-by-frame predictions
- `save_predictions()`: Save results to JSON
- `predict_multiple_videos()`: Batch prediction

**Example**:
```python
from inference import ActionRecognitionInference

inference = ActionRecognitionInference("./checkpoints/run_xxx/best_model.pt")
predictions = inference.predict_video("./Videos/video_1.mp4", stride=2)
```

---

#### `evaluation.py` (8.6KB)
**Purpose**: Compute metrics and create visualizations

**Functions**:
- `evaluate()`: Compute metrics on validation set
- `compute_metrics()`: F1, precision, recall per class
- `plot_confusion_matrix()`: Confusion matrix heatmap
- `plot_per_class_metrics()`: F1/precision/recall bar plots
- `evaluate_and_save_results()`: Full evaluation pipeline

**Metrics**:
- F1 (macro, weighted, micro)
- Precision & recall per class
- Confusion matrix
- Classification report

**Use When**: Evaluating trained models, creating publication figures

---

### Approach 2: Multimodal

**Use Case**: Have DeepLabCut pose data, want better pain detection

#### `multimodal_train.py` (16KB)
**Purpose**: Training script for dual-stream (visual + pose) model

**Features**:
- Same as `train.py` but loads visual + pose data
- Dual-stream optimization
- Multi-GPU support (DDP)

**Run**:
```bash
python multimodal_train.py
```

**Data Requirements**:
- Videos in `Videos/`
- Action annotations in `Annotations/`
- DLC CSVs in `DLC/`

---

#### `multimodal_model.py` (20KB)
**Purpose**: Dual-stream architecture

**Classes**:
- `PoseStream`: MLP + temporal attention for pose features
- `MultimodalModel`: Combined visual + pose architecture
- `create_multimodal_model()`: Factory function

**Architecture**:
```
Visual: (B,1,16,H,W) → 3D CNN → 256D
Pose: (B,16,18) → MLP+Attention → 256D
  ↓
Fusion: Concat(512) → MLP → 7 classes
```

---

#### `multimodal_data_loader.py` (31KB)
**Purpose**: Load visual + pose data

**Features**:
- Loads videos AND DLC keypoint CSVs
- Extracts 18 pose features (8 edges + 10 angles)
- Synchronized visual + pose temporal clips
- Handles missing/low-confidence keypoints

**Use When**: Training or evaluating multimodal models

---

#### `multimodal_inference.py` (11KB)
**Purpose**: Inference for multimodal models

**Classes**:
- `MultimodalActionRecognitionInference`: Inference wrapper

**Requires**: Both video + DLC CSV for prediction

**Example**:
```python
from multimodal_inference import MultimodalActionRecognitionInference

inference = MultimodalActionRecognitionInference("checkpoint.pt")
predictions = inference.predict_video(
    "video.mp4",
    "video_dlc.csv",
    stride=2
)
```

---

#### `pose_graph.py` (2.4KB)
**Purpose**: Kinematic feature extraction from keypoints

**Class**:
- `PoseGraph`: Computes geometric features

**Features Extracted**:
- 8 edge lengths (normalized distances)
- 10 angles (joint angles in radians)
- Total: 18 features per frame

**Keypoint Order**: `[mouth, tail_base, L_front, R_front, L_hind, R_hind]`

---

### Approach 3: Multi-Task (Advanced)

**Use Case**: State-of-the-art performance, production deployment

#### `train_multitask.py` (96KB) ⭐
**Purpose**: Comprehensive multi-task training pipeline

**This is the most advanced training script with everything.**

**Features**:
- **Backbones**: VideoMAE2 (3D) or ViT (2D + temporal pooling)
- **Temporal Head**: TCN with dilated convolutions
- **Model Types**: Action-only or multi-task (action + keypoint regression)
- **Pose Graph**: 18 geometric features from 6 keypoints
- **Advanced Training**:
  - Focal loss (gamma=2.5)
  - Rare-class boosting (up to 30x)
  - Label smoothing (0.1)
  - Comprehensive augmentation (brightness, contrast, temporal dropout, keypoint jitter)
  - EMA (exponential moving average)
  - Early stopping
  - Warmup + backbone freezing
- **Data Handling**:
  - 345 or 360-frame annotations
  - Variable trial lengths
  - Invalid trial skipping
  - Robust DLC parsing
- **Monitoring**:
  - W&B logging with auto-generated run names
  - tqdm progress bars
  - Detailed metrics logging

**Run (Action-Only)**:
```bash
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --model_type action_only \
    --epochs 50 \
    --batch_size 2
```

**Run (Multi-Task)**:
```bash
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --model_type multitask \
    --epochs 50 \
    --batch_size 2 \
    --use_wandb
```

**Output**: `./best_model_multitask.pt`

**Detailed Guide**: See [MULTITASK_GUIDE.md](MULTITASK_GUIDE.md)

---

### Testing & Utilities

#### `test_pipeline.py` (5.8KB)
**Purpose**: Verify video-only setup

**Tests**:
1. Data loading (videos + annotations)
2. Model creation and forward pass
3. Metrics computation
4. Lightweight model variant

**Run**:
```bash
python test_pipeline.py
```

**Expected Output**:
```
✓ Data loading test PASSED
✓ Model architecture test PASSED
✓ Metrics test PASSED
✓ Lightweight Model test PASSED
```

---

#### `test_updated_loaders.py` (3.8KB)
**Purpose**: Test data loaders after updates

**Run**:
```bash
python test_updated_loaders.py
```

---

#### `test_multitask_data.py` (2.8KB)
**Purpose**: Test multi-task data loading

**Validates**:
- Video loading
- Action CSV loading
- DLC CSV loading
- Pose graph computation
- Trial discovery

**Run**:
```bash
python test_multitask_data.py
```

---

#### `validate_data_structure.py` (3.9KB)
**Purpose**: Validate data organization

**Checks**:
- Video files exist
- Annotations match videos
- Frame counts are correct
- DLC files present (if needed)

**Run**:
```bash
python validate_data_structure.py
```

---

#### `validate_simple.py` (3.3KB)
**Purpose**: Simple validation script

**Run**:
```bash
python validate_simple.py
```

---

#### `debug_dlc_csv.py` (2.1KB)
**Purpose**: Inspect and debug DLC CSV files

**Usage**:
```bash
python debug_dlc_csv.py path/to/dlc.csv
```

**Output**:
- Header structure
- Keypoint names
- Coordinate ranges
- Missing values
- Likelihood statistics

---

#### `extract_frames.py` (4.2KB)
**Purpose**: Extract frames from videos

**Usage**:
```bash
python extract_frames.py video.mp4 output_dir/
```

---

#### `check_annotations.py` (2.5KB)
**Purpose**: Check annotation file validity

**Validates**:
- CSV format
- Frame indices
- Action values in range
- Row count

---

#### Preprocessing Scripts (Legacy)

- `new_preprocess_data.py` (5.7KB): Old preprocessing
- `preprocess_final.py` (7.3KB): Final preprocessing version
- `reorganize_data.py` (9.6KB): Reorganize data structure

**Note**: These are legacy scripts. Current pipelines handle preprocessing automatically.

---

## Shell Scripts

### `train_multigpu.sh` (1.9KB)
**Purpose**: Multi-GPU training for video-only

**Usage**:
```bash
bash train_multigpu.sh
```

Uses `torch.distributed` for multi-GPU training.

---

### `train_multimodal_multigpu.sh` (2.0KB)
**Purpose**: Multi-GPU training for multimodal

**Usage**:
```bash
bash train_multimodal_multigpu.sh
```

---

### `extract_all_frames.sh` (853B)
**Purpose**: Batch extract frames from all videos

**Usage**:
```bash
bash extract_all_frames.sh
```

---

## Documentation

### Main Guides

| File | Lines | Purpose | Read When |
|------|-------|---------|-----------|
| **README.md** | 515 | Overview of all approaches | Starting out |
| **QUICKSTART.md** | 253 | 5-min video-only tutorial | Quick start |
| **MULTIMODAL_QUICKSTART.md** | 171 | 5-min multimodal tutorial | Have pose data |
| **MULTIMODAL_GUIDE.md** | 419 | Comprehensive multimodal guide | Deep dive multimodal |
| **MULTITASK_GUIDE.md** | 670 | Comprehensive multi-task guide | Using advanced approach |
| **FILES_GUIDE.md** | This file | What each file does | Understanding codebase |
| **SYSTEM_OVERVIEW.md** | 341 | Architecture diagrams | Visual learner |
| **IMPLEMENTATION_SUMMARY.md** | 250 | Design decisions | Understanding why |
| **DELIVERABLES.md** | 442 | Implementation checklist | Project completion |
| **CHANGELOG.md** | 750 | Technical fixes & updates | Troubleshooting |

### Technical Documentation (Archive)

These files are now consolidated into CHANGELOG.md:

- `CLIP_LENGTH_FIX.md` → See CHANGELOG.md
- `DATA_STATISTICS_EXPLANATION.md` → See CHANGELOG.md
- `FIXES_SUMMARY.md` → See CHANGELOG.md
- `GRADIENT_TRACKING_FIX.md` → See CHANGELOG.md
- `MULTITASK_IMPROVEMENTS.md` → See CHANGELOG.md
- `MULTITASK_TRAINING_FIXED.md` → See CHANGELOG.md
- `MULTI_GPU_TRAINING.md` → See CHANGELOG.md
- `MULTI_TRIAL_UPDATE.md` → See CHANGELOG.md
- `POSE_GRAPH_AND_MODEL_TYPES.md` → See CHANGELOG.md
- `SKIPPING_INVALID_TRIALS.md` → See CHANGELOG.md
- `TORCH_UINT8_FIX.md` → See CHANGELOG.md
- `TQDM_PROGRESS_BARS.md` → See CHANGELOG.md
- `VARIABLE_TRIAL_LENGTHS.md` → See CHANGELOG.md
- `VIDEOMAE_HUGGINGFACE_UPDATE.md` → See CHANGELOG.md
- `WANDB_LOGGING.md` → See CHANGELOG.md

**Recommendation**: These files can be archived or deleted as their content is now in [CHANGELOG.md](CHANGELOG.md).

---

## Configuration

### `requirements.txt` (156B)
**Purpose**: Python package dependencies

**Core Dependencies**:
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.14.0
```

**Optional**:
```
transformers  # For VideoMAE2
wandb         # For experiment tracking
timm          # For ViT backbones (multi-task)
```

**Install**:
```bash
pip install -r requirements.txt
pip install transformers wandb timm  # Optional
```

---

### `.gitignore`
**Purpose**: Git ignore rules

Ignores:
- `checkpoints/`, `checkpoints_multimodal/`
- `__pycache__/`, `*.pyc`
- `wandb/`
- `*.pt`, `*.pth` (model checkpoints)
- `Videos/`, `Annotations/`, `DLC/` (data directories)

---

## File Dependency Graph

```
DATA (Videos/, Annotations/, DLC/)
  │
  ├─→ APPROACH 1: Video-Only
  │   └─→ data_loader.py → model.py → train.py → evaluation.py
  │       └─→ inference.py
  │
  ├─→ APPROACH 2: Multimodal
  │   └─→ pose_graph.py → multimodal_data_loader.py → multimodal_model.py
  │       └─→ multimodal_train.py → evaluation.py
  │       └─→ multimodal_inference.py
  │
  └─→ APPROACH 3: Multi-Task
      └─→ train_multitask.py (self-contained, includes pose graph)
```

---

## Typical Workflows

### Workflow 1: Video-Only Training

```
1. Install dependencies: pip install -r requirements.txt
2. Organize data: Videos/, Annotations/
3. Validate: python test_pipeline.py
4. Train: python train.py
5. Monitor: tensorboard --logdir=./checkpoints
6. Inference: python inference.py
7. Evaluate: python evaluation.py
```

**Files Used**:
- `requirements.txt`
- `test_pipeline.py`
- `train.py`, `model.py`, `data_loader.py`
- `inference.py`
- `evaluation.py`

---

### Workflow 2: Multimodal Training

```
1. Install dependencies: pip install -r requirements.txt
2. Organize data: Videos/, Annotations/, DLC/
3. Train: python multimodal_train.py
4. Monitor: tensorboard --logdir=./checkpoints_multimodal
5. Inference: python multimodal_inference.py
6. Evaluate: python evaluation.py
```

**Files Used**:
- `requirements.txt`
- `multimodal_train.py`, `multimodal_model.py`, `multimodal_data_loader.py`
- `pose_graph.py`
- `multimodal_inference.py`
- `evaluation.py`

---

### Workflow 3: Multi-Task Training

```
1. Install all dependencies: pip install -r requirements.txt transformers wandb
2. Organize data: Videos/, Annotations/ (DLC CSVs in Videos/)
3. Test data: python test_multitask_data.py
4. Train: python train_multitask.py --use_wandb
5. Monitor: wandb dashboard OR local metrics
6. Use checkpoint: ./best_model_multitask.pt
```

**Files Used**:
- `requirements.txt`
- `test_multitask_data.py`
- `train_multitask.py` (everything else is inside)

---

## Which File Should I Use?

| Question | File/Command |
|----------|--------------|
| Start training (video-only) | `python train.py` |
| Start training (multimodal) | `python multimodal_train.py` |
| Start training (advanced) | `python train_multitask.py` |
| Make predictions (video-only) | `inference.py` |
| Make predictions (multimodal) | `multimodal_inference.py` |
| Compute metrics | `evaluation.py` |
| Test setup | `test_pipeline.py` |
| Debug DLC files | `debug_dlc_csv.py` |
| Multi-GPU training | `train_multigpu.sh` or `train_multimodal_multigpu.sh` |
| Check data validity | `validate_data_structure.py` |
| Extract video frames | `extract_frames.py` |
| Understand architecture | `model.py` or `multimodal_model.py` |
| Understand data loading | `data_loader.py` or `multimodal_data_loader.py` |

---

## File Sizes & Complexity

| File | Size | Complexity | Lines |
|------|------|------------|-------|
| `train_multitask.py` | 96KB | ⭐⭐⭐⭐⭐ | ~1850 |
| `multimodal_data_loader.py` | 31KB | ⭐⭐⭐⭐ | ~750 |
| `multimodal_model.py` | 20KB | ⭐⭐⭐ | ~500 |
| `data_loader.py` | 19KB | ⭐⭐⭐ | ~450 |
| `multimodal_train.py` | 16KB | ⭐⭐⭐ | ~400 |
| `model.py` | 14KB | ⭐⭐ | ~350 |
| `train.py` | 14KB | ⭐⭐⭐ | ~350 |
| `multimodal_inference.py` | 11KB | ⭐⭐ | ~275 |
| `evaluation.py` | 8.6KB | ⭐⭐ | ~220 |
| `inference.py` | 7.5KB | ⭐⭐ | ~190 |

---

## Output Files Generated

### After Training

**Video-Only**:
```
checkpoints/run_YYYYMMDD_HHMMSS/
├── best_model_epochN.pt
├── config.json
└── events.out.tfevents.*  (TensorBoard logs)
```

**Multimodal**:
```
checkpoints_multimodal/run_YYYYMMDD_HHMMSS/
├── best_model_epochN.pt
├── config.json
└── events.out.tfevents.*
```

**Multi-Task**:
```
./best_model_multitask.pt
wandb/run-YYYYMMDD_HHMMSS-*/  (if --use_wandb)
```

### After Inference

```
predictions/
├── video_001_predictions.json
├── video_002_predictions.json
└── ...
```

### After Evaluation

```
results/
├── metrics.txt
├── confusion_matrix.png
├── per_class_metrics.png
└── class_distribution.png
```

---

## Summary

**Total Python Files**: 20+
**Total Documentation**: 10 main + 15 archived
**Total Lines of Code**: ~7000+
**Total Documentation Lines**: ~4000+

**Most Important Files**:
1. `README.md` - Start here
2. `train.py` / `multimodal_train.py` / `train_multitask.py` - Training
3. `model.py` / `multimodal_model.py` - Architectures
4. `data_loader.py` / `multimodal_data_loader.py` - Data pipelines
5. `evaluation.py` - Metrics & analysis

**For Quick Start**:
- Beginners → [QUICKSTART.md](QUICKSTART.md)
- Have pose data → [MULTIMODAL_QUICKSTART.md](MULTIMODAL_QUICKSTART.md)
- Advanced users → [MULTITASK_GUIDE.md](MULTITASK_GUIDE.md)

---

Good luck with your mouse pain detection research! 🐭🔬
