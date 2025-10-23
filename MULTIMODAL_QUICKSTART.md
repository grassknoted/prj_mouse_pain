# Multimodal Model - Quick Start

## 30-Second Overview

Your multimodal model combines:
- **Visual stream**: Raw video frames (3D CNN)
- **Pose stream**: Joint movements from DLC (MLP + temporal attention)
- **Fusion**: Learned combination for action prediction

**Expected improvement**: +5-8% F1 on pain detection (most important!)

## File Checklist

Ensure your data structure is:
```
prj_mouse_pain/
├── Videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ... (1000 videos)
├── Annotations/
│   ├── video_001_*.csv
│   ├── video_002_*.csv
│   └── ... (1000 CSVs with action labels)
└── DLC/
    ├── video_001_*.csv
    ├── video_002_*.csv
    └── ... (1000 CSVs with DLC coordinates)
```

## Training (3 Lines of Code)

```python
from multimodal_train import train_multimodal_model

train_multimodal_model(
    video_dir="./Videos",
    annotation_dir="./Annotations",
    dlc_dir="./DLC",
    num_epochs=100,
    batch_size=32
)
```

Or just run:
```bash
python multimodal_train.py
```

## Inference (3 Lines of Code)

```python
from multimodal_inference import MultimodalActionRecognitionInference

inference = MultimodalActionRecognitionInference("checkpoints_multimodal/run_xxx/best_model.pt")
predictions = inference.predict_video("video.mp4", "video_dlc.csv", stride=2)
```

## Expected Results

| Metric | Visual-Only | Multimodal |
|--------|-------------|-----------|
| F1 (macro) | 0.78 | 0.82 |
| paw_withdraw F1 | 0.73 | 0.78 |
| Training time | 4h | 5h |

## Key Differences from Visual-Only

### Data Loading
```python
# Old (visual-only)
visual_clip, label = dataset[idx]

# New (multimodal)
visual_clip, pose_clip, label = dataset[idx]
```

### Model
```python
# Old: 1 input, 1 stream
output = model(visual)

# New: 2 inputs, 2 streams, fusion
output = model(visual, pose)
```

### Pose Features
- 8 edge lengths (joint distances)
- 10 angles (joint angles)
- Computed automatically from DLC coordinates
- Z-score normalized

## File Organization

**Old (Visual-Only)**:
```
data_loader.py → model.py → train.py → inference.py
```

**New (Multimodal)**:
```
multimodal_data_loader.py → multimodal_model.py
         ↓ (uses pose_graph.py)
multimodal_train.py → multimodal_inference.py
```

## Monitoring Training

```bash
tensorboard --logdir=./checkpoints_multimodal
```

Open http://localhost:6006 to see:
- Loss curves (train/val)
- Accuracy curves
- F1 score progression

## Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| DLC file not found | Check DLC/ folder, verify filenames match videos |
| Shape mismatch | Ensure DLC CSVs have 2-row headers (standard DeepLabCut) |
| OOM error | Reduce batch_size to 16 or use "light" model |
| Poor results | Check annotation quality, verify DLC tracking is good |

## Batch Prediction

```python
from multimodal_inference import predict_multiple_videos_multimodal
from pathlib import Path

videos = list(Path("Videos").glob("*.mp4"))
dlcs = [Path("DLC") / f"{v.stem}_dlc.csv" for v in videos]

predict_multiple_videos_multimodal(
    checkpoint_path="checkpoint.pt",
    video_paths=videos,
    dlc_paths=dlcs,
    stride=2
)
```

## Evaluation

```python
# After inference, use existing evaluation.py
from evaluation import evaluate_and_save_results

evaluate_and_save_results(model, val_loader, device, "./results")
```

## Hyperparameters

**Default (good for most cases)**:
- batch_size: 32
- learning_rate: 1e-3
- clip_length: 16
- dropout: 0.5

**For memory constraints**:
- batch_size: 16 (reduce)
- model_type: "light" (use lightweight)
- clip_length: 8 (shorter clips)

## Publication Checklist

- [ ] Train multimodal and visual-only models
- [ ] Compare F1 scores
- [ ] Show improvement in pain response detection
- [ ] Plot confusion matrices
- [ ] Discuss why multimodal helps
- [ ] Report hyperparameters used
- [ ] Optional: analyze attention weights

## Next Steps

1. **Prepare data**: Organize Videos/, Annotations/, DLC/ folders
2. **Train**: Run `python multimodal_train.py`
3. **Monitor**: Check TensorBoard for convergence
4. **Evaluate**: Analyze results, compare with visual-only
5. **Publish**: Report metrics and improvements

## Detailed Docs

- **Full guide**: See `MULTIMODAL_GUIDE.md`
- **Architecture details**: See `multimodal_model.py`
- **Data loading**: See `multimodal_data_loader.py`
- **Code examples**: See `multimodal_train.py` and `multimodal_inference.py`

## Need Help?

1. Check `MULTIMODAL_GUIDE.md` (comprehensive)
2. Review code comments in `multimodal_*.py`
3. See troubleshooting section for common issues
4. Verify data format matches specifications

Good luck! 🚀
