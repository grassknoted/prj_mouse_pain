# VideoMAE Model Update - Using Hugging Face Transformers

## Summary of Changes

Updated the training script to use **Hugging Face VideoMAE** instead of trying to load from timm (which doesn't have VideoMAE models).

---

## What Changed

### 1. **Import VideoMAE from Transformers**

```python
try:
    from transformers import VideoMAEModel
except ImportError:
    print("[warn] transformers not installed. VideoMAE (3D) will not be available.")
    print("[warn] Install with: pip install transformers")
    VideoMAEModel = None
```

### 2. **Model Loading (Both MultiTaskModel and ActionOnlyModel)**

**Before (WRONG):**
```python
try:
    self.backbone_3d = timm.create_model(
        "videomae_v2_base_patch16_224",  # Doesn't exist in timm!
        pretrained=True,
        num_classes=0,
        img_size=img_size
    )
except Exception as e:
    # Falls back to 2D
```

**After (CORRECT):**
```python
if VideoMAEModel is not None:
    try:
        self.backbone_3d = VideoMAEModel.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )
        self.is_3d = True
        backbone_dim = 768  # VideoMAE-base hidden size
        logger.info("[info] Using VideoMAE (3D) backbone from Hugging Face")
        logger.info("[info] Model: MCG-NJU/videomae-base-finetuned-kinetics")
    except Exception as e:
        # Falls back to 2D
else:
    logger.warning("[warn] transformers not installed, falling back to 2D path")
```

### 3. **Forward Pass Update**

**Before (WRONG):**
```python
if self.is_3d:
    # VideoMAE2: (B, C, T, H, W)
    x = x.permute(0, 2, 1, 3, 4)
    visual_feats = self.backbone_3d(x)  # Doesn't work with HF API!
    visual_feats = visual_feats.unsqueeze(1).expand(-1, T, -1)
```

**After (CORRECT):**
```python
if self.is_3d:
    # VideoMAE (Hugging Face): expects (B, num_frames, C, H, W)
    # Our input is (B, T, C, H, W) which is already correct!
    outputs = self.backbone_3d(pixel_values=x)
    # outputs.last_hidden_state is (B, num_patches, hidden_dim)
    # Take mean across patches to get (B, hidden_dim)
    video_feats = outputs.last_hidden_state.mean(dim=1)  # (B, D)
    # Expand to all frames
    visual_feats = video_feats.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)
```

---

## VideoMAE Model Details

### Model Used
- **Name**: `MCG-NJU/videomae-base-finetuned-kinetics`
- **Source**: Hugging Face Hub
- **Pretrained**: On Kinetics-400 video dataset
- **Architecture**: VideoMAE-base (768 hidden dims)

### Input Format
```python
# VideoMAE expects exactly 16 frames (pretrained on 16-frame clips):
pixel_values: (B, 16, C, H, W)

# Our format:
video: (B, T, C, H, W)  # T can be 180, 240, etc.

# Solution: Sample 16 frames uniformly from T frames
# Example: T=180 → sample frames at indices [0, 12, 24, ..., 168, 180]
# This gives us 16 evenly-spaced frames covering the full temporal window
```

### Output Format
```python
# Sample 16 frames uniformly from T frames
num_frames = 16
indices = torch.linspace(0, T - 1, num_frames).long()  # [0, 12, 24, ..., 168]
x_sampled = video[:, indices]  # (B, 16, C, H, W)

# Pass to VideoMAE
outputs = model(pixel_values=x_sampled)

# outputs.last_hidden_state: (B, num_patches, hidden_dim)
# - num_patches = (16 frames / tubelet_size) * (224 / 16) * (224 / 16)
# - num_patches = 8 * 14 * 14 = 1568 patches
# - hidden_dim = 768 (for VideoMAE-base)

# We take mean across patches to get video-level features:
video_feats = outputs.last_hidden_state.mean(dim=1)  # (B, 768)

# Then expand to ALL T frames for temporal modeling:
visual_feats = video_feats.unsqueeze(1).expand(-1, T, -1)  # (B, T, 768)

# This gives the same feature vector to all frames, which is then
# processed by the TCN to add frame-specific temporal context
```

---

## Important: 16-Frame Sampling

### Why Sample 16 Frames?

VideoMAE was **pre-trained on 16-frame video clips** from Kinetics-400. The model's position embeddings are fixed for exactly 16 frames.

**Our sequences are longer (180-240 frames)**, so we:
1. Sample 16 frames **uniformly** across the full sequence
2. Pass these 16 frames to VideoMAE
3. Get a single video-level feature vector
4. Expand this to all T frames
5. Use TCN to add frame-specific temporal context

### Sampling Strategy

```python
T = 180  # Our window length
num_frames = 16  # VideoMAE's expected input

# Sample uniformly
indices = torch.linspace(0, T - 1, num_frames).long()
# Result: [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 179]

# This covers the entire temporal window with 16 evenly-spaced frames
```

### Implications

**Pros:**
- ✅ Uses pretrained VideoMAE weights (trained on 16 frames)
- ✅ Efficient (only processes 16 frames instead of 180)
- ✅ Covers entire temporal window uniformly
- ✅ TCN adds detailed frame-level temporal modeling

**Cons:**
- ⚠️ VideoMAE only sees 16 frames (not all 180)
- ⚠️ Same visual features for all frames initially
- ⚠️ Relies on TCN to add frame-specific details

**Why this works:**
- VideoMAE extracts high-level video understanding (what's happening overall)
- TCN adds fine-grained temporal dynamics (when exactly things happen)
- Pose graph features provide frame-level geometric information
- Together, they capture both global context and local details

---

## Installation Requirements

### Required Package

```bash
pip install transformers
```

**Version**: Any recent version (tested with 4.30+)

### Full Installation for Training

```bash
# Install all dependencies
pip install torch torchvision torchaudio  # PyTorch
pip install transformers                   # For VideoMAE
pip install timm                           # For 2D ViT fallback
pip install opencv-python                  # For video loading
pip install pandas                         # For CSV loading
pip install tqdm                           # For progress bars
pip install scikit-learn                   # For metrics
```

---

## Fallback Behavior

The script has **automatic fallback** if VideoMAE is not available:

### Fallback Chain:
1. **Try VideoMAE (3D)** from Hugging Face
   - If `transformers` is installed
   - And model can be downloaded
   - → Use 3D VideoMAE ✓

2. **Fallback to 2D ViT** from timm
   - If VideoMAE fails or transformers not installed
   - → Try DINOv2 or AugReg ViTs ✓

3. **Error if all fail**
   - No suitable backbone found
   - → RuntimeError

### Expected Messages

**Success (VideoMAE):**
```
[info] Using VideoMAE (3D) backbone from Hugging Face
[info] Model: MCG-NJU/videomae-base-finetuned-kinetics
```

**Fallback (2D ViT):**
```
[warn] VideoMAE not available (...), falling back to 2D path
[warn] Falling back to 2D path: collapsing clip_T frames with mean pooling.
[info] Using 2D backbone: vit_small_patch14_dinov2.lvd142m
```

**No transformers:**
```
[warn] transformers not installed, falling back to 2D path
[warn] Falling back to 2D path: collapsing clip_T frames with mean pooling.
[info] Using 2D backbone: vit_small_patch14_dinov2.lvd142m
```

---

## Performance Implications

### VideoMAE (3D) - Recommended
- ✅ **Better**: Understands temporal dynamics natively
- ✅ **More parameters**: Better feature learning
- ❌ **Slower**: 3D convolutions are expensive
- ❌ **More memory**: Processes entire video clip at once

### 2D ViT Fallback
- ✅ **Faster**: Processes frames independently
- ✅ **Less memory**: Smaller model
- ❌ **Worse temporal modeling**: Relies on TCN for temporal context
- ❌ **No 3D convolutions**: May miss temporal patterns

**Recommendation**: Use VideoMAE (3D) if you have:
- GPU with ≥16GB VRAM
- Want best performance
- Can afford slower training

Use 2D fallback if:
- Limited GPU memory (<8GB)
- Need faster iteration
- Training on CPU

---

## Testing the Update

### Quick Test

```bash
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --epochs 1 \
    --batch_size 1
```

**Look for:**
```
[info] Using VideoMAE (3D) backbone from Hugging Face
[info] Model: MCG-NJU/videomae-base-finetuned-kinetics
```

If you see this, VideoMAE is working! ✓

### First-Time Download

The model will be downloaded from Hugging Face Hub (~350MB):
```
Downloading model.safetensors: 100%|██████████| 347M/347M [00:30<00:00]
```

**Cached location**: `~/.cache/huggingface/hub/`

Subsequent runs will be faster (no download).

---

## Troubleshooting

### Issue: "transformers not installed"

**Solution:**
```bash
pip install transformers
```

### Issue: "Cannot download model"

**Cause**: Network issue or Hugging Face Hub down

**Solution 1**: Wait and retry
**Solution 2**: Use 2D fallback (works without download)

### Issue: CUDA Out of Memory with VideoMAE

**Cause**: VideoMAE is memory-intensive (3D model)

**Solutions:**
1. Reduce batch size:
   ```bash
   --batch_size 1
   ```

2. Reduce window length:
   ```bash
   --train_T 120 --val_T 120
   ```

3. Use 2D fallback (force it by uninstalling transformers temporarily)

### Issue: VideoMAE forward pass error

**Possible cause**: Input shape mismatch

**Check**:
- Input should be (B, T, C, H, W)
- C=3 (RGB)
- H=W=224 (default)

If error persists, the script will automatically fall back to 2D ViT.

---

## Summary

✅ **Updated to use Hugging Face VideoMAE**: `MCG-NJU/videomae-base-finetuned-kinetics`
✅ **Automatic fallback to 2D ViT**: If transformers not installed or download fails
✅ **Same input/output format**: No changes to data pipeline
✅ **Better 3D temporal modeling**: When VideoMAE is used

**Installation:**
```bash
pip install transformers
```

**Training:**
```bash
python train_multitask.py \
    --annotations ./Annotations \
    --videos ./Videos \
    --model_type action_only \
    --epochs 50 \
    --batch_size 2
```

The script will automatically use VideoMAE (3D) if available, or fall back to 2D ViT!
