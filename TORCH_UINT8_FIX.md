# PyTorch uint8 Fix

## Issue
```python
AttributeError: 'Tensor' object has no attribute 'uint8'
```

This error occurred because the older PyTorch syntax `.uint8()` is no longer valid in newer PyTorch versions.

## Root Cause

**Old PyTorch syntax (pre-1.0):**
```python
tensor.uint8()  # Method call
```

**New PyTorch syntax (1.0+):**
```python
tensor.to(torch.uint8)  # Using .to() method
```

## Fix Applied

Changed in both data loaders:

**Before:**
```python
return torch.from_numpy(clip).uint8()
```

**After:**
```python
return torch.from_numpy(clip).to(torch.uint8)
```

## Files Updated

1. ✅ `data_loader.py` (line 207)
2. ✅ `multimodal_data_loader.py` (line 296)

## Why This Matters

The video frames are loaded as numpy arrays with dtype `uint8` (0-255 pixel values). When converting to PyTorch tensors, we need to preserve this data type to:

1. **Save memory**: uint8 uses 1 byte per pixel vs float32 (4 bytes)
2. **Maintain format**: Video frames are naturally in 0-255 range
3. **Normalize later**: The `__getitem__` method converts to float and normalizes to [0, 1]

## Data Flow

```
Video Frame (numpy uint8)
  → torch.from_numpy() → Tensor (uint8)
  → .to(torch.uint8) → Ensure uint8 dtype
  → normalize in __getitem__() → Float tensor [0, 1]
  → Model input
```

## Now Ready to Train!

This fix applies to both:
- **Visual-only model** (`train.py`)
- **Multimodal model** (`multimodal_train.py`)

You should no longer see this error during data loading!
