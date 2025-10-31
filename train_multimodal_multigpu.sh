#!/bin/bash
#
# Multi-GPU training launcher for multimodal mouse action recognition
# Supports training on multiple NVIDIA GPUs using PyTorch Distributed Data Parallel (DDP)
# Combines visual (video) and pose (DLC) features for improved action recognition
#

# Number of GPUs to use (default: all available)
NUM_GPUS=${1:-3}

echo "=========================================="
echo "Multi-GPU Multimodal Training Launcher"
echo "=========================================="
echo "Number of GPUs: $NUM_GPUS"
echo ""

# Check if GPUs are available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are NVIDIA drivers installed?"
    exit 1
fi

# Display GPU information
echo "Available GPUs:"
nvidia-smi --list-gpus
echo ""

# IMPORTANT: Unset CUDA_VISIBLE_DEVICES to allow all GPUs to be visible
# torchrun will handle GPU assignment per rank
unset CUDA_VISIBLE_DEVICES

# Launch distributed training
echo "Launching distributed multimodal training..."
echo ""

# Use python -m torch.distributed.run (most compatible) or torchrun
# Try torch.distributed.run first (works with any Python)
if python -m torch.distributed.run --help &> /dev/null; then
    python -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29501 \
        multimodal_train.py
elif command -v torchrun &> /dev/null; then
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29501 \
        multimodal_train.py
else
    # Fallback to older torch.distributed.launch
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29501 \
        multimodal_train.py
fi

echo ""
echo "=========================================="
echo "Multimodal Training Complete!"
echo "=========================================="
