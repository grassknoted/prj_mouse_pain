#!/bin/bash
#
# Extract frames from all videos for fast training
# This is a one-time preprocessing step
#

VIDEO_DIR="/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_mouse_pain/Dec24/REMY2/ALL_VIDEOS/"
OUTPUT_DIR="/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_mouse_pain/Dec24/REMY2/extracted_frames/"

echo "=========================================="
echo "Frame Extraction for Fast Training"
echo "=========================================="
echo "Video directory: $VIDEO_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

python extract_frames.py \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Extraction Complete!"
echo "=========================================="
echo ""
echo "Now you can train with extracted frames:"
echo "./train_multimodal_multigpu.sh 3"
