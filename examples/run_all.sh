#!/bin/bash
# Complete pipeline script for frame segmentation
# Usage: ./run_all.sh [--n N] [--skip-train]

set -e

# Default values
N_PAGES=100
SKIP_TRAIN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n)
            N_PAGES="$2"
            shift 2
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--n N] [--skip-train]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Frame Segmentation Pipeline"
echo "=========================================="
echo "Pages to generate: $N_PAGES"
echo "Skip training: $SKIP_TRAIN"
echo ""

# Change to script directory
cd "$(dirname "$0")/.."

# Step 1: Generate HTML pages
echo "Step 1/6: Generating HTML pages..."
py data-generation/html_generator.py --n "$N_PAGES"
echo ""

# Step 2: Render screenshots
echo "Step 2/6: Rendering screenshots with Playwright..."
py data-generation/playwright_render.py --workers 4
echo ""

# Step 3: Generate masks
echo "Step 3/6: Generating instance masks..."
py data-generation/make_masks.py
echo ""

# Step 4: Convert to COCO format
echo "Step 4/6: Converting to COCO format..."
py data-generation/coco_converter.py
echo ""

# Step 5: Train model (optional)
if [ "$SKIP_TRAIN" = false ]; then
    echo "Step 5/6: Training Mask R-CNN model..."
    echo "Note: This may take a long time. Use --skip-train to skip."
    py detectron/train.py --epochs 10 --batch-size 2
    echo ""
else
    echo "Step 5/6: Skipping training (--skip-train flag set)"
    echo ""
fi

# Step 6: Inference and postprocessing
if [ "$SKIP_TRAIN" = false ] && [ -f "outputs/model_final.pth" ]; then
    echo "Step 6/6: Running inference and postprocessing..."
    py detectron/infer_and_postprocess.py --weights outputs/model_final.pth
    echo ""
else
    echo "Step 6/6: Skipping inference (no trained model found)"
    echo "Train a model first or use pre-trained weights:"
    echo "  py detectron/infer_and_postprocess.py --weights <path_to_weights>"
    echo ""
fi

echo "=========================================="
echo "Pipeline completed!"
echo "=========================================="
echo "Results:"
echo "  - HTML pages: data/pages/"
echo "  - Screenshots: data/screenshots/"
echo "  - Masks: data/masks/"
echo "  - COCO dataset: data/coco/"
if [ "$SKIP_TRAIN" = false ]; then
    echo "  - Model: outputs/"
fi
echo "  - Inference results: data/results/"

