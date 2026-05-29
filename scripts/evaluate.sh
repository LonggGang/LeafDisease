#!/bin/bash
# Description: Wrapper script to evaluate the model
set -e

CHECKPOINT=${1:-"checkpoints/best_model.pth"}

echo "Starting Evaluation Pipeline"
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Make sure you have trained the model first!"
    exit 1
fi

python main.py --mode eval --checkpoint "$CHECKPOINT"
