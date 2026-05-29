#!/bin/bash
# Description: Wrapper script to start the training pipeline
set -e

# Configurable arguments with defaults
TRAIN_CFG=${1:-"configs/train.yaml"}
AUG_CFG=${2:-"configs/augment.yaml"}

echo "🚀 Starting Training Pipeline..."
echo "Using configs: $TRAIN_CFG and $AUG_CFG"

python main.py --mode train --train_cfg "$TRAIN_CFG" --augment_cfg "$AUG_CFG"
