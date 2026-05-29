#!/bin/bash
# Description: Environment setup and data preparation for the Plant Disease Pipeline
set -e

echo "=== 1. Installing Dependencies ==="
pip install -r requirements.txt

echo "=== 2. Creating Project Directories ==="
mkdir -p data/raw data/train data/val data/test checkpoints logs

echo "=== 3. Data Preparation ==="
# Allow user to pass a raw data directory as an argument, otherwise use default
RAW_DIR=${1:-"data/raw"}
OUT_DIR=${2:-"data"}

if [ -z "$(ls -A $RAW_DIR 2>/dev/null)" ]; then
    echo "⚠️  Warning: $RAW_DIR is empty."
    echo "Please download your datasets (PlantDoc/IDADP) and extract them into $RAW_DIR."
    echo "Once the data is in $RAW_DIR, run this script again to execute the physical splitting."
else
    echo "Found data in $RAW_DIR. Running physical dataset split..."
    python src/utils/split_dataset.py --raw_dir "$RAW_DIR" --out_dir "$OUT_DIR"
fi

echo "=== Setup Complete! ==="
