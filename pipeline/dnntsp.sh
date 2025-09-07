#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Load environment variables from .env file
if [ -f "../.env" ]; then
    export $(grep -v '^#' ../.env | xargs)
fi

cd ../methods/dnntsp/train

echo "Setting up DNNTSP environment..."

# Create virtual environment if it doesn't exist
if ! conda env list | grep -q venv-dnntsp; then
    echo "Creating virtual environment for DNNTSP..."
    conda create --name venv-dnntsp python=3.6.8 -y
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate venv-dnntsp

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "Running DNNTSP method..."

# Handle dataset iteration
if [ -n "$DATASET" ]; then
    # Single dataset specified
    DATASETS="$DATASET"
elif [ -n "$DATASET_NAMES" ]; then
    # Multiple datasets from .env file
    DATASETS=$(echo "$DATASET_NAMES" | tr ',' ' ')
else
    echo "Error: Neither DATASET nor DATASET_NAMES environment variable is set"
    echo "Available datasets: tafeng, instacart"
    echo "Set DATASET=<dataset_name> or add it to .env file"
    exit 1
fi

FOLD_ID=${FOLD_ID:-0}

# Iterate over each dataset
for dataset in $DATASETS; do
    echo "========================================="
    echo "Processing dataset: $dataset"
    echo "========================================="
    
    echo "Generating config for dataset: $dataset, fold: $FOLD_ID"
    python ../generate_config.py "$dataset" "$FOLD_ID"
    
    echo "Training DNNTSP for dataset: $dataset..."
    python train_main.py
    
    echo "DNNTSP training completed for dataset: $dataset"
    echo "Saved model for dataset: $dataset"
    echo ""
done

echo "========================================="
echo "All datasets processed. Check saved models and run predictions manually."
echo "Use: python pred_results.py --dataset <dataset> --fold_id <fold> --best_mode_path <model_path>"