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

# Iterate over each dataset and fold
for dataset in $DATASETS; do
    for fold in 0 1 2; do
        echo "========================================="
        echo "Processing dataset: $dataset, fold: $fold"
        echo "========================================="
        
        echo "Generating config for dataset: $dataset, fold: $fold"
        python ../generate_config.py "$dataset" "$fold"
        
        echo "Training DNNTSP for dataset: $dataset, fold: $fold..."
        python train_main.py
        
        echo "DNNTSP training completed for dataset: $dataset, fold: $fold"
        
        # Find the best model for this dataset and fold
        model_folder="../save_model_folder/${dataset}/DNNTSP"
        if [ -d "$model_folder" ]; then
            # Find the latest/best model file
            best_model=$(ls -t "$model_folder"/model_epoch_*.pkl 2>/dev/null | head -1)
            if [ -n "$best_model" ]; then
                echo "Running predictions for dataset: $dataset, fold: $fold"
                cd ..
                python pred_results.py --dataset "$dataset" --fold_id "$fold" --best_model_path "$best_model"
                cd train
                echo "Predictions completed for dataset: $dataset, fold: $fold"
            else
                echo "Warning: No model file found in $model_folder"
            fi
        else
            echo "Warning: Model folder not found: $model_folder"
        fi
        echo ""
    done
done

echo "========================================="
echo "All datasets processed. Training and predictions completed."