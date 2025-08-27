#!/bin/bash

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
echo "Note: You need to manually configure the config files for each dataset/fold"

echo "Training DNNTSP..."
python train_main.py

echo "DNNTSP training completed. Check saved models and run predictions manually."
echo "Use: python pred_results.py --dataset <dataset> --fold_id <fold> --best_mode_path <model_path>"