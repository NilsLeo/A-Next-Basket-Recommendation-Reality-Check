#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status


cd ../methods/beacon

echo "Setting up BEACON environment..."

# Create virtual environment if it doesn't exist
if ! conda env list | grep -q venv-beacon; then
    echo "Creating virtual environment for BEACON..."
    conda create --name venv-beacon python=3.6 -y
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate venv-beacon

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "Running BEACON method..."

# Source environment variables
source ../../.env

# Parse dataset names from env variable
IFS=',' read -ra DATASET_ARRAY <<< "$DATASET_NAMES"

for dataset in "${DATASET_ARRAY[@]}"; do
    for foldk in 0 1 2; do
        echo "Generating correlation matrix for $dataset fold $foldk"
        python cmatrix_generator.py --dataset $dataset --foldk $foldk
        
        echo "Training BEACON for $dataset fold $foldk"
        python main_gpu.py --dataset $dataset --foldk $foldk --train_mode True --emb_dim 64
        
        echo "Predicting with BEACON for $dataset fold $foldk"
        python main_gpu.py --dataset $dataset --foldk $foldk --prediction_mode True --emb_dim 64
    done
done

echo "BEACON method completed."