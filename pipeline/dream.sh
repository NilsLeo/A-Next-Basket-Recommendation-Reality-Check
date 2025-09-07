#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

cd ../methods/dream

echo "Setting up DREAM environment..."

# Create virtual environment if it doesn't exist
if ! conda env list | grep -q venv-dream; then
    echo "Creating virtual environment for DREAM..."
    conda create --name venv-dream python=3.6 -y
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate venv-dream

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "Generating dynamic config files..."
python generate_configs.py

echo "Running DREAM method..."

# Source environment variables
source ../../.env

# Parse dataset names from env variable
IFS=',' read -ra DATASET_ARRAY <<< "$DATASET_NAMES"

for dataset in "${DATASET_ARRAY[@]}"; do
    for fold_id in 0 1 2; do
        echo "Training DREAM for $dataset fold $fold_id"
        python trainer.py --dataset $dataset --fold_id $fold_id --attention 1
        
        echo "Predicting with DREAM for $dataset fold $fold_id"
        python pred_results.py --dataset $dataset --fold_id $fold_id
    done
done

echo "DREAM method completed."