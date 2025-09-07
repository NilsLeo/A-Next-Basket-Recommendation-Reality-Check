#!/bin/bash

# Load environment variables
if [ -f "../.env" ]; then
    source ../.env
else
    echo "Error: .env file not found in parent directory"
    exit 1
fi

cd ../methods/tifuknn

echo "Setting up TIFUKNN environment..."

# Create virtual environment if it doesn't exist
if ! conda env list | grep -q venv-tifuknn; then
    echo "Creating virtual environment for TIFUKNN..."
    conda create --name venv-tifuknn python=3.6.8 -y
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate venv-tifuknn

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "Running TIFUKNN method..."

# Parse dataset names from .env
IFS=',' read -ra DATASETS <<< "$DATASET_NAMES"

# Common parameters for all datasets
COMMON_PARAMS="300 0.9 0.7 0.7 7 20"

for fold_id in 0 1 2; do
    for dataset in "${DATASETS[@]}"; do
        dataset=$(echo "$dataset" | tr '[:upper:]' '[:lower:]' | xargs)
        echo "Running TIFUKNN for $dataset fold $fold_id"
        python tifuknn_new.py "../../jsondata/${dataset}_history.json" "../../jsondata/${dataset}_future.json" "../../keyset/${dataset}_keyset_$fold_id.json" $COMMON_PARAMS
    done
done

echo "TIFUKNN method completed."