#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status


cd ../methods/upcf

echo "Setting up UP-CF environment..."

# Create virtual environment if it doesn't exist
if ! conda env list | grep -q venv-upcf; then
    echo "Creating virtual environment for UP-CF..."
    conda create --name venv-upcf python=3.8 -y
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate venv-upcf

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "Running UP-CF method..."

# Source environment variables
source ../../.env

# Parse dataset names from env variable
IFS=',' read -ra DATASET_ARRAY <<< "$DATASET_NAMES"

for dataset in "${DATASET_ARRAY[@]}"; do
    for foldk in 0 1 2; do
        echo "Running UP-CF for $dataset fold $foldk (using default parameters)"
        python racf.py --dataset $dataset --foldk $foldk --asymmetry 0.75
    done
done

echo "UP-CF method completed."