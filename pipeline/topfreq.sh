#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status


cd ../methods/g-p-gp-topfreq

echo "Setting up TopFreq methods environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv-topfreq" ]; then
    echo "Creating virtual environment for TopFreq methods..."
    conda create --name venv-topfreq python=3.9 -y
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate venv-topfreq

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Generate popularity files if they don't exist
echo "Generating popularity files..."
python generate_popularity.py

echo "Running TopFreq methods..."

# Source environment variables
source ../../.env

# Parse dataset names from env variable
IFS=',' read -ra DATASET_ARRAY <<< "$DATASET_NAMES"

for dataset in "${DATASET_ARRAY[@]}"; do
    for fold_id in 0 1 2; do
        echo "Running G-TopFreq for $dataset fold $fold_id"
        python g_topfreq.py --dataset $dataset --fold_id $fold_id
        
        echo "Running P-TopFreq for $dataset fold $fold_id"
        python p_topfreq.py --dataset $dataset --fold_id $fold_id
        
        echo "Running GP-TopFreq for $dataset fold $fold_id"
        python gp_topfreq.py --dataset $dataset --fold_id $fold_id
    done
done

echo "TopFreq methods completed."