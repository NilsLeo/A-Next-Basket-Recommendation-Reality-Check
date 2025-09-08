#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status


cd ../methods/sets2sets

echo "Setting up Sets2Sets environment..."

# Create virtual environment if it doesn't exist
if ! ~/miniconda3/bin/conda env list | grep -q venv-sets2sets; then
    echo "Creating virtual environment for Sets2Sets..."
    ~/miniconda3/bin/conda create --name venv-sets2sets python=3.6.8 -y
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv-sets2sets

if [ -f "requirements.txt" ]; then
    # Install non-PyTorch dependencies first
    pip install certifi==2021.5.30 numpy==1.19.5 Pillow==8.4.0 six==1.17.0 tqdm==4.19.9
    # Install PyTorch with CUDA support
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
fi

echo "Running Sets2Sets method..."

# Source environment variables
source ../../.env

# Parse dataset names from env variable
IFS=',' read -ra DATASET_ARRAY <<< "$DATASET_NAMES"

# Dynamically assign dataset IDs based on order in .env
declare -A dataset_ids
for i in "${!DATASET_ARRAY[@]}"; do
    dataset_ids["${DATASET_ARRAY[$i]}"]="$i"
done

for dataset in "${DATASET_ARRAY[@]}"; do
    dataset_id=${dataset_ids[$dataset]}
    for fold_id in 0 1 2; do
        echo "Training Sets2Sets for $dataset fold $fold_id"
        python sets2sets_new.py $dataset $fold_id 10 1
        
        echo "Predicting with Sets2Sets for $dataset fold $fold_id" 
        python sets2sets_new.py $dataset $fold_id 10 0
    done
done

echo "Sets2Sets method completed."
