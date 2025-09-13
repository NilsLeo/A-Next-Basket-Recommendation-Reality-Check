#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status


cd ../methods/sets2sets

echo "Setting up Sets2Sets environment..."

# Create virtual environment if it doesn't exist
if ! ~/miniconda3/bin/conda env list | grep -q venv-sets2sets; then
    echo "Creating virtual environment for Sets2Sets (Python 3.9)..."
    ~/miniconda3/bin/conda create --name venv-sets2sets python=3.9 -y
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv-sets2sets

echo "Installing dependencies..."
# Use CPU-only by default for broad compatibility; set SETS2SETS_CUDA=1 to use CUDA wheels
pip install --upgrade pip
# Numpy 1.24.4 works across Python 3.8/3.9 in offline/limited indexes
pip install numpy==1.24.4 tqdm==4.66.4 six==1.17.0 pillow==10.4.0 certifi
if [ "${SETS2SETS_CUDA}" = "1" ]; then
    echo "Installing PyTorch (CUDA 12.1) for Sets2Sets..."
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121 || true
else
    echo "Installing PyTorch (CPU) for Sets2Sets..."
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
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
        enc_path="./models/encoder_${dataset}${fold_id}_model_best"
        dec_path="./models/decoder_${dataset}${fold_id}_model_best"
        if [ -f "$enc_path" ] && [ -f "$dec_path" ]; then
            echo "Found existing models for $dataset fold $fold_id; skipping training."
        else
            echo "Training Sets2Sets for $dataset fold $fold_id"
            python sets2sets_new.py $dataset $fold_id 10 1
        fi
        
        echo "Predicting with Sets2Sets for $dataset fold $fold_id"
        # Write predictions to JSON for evaluation compatibility
        python -m pip show torch >/dev/null 2>&1 || { echo "Torch not installed"; exit 1; }
        python pred_results.py --dataset "$dataset" --fold_id "$fold_id"
    done
done

echo "Sets2Sets method completed."
