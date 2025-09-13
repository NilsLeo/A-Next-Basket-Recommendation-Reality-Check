#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Load environment variables from .env file
if [ -f "../.env" ]; then
    export $(grep -v '^#' ../.env | xargs)
fi

cd ../methods/dnntsp/train

echo "Setting up DNNTSP environment..."

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

# Create virtual environment if it doesn't exist
if ! conda env list | grep -q "^venv-dnntsp\b"; then
    echo "Creating virtual environment for DNNTSP (Python 3.8)..."
    conda create --name venv-dnntsp python=3.8 -y
fi

# Activate virtual environment
echo "Activating virtual environment and installing dependencies..."
conda activate venv-dnntsp

# Ensure correct Python version (>=3.7) for dgl-cu111/torch wheels
PYVER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYVER" == 3.6* ]]; then
    echo "Detected Python $PYVER in venv-dnntsp; recreating with Python 3.8 for CUDA libs..."
    conda deactivate || true
    conda remove -y --name venv-dnntsp --all
    conda create --name venv-dnntsp python=3.8 -y
    conda activate venv-dnntsp
fi

export DGLBACKEND=pytorch
python -c "import sys; print('Python:', sys.version)"

# Install method dependencies
if [ "${DNNTSP_CPU_ONLY}" = "1" ]; then
    echo "Installing CPU-only dependencies (torch CPU, dgl CPU)..."
    pip install --upgrade pip
    # Remove any CUDA-linked builds if present
    pip uninstall -y dgl-cu111 dgl || true
    pip uninstall -y torch torchvision || true
    # CPU wheels
    pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==1.10.0+cpu torchvision==0.11.0+cpu
    pip install dgl==0.6.1
    # Install remaining requirements excluding torch/torchvision/dgl
    if [ -f "../requirements_frozen.txt" ]; then
        awk '!/^(torch|torchvision|dgl|dgl-cu)/' ../requirements_frozen.txt > /tmp/dnntsp-req-cpu.txt
        pip install -r /tmp/dnntsp-req-cpu.txt
    elif [ -f "../requirements.txt" ]; then
        awk '!/^(torch|torchvision|dgl|dgl-cu)/' ../requirements.txt > /tmp/dnntsp-req-cpu.txt
        pip install -r /tmp/dnntsp-req-cpu.txt
    fi
else
    echo "Installing CUDA 11.1 compatible dependencies (torch/cu111, dgl-cu111)..."
    if [ -f "../requirements_frozen.txt" ]; then
        pip install --upgrade pip
        pip install --extra-index-url https://download.pytorch.org/whl/cu111 \
            torch==1.10.0+cu111 torchvision==0.11.0+cu111 || true
        pip install -r ../requirements_frozen.txt
    elif [ -f "../requirements.txt" ]; then
        pip install -r ../requirements.txt
    fi
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
        
        # Optional CPU-only mode
        if [ "${DNNTSP_CPU_ONLY}" = "1" ]; then
            echo "Forcing CPU-only prediction/training for DNNTSP..."
            python - <<'PY'
import json
with open('utils/config.json') as f:
    cfg = json.load(f)
cfg['cuda'] = -1
with open('utils/config.json','w') as f:
    json.dump(cfg, f, indent=4)
print('Updated utils/config.json with cuda=-1')
PY
            export CUDA_VISIBLE_DEVICES=""
        fi
        
        echo "Training DNNTSP for dataset: $dataset, fold: $fold..."
        python train_main.py
        
        echo "DNNTSP training completed for dataset: $dataset, fold: $fold"
        
        # Find the best model for this dataset and fold (capitalize dataset name to match config)
        dataset_capitalized="$(echo "${dataset}" | sed 's/.*/\u&/')"
        model_folder="../save_model_folder/${dataset_capitalized}/DNNTSP"
        if [ -d "$model_folder" ]; then
            # Find the latest/best model file
            best_model=$(ls -t "$model_folder"/model_epoch_*.pkl 2>/dev/null | head -1)
            if [ -n "$best_model" ]; then
                # Resolve to absolute path before changing directories
                best_model="$(realpath "$best_model")"
                echo "Running predictions for dataset: $dataset, fold: $fold"
                cd ..
                python pred_results.py --dataset "$dataset" --fold_id "$fold" --best_model_path "$best_model"
                cd train
                echo "Predictions completed for dataset: $dataset, fold: $fold"
            else
                echo "ERROR: No model file found in $model_folder"
                echo "Training completed but model was not saved properly!"
                exit 1
            fi
        else
            echo "ERROR: Model folder not found: $model_folder"
            echo "Training completed but model folder was not created!"
            exit 1
        fi
        echo ""
    done
done

echo "========================================="
echo "All datasets processed. Training and predictions completed."
