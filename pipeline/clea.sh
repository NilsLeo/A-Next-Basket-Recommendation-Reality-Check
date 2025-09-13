#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

cd ../methods/clea

echo "Setting up CLEA environment..."

# Create virtual environment if it doesn't exist
if ! conda env list | grep -q venv-clea; then
    echo "Creating virtual environment for CLEA..."
    conda create --name venv-clea python=3.6 -y
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate venv-clea

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "Running CLEA method..."

# Source environment variables
source ../../.env

# Parse dataset names from env variable
IFS=',' read -ra DATASET_ARRAY <<< "$DATASET_NAMES"

# Create models directory
mkdir -p models

for dataset in "${DATASET_ARRAY[@]}"; do
    # Dynamically get user and product counts from the dataset
    num_users=$(python -c "
import pandas as pd
df = pd.read_csv('../../dataset/${dataset}_history.csv')
print(df['user_id'].nunique())
")
    num_products=$(python -c "
import pandas as pd
df_history = pd.read_csv('../../dataset/${dataset}_history.csv')
df_future = pd.read_csv('../../dataset/${dataset}_future.csv')
df_combined = pd.concat([df_history, df_future])
print(df_combined['product_id'].max() + 1)
")
    
    # Create dataset-specific model directory
    mkdir -p models/$dataset
    
    for foldk in 0 1 2; do
        echo "Pre-training CLEA for $dataset fold $foldk"
        python new_main.py --dataset $dataset --foldk $foldk --num_users $num_users --num_product $num_products --pretrain_epoch 20 --before_epoch 0 --epoch 10 --embedding_dim 64
        
        echo "Training CLEA for $dataset fold $foldk"
        python new_main.py --dataset $dataset --foldk $foldk --num_users $num_users --num_product $num_products --log_fire cleamodel --alternative_train_epoch 10 --alternative_train_epoch_D 10 --pretrain_epoch 2 --before_epoch 2 --epoch 30 --temp_learn 0 --temp 10 --embedding_dim 64
        
        echo "Predicting with CLEA for $dataset fold $foldk"
        if [ -f "pred_results.py" ]; then
            # Only run prediction using the saved checkpoints
            python pred_results.py --dataset $dataset --foldk $foldk
        else
            echo "Warning: pred_results.py not found, skipping prediction step"
        fi
    done
done

echo "CLEA method completed."
