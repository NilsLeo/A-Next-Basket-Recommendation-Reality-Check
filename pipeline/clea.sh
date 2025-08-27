#!/bin/bash

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

# Dataset parameters
declare -A num_products=( ["dunnhumby"]="3920" ["tafeng"]="11997" ["instacart"]="13897" )
declare -A num_users=( ["dunnhumby"]="22530" ["tafeng"]="13858" ["instacart"]="19435" )

for dataset in dunnhumby tafeng instacart; do
    for foldk in 0 1 2; do
        echo "Pre-training CLEA for $dataset fold $foldk"
        python new_main.py --dataset $dataset --foldk $foldk --pretrain_epoch 20 --before_epoch 0 --epoch 10 --embedding_dim 64 --num_product ${num_products[$dataset]} --num_users ${num_users[$dataset]}
        
        echo "Training CLEA for $dataset fold $foldk"
        python new_main.py --dataset $dataset --foldk $foldk --log_fire cleamodel --alternative_train_epoch 10 --alternative_train_epoch_D 10 --pretrain_epoch 2 --before_epoch 2 --epoch 30 --temp_learn 0 --temp 10 --embedding_dim 64 --num_product ${num_products[$dataset]} --num_users ${num_users[$dataset]}
        
        echo "Predicting with CLEA for $dataset fold $foldk"
        python pred_results.py --dataset $dataset --foldk $foldk --log_fire cleamodel --alternative_train_epoch 10 --alternative_train_epoch_D 10 --pretrain_epoch 2 --before_epoch 2 --epoch 30 --temp_learn 0 --temp 10 --embedding_dim 64 --num_product ${num_products[$dataset]} --num_users ${num_users[$dataset]}
    done
done

echo "CLEA method completed."