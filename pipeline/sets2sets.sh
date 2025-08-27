#!/bin/bash

cd ../methods/sets2sets

echo "Setting up Sets2Sets environment..."

# Create virtual environment if it doesn't exist
if ! conda env list | grep -q venv-sets2sets; then
    echo "Creating virtual environment for Sets2Sets..."
    conda create --name venv-sets2sets python=3.6.8 -y
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate venv-sets2sets

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "Running Sets2Sets method..."

declare -A dataset_ids=( ["dunnhumby"]="0" ["tafeng"]="1" ["instacart"]="2" )

for dataset in dunnhumby tafeng instacart; do
    dataset_id=${dataset_ids[$dataset]}
    for fold_id in 0 1 2; do
        echo "Training Sets2Sets for $dataset fold $fold_id"
        python sets2sets_new.py $dataset $fold_id 10 1
        
        echo "Predicting with Sets2Sets for $dataset fold $fold_id" 
        python sets2sets_new.py $dataset $fold_id 10 0
    done
done

echo "Sets2Sets method completed."