#!/bin/bash

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

# Dataset parameters
declare -A recency=( ["dunnhumby"]="25" ["tafeng"]="10" ["instacart"]="10" )
declare -A locality=( ["dunnhumby"]="10" ["tafeng"]="10" ["instacart"]="100" )

for dataset in dunnhumby tafeng instacart; do
    for foldk in 0 1 2; do
        echo "Running UP-CF for $dataset fold $foldk"
        python racf.py --dataset $dataset --foldk $foldk --recency ${recency[$dataset]} --asymmetry 0.75 --locality ${locality[$dataset]}
    done
done

echo "UP-CF method completed."