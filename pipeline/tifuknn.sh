#!/bin/bash

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

for fold_id in 0 1 2; do
    echo "Running TIFUKNN for dunnhumby fold $fold_id"
    python tifuknn_new.py ../../jsondata/dunnhumby_history.json ../../jsondata/dunnhumby_future.json ../../keyset/dunnhumby_keyset_$fold_id.json 900 0.9 0.6 0.2 3 20
    
    echo "Running TIFUKNN for tafeng fold $fold_id"
    python tifuknn_new.py ../../jsondata/tafeng_history.json ../../jsondata/tafeng_future.json ../../keyset/tafeng_keyset_$fold_id.json 300 0.9 0.7 0.7 7 20
    
    echo "Running TIFUKNN for instacart fold $fold_id"
    python tifuknn_new.py ../../jsondata/instacart_history.json ../../jsondata/instacart_future.json ../../keyset/instacart_keyset_$fold_id.json 900 0.9 0.7 0.9 3 20
done

echo "TIFUKNN method completed."