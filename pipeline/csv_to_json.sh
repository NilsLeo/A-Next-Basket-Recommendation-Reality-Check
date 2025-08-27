#!/bin/bash

set -e  # stop on first error

# Ensure conda is available in non-interactive shells
source ~/miniconda3/etc/profile.d/conda.sh

# Create environment only if it doesn’t exist yet
if ! conda env list | grep -q '^venv'; then
    conda create --name venv python=3.9.23 -y
fi

# Activate environment
conda activate venv

# Install dependencies (fall back if requirements.txt is missing)
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

cd ../preprocess
# Convert CSV files to separated JSON format
# Used by: Sets2Sets, TIFUKNN, DNNTSP, DREAM

echo "Converting CSV files to separated JSON format..."
echo "Output: jsondata/{dataset}_history.json and jsondata/{dataset}_future.json"

cd ../preprocess
python csv_to_separated_json.py

echo "CSV to JSON conversion completed!"
