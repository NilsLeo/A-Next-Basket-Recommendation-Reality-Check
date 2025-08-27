#!/bin/bash
#!/bin/bash

# Convert CSV files to merged JSON format  
# Used by: BEACON, CLEA, UP-CF


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
echo "Converting CSV files to merged JSON format..."
echo "Output: mergeddataset/{dataset}_merged.json"

cd ../preprocess
python csv_to_merged_json.py

echo "CSV to merged JSON conversion completed!"
