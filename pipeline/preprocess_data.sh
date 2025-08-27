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
echo "Preprocessing datasets..."
cd ../preprocess

# Empty the dataset folder before preprocessing
if [ -d "../dataset" ]; then
    echo "Cleaning existing dataset folder..."
    rm -rf ../dataset/*
else
    echo "Creating dataset folder..."
    mkdir -p ../dataset
fi

# Create dataset directory in preprocess folder for the scripts to use
if [ ! -d "dataset" ]; then
    echo "Creating local dataset folder for preprocessing..."
    mkdir -p dataset
fi

python tafeng.py
python dunnhumby.py
python Instacart.py

# Move generated files to the main dataset folder
if [ -f "dataset/tafeng.csv" ]; then
    mv dataset/tafeng.csv ../dataset/
fi
if [ -f "dataset/tafeng_tmp.csv" ]; then
    mv dataset/tafeng_tmp.csv ../dataset/
fi
if [ -f "dataset/dunnhumby.csv" ]; then
    mv dataset/dunnhumby.csv ../dataset/
fi
if [ -f "dataset/instacart.csv" ]; then
    mv dataset/instacart.csv ../dataset/
fi

# Clean up the temporary dataset folder
rm -rf dataset

cd ..

echo "Preprocessing completed."


