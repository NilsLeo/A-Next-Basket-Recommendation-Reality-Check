#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting full NBR-2 pipeline..."

# Create and activate virtual environment if it doesn't exist
if [ ! -d "../venv" ]; then
    echo "Creating virtual environment..."
    conda create --name nbr-venv python=3.9 -y
fi

echo "Activating virtual environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nbr-venv

# Install main dependencies
echo "Installing main dependencies..."
pip install -r ../requirements.txt

# Create logs directory
mkdir -p logs

echo "Step: Converting data formats... (~3-5 minutes)"
./csv_to_json.sh 2>&1 | tee logs/csv_to_json.log
./csv_to_mergedjson.sh 2>&1 | tee logs/csv_to_mergedjson.log
./keyset.sh 2>&1 | tee logs/keyset.log

# Step 5: Run all models (can be run in parallel)
echo "Step: Running all models in parallel... (~4-6 hours total)"
echo "Note: Models run in parallel, so total time is determined by slowest model"
./topfreq.sh 2>&1 | tee logs/topfreq.log & # 🟢 Fast (~2-3 minutes)
./dream.sh 2>&1 | tee logs/dream.log & # 🟠 Medium (~10-15 minutes) 
./beacon.sh 2>&1 | tee logs/beacon.log & # 🟠Very Slow (~4-6 hours)
./clea.sh 2>&1 | tee logs/clea.log & #🟠Slow (~1-2 hours)
./sets2sets.sh 2>&1 | tee logs/sets2sets.log & # 🟠 Slow (~2-3 hours)
./dnntsp.sh 2>&1 | tee logs/dnntsp.log & # 🟢Medium (~30-60 minutes)
./upcf.sh 2>&1 | tee logs/upcf.log & # 🟢 Slow (~1-2 hours)
./tifuknn.sh 2>&1 | tee logs/tifuknn.log & # 🟢 Fast (~1-2 minutes)

wait

# Step 6: Evaluate performance
echo "Step: Evaluating performance... (~2-5 minutes)"
./performance.sh 2>&1 | tee logs/performance.log
./performance_gain.sh 2>&1 | tee logs/performance_gain.log

echo "Full pipeline completed!"
