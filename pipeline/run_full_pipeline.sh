#!/bin/bash

echo "Starting full NBR-2 pipeline..."

# Step 1: Download data (manual step - user needs raw datasets)
echo "Step 1: Ensure raw datasets are in DataSource/ directory"
# 🟢
./download_data.sh
# Step 2: Preprocess data
echo "Step 2: Preprocessing datasets..."

# 🟠
./preprocess_data.sh

# Step 3: Generate keysets
echo "Step 3: Generating keysets..."

# 🟢
./keyset.sh

# Step 4: Convert to different formats
echo "Step 4: Converting data formats..."
# 🟢
./csv_to_json.sh
# 🟢
./csv_to_mergedjson.sh

# Step 5: Run all models (can be run in parallel)
echo "Step 5: Running all models..."
echo "Running frequency-based methods..."
# 🟢
./topfreq.sh &

echo "Running neural methods..."
./dream.sh &
./beacon.sh &
./clea.sh &
./sets2sets.sh &
./dnntsp.sh &
./upcf.sh &
./tifuknn.sh &

wait

# Step 6: Evaluate performance
echo "Step 6: Evaluating performance..."
./performance.sh
./performance_gain.sh

echo "Full pipeline completed!"
