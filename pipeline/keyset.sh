#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Go up one directory
cd ..

# Source environment variables
source .env

# Parse dataset names from env variable
IFS=',' read -ra DATASET_ARRAY <<< "$DATASET_NAMES"

for dataset in "${DATASET_ARRAY[@]}"; do
    for fold_id in 0 1 2; do
        python keyset_fold.py --dataset $dataset --fold_id $fold_id
    done
done
