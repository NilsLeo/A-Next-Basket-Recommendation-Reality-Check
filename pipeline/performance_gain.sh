#!/bin/bash

cd ../evaluation

echo "Running performance gain evaluation..."

# Evaluate each method
methods=("g_top_freq" "p_top_freq" "gp_top_freq" "dream" "beacon" "clea" "sets2sets" "dnntsp" "upcf" "tifuknn")

for method in "${methods[@]}"; do
    echo "Evaluating $method performance gain..."
    python performance_gain.py --pred_folder "../methods/$method" --fold_list "[0, 1, 2]"
done

echo "Performance gain evaluation completed."