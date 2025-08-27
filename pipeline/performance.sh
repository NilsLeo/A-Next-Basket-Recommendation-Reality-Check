#!/bin/bash

cd ../evaluation

echo "Running performance evaluation..."

# Evaluate each method
methods=("g_top_freq" "p_top_freq" "gp_top_freq" "dream" "beacon" "clea" "sets2sets" "dnntsp" "upcf" "tifuknn")

for method in "${methods[@]}"; do
    echo "Evaluating $method performance..."
    python model_performance.py --pred_folder "../methods/$method" --fold_list "[0, 1, 2]"
done

echo "Performance evaluation completed."