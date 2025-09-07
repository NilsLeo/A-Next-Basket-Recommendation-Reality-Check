#!/bin/bash
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

cd ../evaluation

echo "Running performance evaluation..."

# Load .env file to get methods dynamically
if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
fi

# Get methods from environment variable, default to completed ones
METHODS_LIST=${METHODS:-"beacon,dream,g-p-gp-topfreq,tifuknn,upcf"}

# Convert comma-separated string to array
IFS=',' read -ra methods <<< "$METHODS_LIST"

for method in "${methods[@]}"; do
    echo "Evaluating $method performance..."
    python model_performance.py --pred_folder "../methods/$method" --fold_list "[0, 1, 2]"
done

echo "Performance evaluation completed."

# Generate comparison table from all evaluation results
echo "Generating comparison table..."
python generate_comparison_table.py --results_dir . --output comparison_table.md
echo "Comparison table generated: evaluation/comparison_table.md"
