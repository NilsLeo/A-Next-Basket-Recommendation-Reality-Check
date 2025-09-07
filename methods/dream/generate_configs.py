#!/usr/bin/env python3

import json
import os
import sys

def load_env_file(env_path):
    """Load environment variables from .env file"""
    env_vars = {}
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
    except FileNotFoundError:
        print(f"Error: .env file not found at {env_path}")
        sys.exit(1)
    return env_vars

def generate_config(dataset_name):
    """Generate configuration for a given dataset"""
    config = {
        "model_name": "nbr",
        "loss_uplift": 100,
        "embedding_size": 32,
        "embedding_type": "mean",
        "hidden_size": 32,
        "dropout_prob": 0.1,
        "max_len": 50,
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32,
        "eval_step": 1,
        "stopping_step": 5,
        "valid_metric_bigger": 1,
        "valid_metric": "recall20",
        "log_file": f"logs/{dataset_name}/test.log",
        "train_file": f"../../jsondata/{dataset_name}_history.json",
        "tgt_file": f"../../jsondata/{dataset_name}_future.json",
        "data_config_file": f"../../keyset/{dataset_name}_keyset_",
        "item_candidate": f"../../datacand/{dataset_name}_item",
        "user_candidate": f"../../datacand/{dataset_name}_user",
        "checkpoint_dir": f"models/{dataset_name}/"
    }
    return config

def main():
    # Get the path to .env file (two directories up from methods/dream/)
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    
    # Load environment variables
    env_vars = load_env_file(env_path)
    
    # Get dataset names
    if 'DATASET_NAMES' not in env_vars:
        print("Error: DATASET_NAMES not found in .env file")
        sys.exit(1)
    
    dataset_names = [name.strip() for name in env_vars['DATASET_NAMES'].split(',')]
    
    # Generate config files for each dataset
    for dataset_name in dataset_names:
        if not dataset_name:  # Skip empty names
            continue
            
        config = generate_config(dataset_name)
        config_filename = f"{dataset_name}conf.json"
        
        # Write config file
        with open(config_filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Generated config file: {config_filename}")
    
    print(f"Successfully generated config files for datasets: {', '.join(dataset_names)}")

if __name__ == "__main__":
    main()