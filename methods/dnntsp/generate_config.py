#!/usr/bin/env python3
import json
import sys
import os

def get_items_total(dataset_name):
    """Dynamically calculate total items from history and future JSON files"""
    # Try different path patterns based on execution context
    possible_paths = [
        f"../../jsondata/{dataset_name.lower()}_history.json",  # From /methods/dnntsp/
        f"../../../jsondata/{dataset_name.lower()}_history.json"  # From /methods/dnntsp/train/
    ]
    
    history_path = None
    future_path = None
    
    # Find the correct path that exists
    for base_path in ["../../jsondata", "../../../jsondata"]:
        history_candidate = f"{base_path}/{dataset_name.lower()}_history.json"
        future_candidate = f"{base_path}/{dataset_name.lower()}_future.json"
        
        if os.path.exists(history_candidate) and os.path.exists(future_candidate):
            history_path = history_candidate
            future_path = future_candidate
            break
    
    if not history_path:
        print(f"Error: Could not find data files for {dataset_name}")
        return 0
    
    items = set()
    
    # Read history file
    try:
        with open(history_path, 'r') as f:
            history_data = json.load(f)
            for user_data in history_data.values():
                for basket in user_data:
                    items.update(basket)
    except FileNotFoundError:
        print(f"Warning: {history_path} not found")
    
    # Read future file
    try:
        with open(future_path, 'r') as f:
            future_data = json.load(f)
            for user_data in future_data.values():
                for basket in user_data:
                    items.update(basket)
    except FileNotFoundError:
        print(f"Warning: {future_path} not found")
    
    # Return max item ID + 1, not just count of unique items
    # This is needed for F.one_hot which expects class indices < num_classes
    return max(items) + 1 if items else 0

def generate_config(dataset_name, fold_id=0):
    """Generate DNNTSP config file for a given dataset and fold"""
    
    # Get total items dynamically
    items_total = get_items_total(dataset_name)
    
    # Use paths that work from train/ directory since that's where training runs
    config = {
        "data": dataset_name.capitalize(),
        "save_model_folder": "DNNTSP",
        "history_path": f"../../../jsondata/{dataset_name.lower()}_history.json",
        "future_path": f"../../../jsondata/{dataset_name.lower()}_future.json",
        "keyset_path": f"../../../keyset/{dataset_name.lower()}_keyset_{fold_id}.json",
        "item_embed_dim": 32,
        "items_total": items_total,
        "cuda": 0,
        "loss_function": "multi_label_soft_loss",
        "epochs": 40,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optim": "Adam",
        "weight_decay": 0
    }
    
    # Write config to file (overwrite the main utils/config.json that load_config.py reads)
    # Always write to the correct utils/config.json regardless of current working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "utils", "config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Generated config for dataset: {dataset_name}, fold: {fold_id}, items_total: {items_total}")
    return config_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_config.py <dataset_name> [fold_id]")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    fold_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    generate_config(dataset_name, fold_id)