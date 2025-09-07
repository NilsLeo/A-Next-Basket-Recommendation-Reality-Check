#!/usr/bin/env python3
"""
Convert CSV dataset format to separated JSON format for NBR methods.
Separated format: {user_id: [[-1], basket1, basket2, ..., [-1]], ...}
Used by: Sets2Sets, TIFUKNN, DNNTSP, DREAM
"""

import pandas as pd
import json
from collections import defaultdict
import os
from dotenv import load_dotenv

def csv_to_separated_json(dataset_name):
    """
    Convert history and future CSV files to separated JSON format.
    
    Args:
        dataset_name: Name of dataset (dunnhumby, instacart, tafeng)
    """
    history_csv_path = f'../dataset/{dataset_name}_history.csv'
    future_csv_path = f'../dataset/{dataset_name}_future.csv'
    history_json_path = f'../jsondata/{dataset_name}_history.json'
    future_json_path = f'../jsondata/{dataset_name}_future.json'
    
    print(f"Reading history data from {history_csv_path}")
    history_df = pd.read_csv(history_csv_path)
    
    print(f"Reading future data from {future_csv_path}")
    future_df = pd.read_csv(future_csv_path)
    
    # Process history data
    print("Processing history baskets...")
    history_user_baskets = defaultdict(list)
    
    for (user_id, order_num), group in history_df.groupby(['user_id', 'order_number']):
        basket = group['product_id'].tolist()
        history_user_baskets[str(user_id)].append((order_num, basket))
    
    # Sort history baskets by order number for each user and add [-1] markers
    for user_id in history_user_baskets:
        # Sort by order number
        history_user_baskets[user_id].sort(key=lambda x: x[0])
        # Extract just the baskets and add [-1] markers
        baskets = [basket for _, basket in history_user_baskets[user_id]]
        history_user_baskets[user_id] = [[-1]] + baskets + [[-1]]
    
    # Process future data
    print("Processing future baskets...")
    future_user_baskets = defaultdict(list)
    
    for (user_id, order_num), group in future_df.groupby(['user_id', 'order_number']):
        basket = group['product_id'].tolist()
        future_user_baskets[str(user_id)].append((order_num, basket))
    
    # Sort future baskets by order number for each user and add [-1] markers
    for user_id in future_user_baskets:
        # Sort by order number
        future_user_baskets[user_id].sort(key=lambda x: x[0])
        # Extract just the baskets and add [-1] markers
        baskets = [basket for _, basket in future_user_baskets[user_id]]
        future_user_baskets[user_id] = [[-1]] + baskets + [[-1]]
    
    # Create output directory
    os.makedirs('../jsondata', exist_ok=True)
    
    # Write history JSON
    print(f"Writing history JSON to {history_json_path}")
    with open(history_json_path, 'w') as f:
        json.dump(dict(history_user_baskets), f)
    
    # Write future JSON
    print(f"Writing future JSON to {future_json_path}")
    with open(future_json_path, 'w') as f:
        json.dump(dict(future_user_baskets), f)
    
    print(f"Conversion complete. Processed {len(history_user_baskets)} users for history, {len(future_user_baskets)} users for future")

if __name__ == '__main__':
    # Load environment variables
    load_dotenv('../.env')
    
    # Get datasets from environment variable
    dataset_names = os.getenv('DATASET_NAMES', 'tafeng,instacart')
    datasets = [name.strip() for name in dataset_names.split(',')]
    
    for dataset in datasets:
        print(f"\n=== Processing {dataset} ===")
        try:
            csv_to_separated_json(dataset)
        except Exception as e:
            print(f"Error processing {dataset}: {e}")