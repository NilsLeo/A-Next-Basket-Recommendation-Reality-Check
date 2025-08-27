#!/usr/bin/env python3
"""
Convert CSV dataset format to merged JSON format for NBR methods.
Merged format: {user_id: [basket1, basket2, ..., basketN], ...}
Used by: BEACON, CLEA, UP-CF
"""

import pandas as pd
import json
from collections import defaultdict
import os

def csv_to_merged_json(dataset_name):
    """
    Convert history and future CSV files to merged JSON format.
    
    Args:
        dataset_name: Name of dataset (dunnhumby, instacart, tafeng)
    """
    history_csv_path = f'../dataset/{dataset_name}_history.csv'
    future_csv_path = f'../dataset/{dataset_name}_future.csv'
    output_json_path = f'../mergeddataset/{dataset_name}_merged.json'
    
    print(f"Reading history data from {history_csv_path}")
    history_df = pd.read_csv(history_csv_path)
    
    print(f"Reading future data from {future_csv_path}")
    future_df = pd.read_csv(future_csv_path)
    
    # Combine history and future data
    combined_df = pd.concat([history_df, future_df], ignore_index=True)
    
    # Group by user_id and order_number to create baskets
    user_baskets = defaultdict(list)
    
    print("Processing baskets...")
    for (user_id, order_num), group in combined_df.groupby(['user_id', 'order_number']):
        basket = group['product_id'].tolist()
        user_baskets[str(user_id)].append(basket)
    
    # Sort baskets by order number for each user
    for user_id in user_baskets:
        # Get order numbers for this user
        user_orders = combined_df[combined_df['user_id'] == int(user_id)]['order_number'].unique()
        user_orders.sort()
        
        # Rebuild baskets in correct order
        sorted_baskets = []
        for order_num in user_orders:
            basket = combined_df[
                (combined_df['user_id'] == int(user_id)) & 
                (combined_df['order_number'] == order_num)
            ]['product_id'].tolist()
            sorted_baskets.append(basket)
        
        user_baskets[user_id] = sorted_baskets
    
    print(f"Writing merged JSON to {output_json_path}")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    with open(output_json_path, 'w') as f:
        json.dump(user_baskets, f)
    
    print(f"Conversion complete. Processed {len(user_baskets)} users")

if __name__ == '__main__':
    datasets = ['tafeng','dunnhumby', 'instacart']
    
    for dataset in datasets:
        print(f"\n=== Processing {dataset} ===")
        try:
            csv_to_merged_json(dataset)
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
