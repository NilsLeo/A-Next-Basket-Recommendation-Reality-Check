#!/usr/bin/env python3

import pandas as pd
import os
from dotenv import load_dotenv

def generate_popularity_file(dataset):
    """Generate popularity CSV file for a dataset based on historical data"""
    
    # Read historical data
    history_file = f'../../dataset/{dataset}_history.csv'
    if not os.path.exists(history_file):
        print(f"Error: {history_file} not found")
        return False
    
    print(f"Reading {history_file}")
    data_history = pd.read_csv(history_file)
    
    # Count product occurrences
    product_counts = data_history['product_id'].value_counts().reset_index()
    product_counts.columns = ['product_id', 'count']
    
    # Sort by popularity (count) in descending order
    product_counts = product_counts.sort_values('count', ascending=False)
    
    # Save to popularity file
    pop_file = f'{dataset}_pop.csv'
    product_counts.to_csv(pop_file, index=False)
    print(f"Generated {pop_file} with {len(product_counts)} products")
    print(f"Top 5 products: {product_counts.head()}")
    
    return True

if __name__ == '__main__':
    # Load environment variables from .env file
    load_dotenv('../../.env')
    
    # Get dataset names from environment variable
    dataset_names = os.getenv('DATASET_NAMES', '')
    if not dataset_names:
        print("Error: DATASET_NAMES not found in .env file")
        exit(1)
    
    # Parse comma-separated dataset names
    datasets = [dataset.strip() for dataset in dataset_names.split(',')]
    print(f"Found datasets in .env: {datasets}")
    
    for dataset in datasets:
        print(f"\nGenerating popularity file for {dataset}")
        generate_popularity_file(dataset)