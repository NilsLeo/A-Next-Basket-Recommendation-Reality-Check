#!/usr/bin/env python3
"""
Generate a markdown comparison table from evaluation results files.
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import glob


def parse_eval_file(filepath: str) -> Dict[str, Any]:
    """Parse a single evaluation results file."""
    method_name = extract_method_name(filepath)
    results = {'method': method_name, 'datasets': {}}
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by dataset sections
    dataset_sections = re.split(r'############(\w+)###########', content)[1:]  # Skip first empty element
    
    for i in range(0, len(dataset_sections), 2):
        dataset_name = dataset_sections[i].strip()
        dataset_content = dataset_sections[i + 1]
        
        results['datasets'][dataset_name] = parse_dataset_section(dataset_content)
    
    return results


def extract_method_name(filepath: str) -> str:
    """Extract method name from file path."""
    filename = Path(filepath).stem
    # Extract method name from eval_{method}_results.txt pattern
    match = re.search(r'eval_(.+)_results', filename)
    if match:
        return match.group(1)
    return filename


def parse_dataset_section(content: str) -> Dict[int, Dict[str, float]]:
    """Parse dataset section content for different basket sizes."""
    results = {}
    
    # Split by basket size sections
    lines = content.strip().split('\n')
    current_size = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for basket size line
        size_match = re.search(r'basket size:\s*(\d+)', line)
        if size_match:
            current_size = int(size_match.group(1))
            results[current_size] = {}
            continue
        
        # Look for metrics line
        metrics_match = re.search(r'recall, ndcg, hit:\s*([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)', line)
        if metrics_match and current_size is not None:
            results[current_size]['recall'] = float(metrics_match.group(1))
            results[current_size]['ndcg'] = float(metrics_match.group(2))
            results[current_size]['hit'] = float(metrics_match.group(3))
        
        # Look for repeat-explore ratio
        ratio_match = re.search(r'repeat-explore ratio:([\d\.]+)\s+([\d\.]+)', line)
        if ratio_match and current_size is not None:
            results[current_size]['repeat_ratio'] = float(ratio_match.group(1))
            results[current_size]['explore_ratio'] = float(ratio_match.group(2))
        
        # Look for repeat-explore recall
        rep_recall_match = re.search(r'repeat-explore recall([\d\.]+)\s+([\d\.]+)', line)
        if rep_recall_match and current_size is not None:
            results[current_size]['repeat_recall'] = float(rep_recall_match.group(1))
            results[current_size]['explore_recall'] = float(rep_recall_match.group(2))
        
        # Look for repeat-explore hit
        rep_hit_match = re.search(r'repeat-explore hit:([\d\.]+)\s+([\d\.]+)', line)
        if rep_hit_match and current_size is not None:
            results[current_size]['repeat_hit'] = float(rep_hit_match.group(1))
            results[current_size]['explore_hit'] = float(rep_hit_match.group(2))
    
    return results


def find_best_values(all_results: List[Dict]) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    """Find the best values for each metric across all methods."""
    best_values = {}
    
    # Get all datasets and metrics
    datasets = set()
    metrics = set()
    sizes = set()
    
    for result in all_results:
        datasets.update(result['datasets'].keys())
        for dataset_data in result['datasets'].values():
            sizes.update(dataset_data.keys())
            for size_data in dataset_data.values():
                metrics.update(size_data.keys())
    
    # Initialize best values structure
    for dataset in datasets:
        best_values[dataset] = {}
        for size in sizes:
            best_values[dataset][size] = {}
            for metric in metrics:
                best_values[dataset][size][metric] = float('-inf')
    
    # Find best values
    for result in all_results:
        for dataset, dataset_data in result['datasets'].items():
            for size, size_data in dataset_data.items():
                for metric, value in size_data.items():
                    if value > best_values[dataset][size][metric]:
                        best_values[dataset][size][metric] = value
    
    return best_values


def format_value(value: float, is_best: bool = False) -> str:
    """Format a value for display in markdown table."""
    formatted = f"{value:.4f}"
    if is_best:
        formatted = f"**{formatted}**"
    return formatted


def generate_markdown_table(all_results: List[Dict], output_file: str):
    """Generate markdown comparison table."""
    if not all_results:
        print("No results to process")
        return
    
    best_values = find_best_values(all_results)
    
    # Sort methods for consistent ordering
    all_results.sort(key=lambda x: x['method'])
    
    # Get all datasets and sizes
    datasets = sorted(set().union(*(result['datasets'].keys() for result in all_results)))
    sizes = sorted(set().union(*(
        set().union(*(dataset_data.keys() for dataset_data in result['datasets'].values()))
        for result in all_results
    )))
    
    with open(output_file, 'w') as f:
        f.write("# NBR Methods Comparison Table\n\n")
        f.write("*Generated automatically from evaluation results*\n\n")
        
        for dataset in datasets:
            f.write(f"## {dataset.capitalize()} Dataset\n\n")
            
            # Create table header
            header = "| Method |"
            separator = "|--------|"
            
            for size in sizes:
                header += f" **Size {size}** | | | | | |"
                separator += "---------|------|-----|------------|------------|"
            
            f.write(header + "\n")
            f.write(separator + "\n")
            
            # Subheader with metric names
            subheader = "|        |"
            for size in sizes:
                subheader += " Recall | NDCG | Hit | Rep.Ratio | Rep.Recall |"
            f.write(subheader + "\n")
            
            # Data rows
            for result in all_results:
                if dataset not in result['datasets']:
                    continue
                    
                row = f"| {result['method']} |"
                
                for size in sizes:
                    if size in result['datasets'][dataset]:
                        data = result['datasets'][dataset][size]
                        
                        # Format each metric
                        recall = format_value(
                            data.get('recall', 0),
                            data.get('recall', 0) == best_values[dataset][size].get('recall', float('-inf'))
                        )
                        ndcg = format_value(
                            data.get('ndcg', 0),
                            data.get('ndcg', 0) == best_values[dataset][size].get('ndcg', float('-inf'))
                        )
                        hit = format_value(
                            data.get('hit', 0),
                            data.get('hit', 0) == best_values[dataset][size].get('hit', float('-inf'))
                        )
                        rep_ratio = format_value(data.get('repeat_ratio', 0))
                        rep_recall = format_value(data.get('repeat_recall', 0))
                        
                        row += f" {recall} | {ndcg} | {hit} | {rep_ratio} | {rep_recall} |"
                    else:
                        row += " - | - | - | - | - |"
                
                f.write(row + "\n")
            
            f.write("\n")
        
        # Add summary section
        f.write("## Summary\n\n")
        f.write("- **Bold** values indicate the best performance for each metric\n")
        f.write("- Rep.Ratio = Repetition Ratio (proportion of repeat items in predictions)\n")
        f.write("- Rep.Recall = Recall performance on repeat items only\n")
        f.write("- Methods are grouped by approach: frequency-based, nearest-neighbor, deep learning\n\n")


def main():
    parser = argparse.ArgumentParser(description='Generate comparison table from evaluation results')
    parser.add_argument('--results_dir', type=str, default='.', 
                       help='Directory containing eval_*_results.txt files')
    parser.add_argument('--output', type=str, default='comparison_table.md',
                       help='Output markdown file')
    
    args = parser.parse_args()
    
    # Find all evaluation result files
    pattern = os.path.join(args.results_dir, 'eval_*_results.txt')
    result_files = glob.glob(pattern)
    
    if not result_files:
        print(f"No evaluation result files found in {args.results_dir}")
        print(f"Looking for pattern: {pattern}")
        return
    
    print(f"Found {len(result_files)} result files:")
    for f in result_files:
        print(f"  - {f}")
    
    # Parse all result files
    all_results = []
    for filepath in result_files:
        try:
            result = parse_eval_file(filepath)
            all_results.append(result)
            print(f"Parsed {result['method']}: {list(result['datasets'].keys())}")
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
    
    if not all_results:
        print("No results successfully parsed")
        return
    
    # Generate markdown table
    output_path = os.path.join(args.results_dir, args.output)
    generate_markdown_table(all_results, output_path)
    print(f"Generated comparison table: {output_path}")


if __name__ == '__main__':
    main()