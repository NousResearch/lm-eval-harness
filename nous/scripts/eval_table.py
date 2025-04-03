import json
import fire
import statistics
import os
import glob
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.text import Text

def calculate_metrics_table(
    paths: Union[List[str], str], 
    metric: str = "exact_match",
    higher_is_better: bool = True
):
    """
    Calculate metrics from multiple lm-eval-harness results files and display in a table.
    
    Args:
        paths: List of paths, comma-separated string of paths to results JSON files, 
               or a single directory path containing JSON files
        metric: Metric name to calculate (defaults to 'exact_match')
        higher_is_better: Whether higher values are better (True) or lower values are better (False)
    """
    console = Console()
    
    # Handle the case where paths might be a single string
    if isinstance(paths, str):
        # Check if it's a directory
        if os.path.isdir(paths):
            # Get all JSON files in the directory
            path_list = glob.glob(os.path.join(paths, "*.json"))
            console.print(f"Found {len(path_list)} JSON files in directory: {paths}")
        else:
            # Treat as comma-separated paths
            path_list = [p.strip() for p in paths.split(',')]
    else:
        # Handle the case where paths is already a list
        path_list = paths
        # Fire might pass a list of individual characters if there's only one path
        if len(path_list) > 1 and all(len(p) == 1 for p in path_list):
            path_list = [''.join(path_list)]
            # Check if the joined path is a directory
            if os.path.isdir(path_list[0]):
                path_list = glob.glob(os.path.join(path_list[0], "*.json"))
                console.print(f"Found {len(path_list)} JSON files in directory: {path_list[0]}")
    
    console.print(f"Processing {len(path_list)} results files: {', '.join(path_list)}")
    
    # Create a dictionary to store results by task for each file
    all_results = {}
    # Keep track of all task names we encounter
    all_tasks = set()
    # Store the file stems to use as column headers
    file_stems = []
    
    # Process each results file
    for path in path_list:
        file_path = Path(path)
        file_stem = file_path.stem
        file_stems.append(file_stem)
        
        # Load the results JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract results for the specified metric
        file_results = {}
        for task_name, task_data in data.get("results", {}).items():
            task_alias = task_data.get("alias", task_name)
            for key, value in task_data.items():
                # Skip 'alias' field and stderr values
                if key == "alias" or "stderr" in key:
                    continue
                
                # Check if this is the metric we're looking for
                if key.startswith(f"{metric},"):
                    # Ensure the value is numeric
                    if isinstance(value, (int, float)):
                        file_results[task_alias] = value
                        all_tasks.add(task_alias)
        
        all_results[file_stem] = file_results
    
    # Create a pretty table
    table = Table(title=f"Results for metric: {metric}")
    
    # Add task column
    table.add_column("Task", style="bold")
    
    # Add columns for each file
    for file_stem in file_stems:
        table.add_column(file_stem, justify="right")
    
    # Sort task names for consistent display
    sorted_tasks = sorted(all_tasks)
    
    # Fill table with results
    for task in sorted_tasks:
        # Get results for this task across all files
        task_results = {}
        for file_stem in file_stems:
            if task in all_results[file_stem]:
                task_results[file_stem] = all_results[file_stem][task]
        
        # Skip empty tasks
        if not task_results:
            continue
            
        # Determine the best value for this task
        if task_results:
            if higher_is_better:
                best_value = max(task_results.values())
            else:
                best_value = min(task_results.values())
        
        # Add row for this task
        row = [task]
        for file_stem in file_stems:
            if file_stem in task_results:
                value = task_results[file_stem]
                # Highlight the best value in cyan
                if value == best_value:
                    row.append(Text(f"{value:.4f}", style="cyan"))
                else:
                    row.append(f"{value:.4f}")
            else:
                row.append("-")
        
        table.add_row(*row)
    
    # Calculate averages
    averages = []
    for file_stem in file_stems:
        file_results = all_results[file_stem]
        if file_results:
            avg = statistics.mean(file_results.values())
            averages.append(avg)
        else:
            averages.append(None)
    
    # Determine the best average
    valid_averages = [avg for avg in averages if avg is not None]
    if valid_averages:
        if higher_is_better:
            best_avg = max(valid_averages)
        else:
            best_avg = min(valid_averages)
    
    # Add average row
    avg_row = ["AVERAGE"]
    for i, file_stem in enumerate(file_stems):
        if averages[i] is not None:
            # Highlight the best average in cyan
            if averages[i] == best_avg:
                avg_row.append(Text(f"{averages[i]:.4f}", style="cyan bold"))
            else:
                avg_row.append(Text(f"{averages[i]:.4f}", style="bold"))
        else:
            avg_row.append("-")
    
    table.add_row(*avg_row)
    
    # Display the table
    console.print(table)

def main(
    paths: Union[List[str], str], 
    metric: str = "exact_match",
    higher_is_better: bool = True
):
    """
    Display metrics from multiple lm-eval-harness results files in a pretty table.
    
    Args:
        paths: Comma-separated list of paths to results JSON files, or a directory 
               containing JSON files to process
        metric: Metric name to calculate (defaults to 'exact_match')
        higher_is_better: Whether higher values are better (defaults to True)
    """
    calculate_metrics_table(paths, metric, higher_is_better)

if __name__ == "__main__":
    fire.Fire(main)
