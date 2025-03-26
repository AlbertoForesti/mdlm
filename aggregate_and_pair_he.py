import argparse
import os
import re
import json
import statistics

def extract_model_id(path):
    """Extract model ID (like M7) from a path"""
    # Use regex to find MX pattern in the path
    match = re.search(r'/M(\d+)/', path)
    if match:
        return f"M{match.group(1)}"
    return None

def main():
    parser = argparse.ArgumentParser(description='Aggregate and pair HE images')
    parser.add_argument('--he_file', type=str, required=True, help='File with human evaluation of summaries')
    parser.add_argument('--summary_file', type=str, required=True, help='Path to summary file')
    parser.add_argument('--output_file', type=str, help='Optional output file path')

    args = parser.parse_args()

    model_id = extract_model_id(args.summary_file)
    if not model_id:
        raise ValueError(f"Model ID not found in {args.summary_file}")
    
    print(f"Analyzing expert annotations for model {model_id}")
    
    # First, load all summaries from the summary file into a dictionary by filepath
    summaries = {}
    with open(args.summary_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if 'filepath' in entry and 'decoded' in entry:
                    summaries[entry['filepath']] = entry['decoded']
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON in summary file: {line[:50]}...")
    
    print(f"Loaded {len(summaries)} summaries from {args.summary_file}")
    
    # Initialize counters for expert annotation metrics
    metric_sums = {
        'coherence': 0,
        'consistency': 0,
        'fluency': 0, 
        'relevance': 0
    }
    metric_values = {
        'coherence': [],
        'consistency': [],
        'fluency': [],
        'relevance': []
    }
    
    entry_count = 0
    annotation_count = 0
    matched_count = 0
    
    # Process the HE JSONL file
    with open(args.he_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                
                # Skip entries that don't match our model_id
                if entry.get('model_id') != model_id:
                    continue
                
                # Check if this entry has a filepath and a decoded field that matches the summary file
                filepath = entry.get('filepath')
                decoded = entry.get('decoded')
                
                if not filepath or not decoded or filepath not in summaries:
                    continue
                
                # Check if decoded text matches
                if decoded != summaries[filepath]:
                    print(f"Warning: Decoded text does not match for {filepath}")
                    print(f"Exiting...")
                    return
                
                # This is a valid match
                matched_count += 1
                entry_count += 1
                
                # Process expert annotations
                expert_annotations = entry.get('expert_annotations', [])
                for annotation in expert_annotations:
                    for metric, value in annotation.items():
                        if metric in metric_sums:
                            metric_sums[metric] += value
                            metric_values[metric].append(value)
                    annotation_count += 1
                
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON in HE file: {line[:50]}...")
    
    # Calculate mean scores
    results = {}
    if annotation_count > 0:
        metric_means = {metric: sum_value / annotation_count 
                       for metric, sum_value in metric_sums.items()}
        
        # Calculate standard deviations
        metric_stds = {metric: statistics.stdev(values) if len(values) > 1 else 0 
                      for metric, values in metric_values.items()}
        
        # Print results
        print(f"\nResults for {model_id} (based on {annotation_count} expert annotations across {entry_count} entries):")
        print(f"Matched {matched_count} entries between the HE file and summary file")
        print(f"{'Metric':<12} {'Mean':<8} {'StdDev':<8}")
        print("-" * 30)
        for metric in sorted(metric_means.keys()):
            print(f"{metric:<12} {metric_means[metric]:.2f}     {metric_stds[metric]:.2f}")
        
        # Prepare results dictionary
        results = {
            "model_id": model_id,
            "summary_file": args.summary_file,
            "he_file": args.he_file,
            "entries_processed": entry_count,
            "annotation_count": annotation_count,
            "matched_entries": matched_count,
            "metrics": {
                metric: {
                    "mean": round(metric_means[metric], 4),
                    "std_dev": round(metric_stds[metric], 4)
                } for metric in metric_means
            }
        }
        
        # Write results to file if specified
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output_file}")
    else:
        print(f"No matching expert annotations found for model {model_id}")
    
    return results

if __name__ == '__main__':
    main()