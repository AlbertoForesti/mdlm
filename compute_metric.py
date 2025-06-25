import os
import json
import pandas as pd
import yaml
import argparse
from pathlib import Path

def fix_rouge():
    import summ_eval
    import os
    from pathlib import Path

    # Find the actual summ_eval installation path
    summ_eval_path = os.path.dirname(summ_eval.__file__)
    print(f"summ_eval is installed at: {summ_eval_path}")

    # Check if ROUGE-1.5.5 directory exists there
    rouge_dir = os.path.join(summ_eval_path, 'ROUGE-1.5.5')
    print(f"Looking for ROUGE at: {rouge_dir}")
    print(f"ROUGE directory exists: {os.path.exists(rouge_dir)}")

    if os.path.exists(rouge_dir):
        print(f"Contents: {os.listdir(rouge_dir)}")
    else:
        print("ROUGE directory not found - we need to create it")
        
        # Create the ROUGE directory structure
        os.makedirs(rouge_dir, exist_ok=True)
        os.makedirs(os.path.join(rouge_dir, 'data'), exist_ok=True)
        
        # Create minimal required files
        required_files = [
            'ROUGE-1.5.5.pl',
            'data/WordNet-2.0.exc.db',
            'data/smart_common_words.txt'
        ]
        
        for file_path in required_files:
            full_path = os.path.join(rouge_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            Path(full_path).touch()
            print(f"Created: {full_path}")

    # Set the environment variable to the correct path
    os.environ['ROUGE_HOME'] = rouge_dir
    print(f"ROUGE_HOME set to: {rouge_dir}")

    # Now test if pyrouge works
    try:
        import pyrouge
        rouge = pyrouge.Rouge155()
        print("✓ pyrouge.Rouge155() initialized successfully!")
    except Exception as e:
        print(f"✗ Error initializing pyrouge: {e}")

# Download required NLTK data
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

# Mapping of metric names to their module/class information
METRIC_MODULES = {
    'meteor': ('summ_eval.meteor_metric', 'MeteorMetric'),
    # 's3': ('summ_eval.s3_metric', 'S3Metric'),
    'bleu': ('summ_eval.bleu_metric', 'BleuMetric'),
    'blanc': ('summ_eval.blanc_metric', 'BlancMetric'),
    'cider': ('summ_eval.cider_metric', 'CiderMetric'),
    'rouge': ('summ_eval.rouge_metric', 'RougeMetric'),
    'chrfpp': ('summ_eval.chrfpp_metric', 'ChrfppMetric'),
    'supert': ('summ_eval.supert_metric', 'SupertMetric'),
    'rouge_we': ('summ_eval.rouge_we_metric', 'RougeWeMetric'),
    'summa_qa': ('summ_eval.summa_qa_metric', 'SummaQAMetric'),
    'syntactic': ('summ_eval.syntactic_metric', 'SyntacticMetric'),
    'bert_score': ('summ_eval.bert_score_metric', 'BertScoreMetric'),
    'data_stats': ('summ_eval.data_stats_metric', 'DataStatsMetric'),
    'sentence_movers': ('summ_eval.sentence_movers_metric', 'SentenceMoversMetric'),
    'mover_score': ('summ_eval.mover_score_metric', 'MoverScoreMetric'),
    'bleurt': ('summ_eval.bleurt_metric', 'BleurtMetric'),
    'questeval': ('summ_eval.questeval_metric', 'QuestevalMetric'),
}

def import_metric(metric_name):
    """Dynamically import a specific metric"""
    if metric_name not in METRIC_MODULES:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    module_name, class_name = METRIC_MODULES[metric_name]
    
    # Handle special cases that need fix_rouge
    if metric_name in ['rouge', 'rouge_we']:
        fix_rouge()
    
    try:
        import importlib
        module = importlib.import_module(module_name)
        metric_class = getattr(module, class_name)
        print(f"✓ {metric_name} metric import successful")
        return metric_class
    except Exception as e:
        print(f"✗ Failed to import {metric_name} metric: {e}")
        return None

def get_all_available_metrics():
    """Get all metrics that can be imported successfully"""
    available = {}
    for metric_name in METRIC_MODULES.keys():
        metric_class = import_metric(metric_name)
        if metric_class is not None:
            available[metric_name] = metric_class
    return available

def load_yaml_data(file_path):
    """Load data from YAML file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data

def load_jsonl_data(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def extract_texts_from_data(data, file_format='yaml'):
    """Extract model summaries, reference summaries, and source texts from data"""
    model_summaries = []
    reference_summaries = []
    source_texts = []
    
    if file_format == 'yaml':
        # Handle YAML format - assuming it's a list of dictionaries
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # If it's a dict, try to find the data in common keys
            items = data.get('data', data.get('examples', data.get('items', [data])))
        else:
            items = [data]
    else:
        # Handle JSONL format
        items = data
    
    for item in items:
        # Adjust these field names based on your file structure
        model_summary = (item.get('model_summary') or 
                        item.get('generated_summary') or 
                        item.get('prediction') or 
                        item.get('output') or 
                        item.get('decoded') or '')
        
        reference_summary = (item.get('reference_summary') or 
                           item.get('reference') or 
                           item.get('target') or 
                           item.get('summary') or '')
        
        source_text = (item.get('source_text') or 
                      item.get('text') or 
                      item.get('document') or 
                      item.get('input') or '')
        
        model_summaries.append(str(model_summary))
        reference_summaries.append(str(reference_summary))
        source_texts.append(str(source_text))
    
    return model_summaries, reference_summaries, source_texts

def initialize_metrics(metric_names):
    """Initialize multiple metrics by dynamically importing them"""
    metrics = {}
    failed_metrics = []
    
    for metric_name in metric_names:
        try:
            metric_class = import_metric(metric_name)
            if metric_class is None:
                print(f"✗ Metric '{metric_name}' not available (import failed)")
                failed_metrics.append(metric_name)
                continue
                
            metrics[metric_name] = metric_class()
            print(f"✓ {metric_name} metric initialized")
        except Exception as e:
            print(f"✗ Error initializing {metric_name} metric: {str(e)}")
            failed_metrics.append(metric_name)
    
    if failed_metrics:
        print(f"Failed to initialize: {failed_metrics}")
    
    return metrics

def get_metric_requirements(metric_name):
    """Determine what inputs a metric needs based on its name"""
    # Metrics that typically need source text
    source_dependent = ['summa_qa', 'supert', 'blanc', 'data_stats']
    
    if metric_name.lower() in source_dependent:
        return 'source_dependent'
    else:
        return 'summary_reference_only'

def compute_metrics_for_file(file_path, metrics):
    """Compute specified metrics for a single file"""
    print(f"Processing: {os.path.basename(file_path)}")
    
    # Determine file format
    file_format = 'yaml' if file_path.endswith('.yaml') or file_path.endswith('.yml') else 'jsonl'
    
    # Load data
    if file_format == 'yaml':
        data = load_yaml_data(file_path)
    else:
        data = load_jsonl_data(file_path)
    
    print(f"  Loaded data from {file_format} file")
    
    # Extract texts
    model_summaries, reference_summaries, source_texts = extract_texts_from_data(data, file_format)
    
    # Verify we have valid data
    valid_pairs = [(m, r) for m, r in zip(model_summaries, reference_summaries) 
                   if m.strip() and r.strip()]
    
    if len(valid_pairs) == 0:
        print("  No valid summary pairs found")
        return {}
    
    model_summaries, reference_summaries = zip(*valid_pairs)
    model_summaries = list(model_summaries)
    reference_summaries = list(reference_summaries)
    
    print(f"  Found {len(valid_pairs)} valid summary pairs")
    
    results = {}
    
    # Compute each metric
    for metric_name, metric_instance in metrics.items():
        try:
            print(f"    Computing {metric_name}...")
            
            # Determine metric requirements
            metric_type = get_metric_requirements(metric_name)
            
            # Try different calling patterns based on metric requirements
            if metric_type == 'source_dependent':
                scores = metric_instance.evaluate_batch(model_summaries, source_texts)
            else:
                # Standard call with just summaries and references
                scores = metric_instance.evaluate_batch(model_summaries, reference_summaries)
            
            # Store results
            results[metric_name] = scores
            
            # Print summary statistics
            if isinstance(scores, dict):
                for score_type, score_values in scores.items():
                    if isinstance(score_values, list) and len(score_values) > 0:
                        avg_score = sum(score_values) / len(score_values)
                        print(f"      {score_type}: {avg_score:.4f}")
                    elif isinstance(score_values, (int, float)):
                        print(f"      {score_type}: {score_values:.4f}")
            elif isinstance(scores, list) and len(scores) > 0:
                avg_score = sum(scores) / len(scores)
                print(f"      Average: {avg_score:.4f}")
            
            print(f"    ✓ {metric_name} completed")
            
        except Exception as e:
            print(f"    ✗ Error computing {metric_name}: {str(e)}")
            results[metric_name] = None
    
    return results

def save_results_to_csv(all_results, output_path, metrics_computed):
    """Save results to CSV files"""
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a summary table
    summary_rows = []
    
    for file_name, file_results in all_results.items():
        for computed_metric_name, scores in file_results.items():
            if scores is not None:
                if isinstance(scores, dict):
                    # Handle metrics that return multiple scores (like ROUGE)
                    for score_name, score_values in scores.items():
                        if isinstance(score_values, list) and len(score_values) > 0:
                            avg_score = sum(score_values) / len(score_values)
                            summary_rows.append({
                                'file': file_name,
                                'metric': f"{computed_metric_name}_{score_name}",
                                'average_score': avg_score,
                                'num_examples': len(score_values)
                            })
                        elif isinstance(score_values, (int, float)):
                            # Single score
                            summary_rows.append({
                                'file': file_name,
                                'metric': f"{computed_metric_name}_{score_name}",
                                'average_score': score_values,
                                'num_examples': 1
                            })
                elif isinstance(scores, list) and len(scores) > 0:
                    # Single metric with list of scores
                    avg_score = sum(scores) / len(scores)
                    summary_rows.append({
                        'file': file_name,
                        'metric': computed_metric_name,
                        'average_score': avg_score,
                        'num_examples': len(scores)
                    })
                elif isinstance(scores, (int, float)):
                    # Single score value
                    summary_rows.append({
                        'file': file_name,
                        'metric': computed_metric_name,
                        'average_score': scores,
                        'num_examples': 1
                    })
    
    # Save summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
        # Create a pivot table for easy comparison across files
        base_name = os.path.splitext(output_path)[0]
        pivot_path = f"{base_name}_comparison.csv"
        
        try:
            pivot_df = summary_df.pivot(index='file', columns='metric', values='average_score')
            pivot_df.to_csv(pivot_path)
            print(f"Comparison table saved to: {pivot_path}")
        except Exception as e:
            print(f"Could not create pivot table: {e}")
        
        # Also save detailed results
        detailed_path = f"{base_name}_detailed.csv"
        
        # Save detailed results for each file
        detailed_rows = []
        for file_name, file_results in all_results.items():
            for computed_metric_name, scores in file_results.items():
                if scores is not None:
                    if isinstance(scores, dict):
                        for score_name, score_values in scores.items():
                            if isinstance(score_values, list):
                                for i, score in enumerate(score_values):
                                    detailed_rows.append({
                                        'file': file_name,
                                        'example_id': i,
                                        'metric': f"{computed_metric_name}_{score_name}",
                                        'score': score
                                    })
                    elif isinstance(scores, list):
                        for i, score in enumerate(scores):
                            detailed_rows.append({
                                'file': file_name,
                                'example_id': i,
                                'metric': computed_metric_name,
                                'score': score
                            })
        
        if detailed_rows:
            detailed_df = pd.DataFrame(detailed_rows)
            detailed_df.to_csv(detailed_path, index=False)
            print(f"Detailed results saved to: {detailed_path}")
        else:
            print("No detailed results to save")
    else:
        print("No results to save")

def parse_metrics(metric_args):
    """Parse metric arguments to return list of metrics"""
    if 'all' in metric_args:
        return list(METRIC_MODULES.keys())
    else:
        return metric_args

def list_available_metrics():
    """List all available metrics that can be imported"""
    print("Available metrics:")
    for metric_name in METRIC_MODULES.keys():
        print(f"  - {metric_name}")
    print("\nUse 'all' to compute all available metrics.")
    print("Example: python compute_metric.py data/ -m bleu rouge meteor -o results.csv")

def main():
    """Main function to process files and compute metrics"""
    parser = argparse.ArgumentParser(description='Compute evaluation metrics for summary files')
    parser.add_argument('input_path', nargs='?', help='Path to YAML/JSONL file or directory containing such files')
    parser.add_argument('--metric', '-m', nargs='+', 
                        choices=list(METRIC_MODULES.keys()) + ['all'],
                        help='Metric(s) to compute. Use "all" for all metrics, or specify multiple: -m bleu rouge meteor')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--list-metrics', action='store_true', help='List all available metrics and exit')
    
    args = parser.parse_args()
    
    # List metrics if requested
    if args.list_metrics:
        list_available_metrics()
        return
    
    # Validate required arguments
    if not args.input_path:
        parser.error("input_path is required unless using --list-metrics")
    if not args.metric:
        parser.error("--metric is required unless using --list-metrics")
    if not args.output:
        parser.error("--output is required unless using --list-metrics")
    
    # Parse metrics
    metrics_to_compute = parse_metrics(args.metric)
    
    print(f"Computing metrics: {metrics_to_compute}")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output}")
    
    # Initialize the specified metrics
    metrics = initialize_metrics(metrics_to_compute)
    
    if not metrics:
        print("No metrics were successfully initialized. Exiting.")
        return
    
    all_results = {}
    total_files = 0
    
    # Check if input is a file or directory
    if os.path.isfile(args.input_path):
        # Single file
        file_path = args.input_path
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        print(f"\n=== Processing file: {file_path} ===")
        results = compute_metrics_for_file(file_path, metrics)
        all_results[file_name] = results
        total_files = 1
        
    elif os.path.isdir(args.input_path):
        # Directory - find all YAML and JSONL files
        for root, dirs, files in os.walk(args.input_path):
            for file_name in files:
                if file_name.endswith(('.yaml', '.yml', '.jsonl')):
                    file_path = os.path.join(root, file_name)
                    
                    # Create a unique model name including the subdirectory
                    relative_path = os.path.relpath(file_path, args.input_path)
                    model_name = os.path.splitext(relative_path)[0].replace(os.sep, '_')
                    
                    print(f"\n=== Processing file: {relative_path} ===")
                    
                    # Compute metrics for this file
                    results = compute_metrics_for_file(file_path, metrics)
                    all_results[model_name] = results
                    total_files += 1
    else:
        print(f"Error: {args.input_path} is not a valid file or directory")
        return
    
    print(f"\n=== Processing Complete ===")
    print(f"Processed {total_files} files")
    print(f"Successfully computed: {list(metrics.keys())}")
    
    if all_results:
        # Save all results
        save_results_to_csv(all_results, args.output, list(metrics.keys()))
    else:
        print("No results to save")

if __name__ == "__main__":
    main()