#!/bin/bash
# filepath: /home/foresti/mdlm/batch_clean_jsonl.sh

# Help function
function show_help {
  echo "Usage: $0 --input_dir DIR --output_dir DIR"
  echo "Clean and standardize schema for all JSONL files in a directory"
  echo "Organizes output files into model-specific subdirectories (M0, M1, etc.)"
  echo ""
  echo "Required arguments:"
  echo "  --input_dir DIR      Directory containing JSONL files to clean"
  echo "  --output_dir DIR     Directory where cleaned files will be saved"
  echo "  -h, --help           Show this help message"
}

# Parse command line arguments
INPUT_DIR=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --input_dir)
      INPUT_DIR="$2"
      shift
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Check if required arguments are provided
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Error: Missing required arguments"
  show_help
  exit 1
fi

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Input directory does not exist: $INPUT_DIR"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
  echo "Error: Failed to create output directory: $OUTPUT_DIR"
  exit 1
fi

echo "Starting batch processing of JSONL files..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Find all JSONL files in the input directory
jsonl_files=$(find "$INPUT_DIR" -type f -name "*.jsonl")
total_files=$(echo "$jsonl_files" | wc -l)

if [ "$total_files" -eq 0 ]; then
  echo "No JSONL files found in $INPUT_DIR"
  exit 0
fi

echo "Found $total_files JSONL files to process"
echo "----------------------------------------"

# Process each file
count=0
success=0
failed=0
models_processed=()

for file in $jsonl_files; do
  ((count++))
  filename=$(basename "$file")
  
  # Extract model ID from the file path
  model_id=$(python3 -c "
import sys
sys.path.append('$(dirname "$0")')
from clean_and_save_aligned import extract_model_id
print(extract_model_id('$file') or 'unknown')
")
  
  # Create model-specific subdirectory
  model_dir="$OUTPUT_DIR/$model_id"
  mkdir -p "$model_dir"
  
  # Define output file path
  output_file="$model_dir/$filename"
  
  echo "[$count/$total_files] Processing: $filename (Model: $model_id)"
  
  # Add model to processed list if not already there
  if [[ ! " ${models_processed[@]} " =~ " ${model_id} " ]]; then
    models_processed+=("$model_id")
  fi
  
  # Run the Python script to clean the file
  python3 clean_and_save_aligned.py --input_file "$file" --output_file "$output_file"
  
  if [ $? -eq 0 ]; then
    echo "✓ Successfully processed $filename → $model_id/$filename"
    ((success++))
  else
    echo "✗ Failed to process $filename"
    ((failed++))
  fi
  
  echo "----------------------------------------"
done

echo "Processing complete:"
echo "  Total files: $total_files"
echo "  Successfully processed: $success"
echo "  Failed: $failed"
echo "  Models processed: ${models_processed[@]}"
echo "Cleaned files saved to model-specific subdirectories in: $OUTPUT_DIR"