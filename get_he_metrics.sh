#!/bin/bash
# filepath: /home/foresti/mdlm/run_aggregate_metrics.sh

# Help function
function show_help {
  echo "Usage: $0 [OPTIONS]"
  echo "Process summary files against a human evaluation file"
  echo ""
  echo "Required arguments:"
  echo "  --summary_dir DIR     Directory containing summary files"
  echo "  --he_file FILE        Path to the human evaluation file"
  echo "  --output_dir DIR      Directory to save result JSON files"
  echo "  -h, --help            Show this help message"
}

# Parse command line arguments
SUMMARY_DIR=""
HE_FILE=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --summary_dir)
      SUMMARY_DIR="$2"
      shift
      shift
      ;;
    --he_file)
      HE_FILE="$2"
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

# Check if all required arguments are provided
if [ -z "$SUMMARY_DIR" ] || [ -z "$HE_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Error: Missing required arguments"
  show_help
  exit 1
fi

# Check if the summary directory exists
if [ ! -d "$SUMMARY_DIR" ]; then
  echo "Error: Summary directory not found: $SUMMARY_DIR"
  exit 1
fi

# Check if the HE file exists
if [ ! -f "$HE_FILE" ]; then
  echo "Error: Human evaluation file not found: $HE_FILE"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
  echo "Error: Failed to create output directory: $OUTPUT_DIR"
  exit 1
fi

echo "Starting processing of summary files..."
echo "Summary files directory: $SUMMARY_DIR"
echo "Human evaluation file: $HE_FILE"
echo "Output directory: $OUTPUT_DIR"

# Counter for processed files
processed=0
success=0
failed=0

# Find all JSON and JSONL files in the summary directory
find "$SUMMARY_DIR" -type f -name "*.json*" | while read summary_file; do
  # Extract model ID from the file name or path for the output file name
  model_id=$(python3 -c "
import re
match = re.search(r'/M(\d+)/', '$summary_file')
if match:
  print(f'M{match.group(1)}')
else:
  print('unknown')
")
  
  # Create output file path
  output_file="$OUTPUT_DIR/${model_id}_metrics.json"
  
  echo "Processing: $summary_file"
  echo "  Model ID: $model_id"
  echo "  Output: $output_file"
  
  # Run the Python script
  python3 aggregate_and_pair_he.py --he_file "$HE_FILE" --summary_file "$summary_file" --output_file "$output_file"
  
  # Check if the command was successful
  if [ $? -eq 0 ]; then
    echo "✓ Successfully processed $summary_file"
    ((success++))
  else
    echo "✗ Failed to process $summary_file"
    ((failed++))
  fi
  
  ((processed++))
  echo "----------------------------------------"
done

echo "Processing complete:"
echo "  Total files processed: $processed"
echo "  Successful: $success"
echo "  Failed: $failed"
echo "Results saved to: $OUTPUT_DIR"