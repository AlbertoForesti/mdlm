#!/bin/bash
# filepath: /home/foresti/mdlm/pair_summaries.sh

echo "Starting to pair summaries for models M0-M23..."

# Loop from 0 to 23
for i in {0..23}; do
  echo "========================================"
  echo "Processing model M$i..."
  echo "========================================"
  
  # Execute the Python script with appropriate paths
  python3 SummEval/data_processing/pair_data.py \
    --model_outputs "/home/foresti/mdlm/model_summaries_unpaired/M$i/M$i" \
    --story_files "/home/foresti/mdlm/"
    
  # Check if the command was successful
  if [ $? -eq 0 ]; then
    echo "✓ Successfully processed M$i"
  else
    echo "✗ Error processing M$i"
    # You might want to add error handling here
  fi
done

echo "========================================"
echo "Pairing process complete for all models!"
echo "========================================"