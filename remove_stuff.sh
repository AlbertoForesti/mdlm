#!/bin/bash

# Directory to process
TARGET_DIR="/home/foresti/mdlm/outputs/home/foresti/mdlm/model_summaries_cleaned/M22/outputs_cnndm.aligned.paired.jsonl"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Directory $TARGET_DIR does not exist."
  exit 1
fi

# Create a list of files to delete
echo "Looking for checkpoint files to clean up..."
FILE_LIST=$(find "$TARGET_DIR" -name "*.ckpt" ! -name "last.ckpt" -type f)

# Count files and calculate total size
FILE_COUNT=$(echo "$FILE_LIST" | wc -l)
TOTAL_SIZE=$(du -ch $FILE_LIST 2>/dev/null | grep total)

# Show preview
echo "Found $FILE_COUNT checkpoint files to delete (excluding last.ckpt files)"
echo "Total size: $TOTAL_SIZE"
echo "Files to be deleted:"
echo "$FILE_LIST"

# Ask for confirmation before deletion
read -p "Are you sure you want to proceed with deletion? (y/n): " confirm

if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
  # Delete the files
  echo "$FILE_LIST" | xargs rm -f
  echo "Files deleted successfully."
else
  echo "Operation cancelled."
  exit 0
fi