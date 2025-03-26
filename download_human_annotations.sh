ANNOTATION_URL="https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl"
DOWNLOAD_DIR="./summeval_human_annotations"

# Create download directory if it doesn't exist
mkdir -p "$DOWNLOAD_DIR"
if [ $? -ne 0 ]; then
  echo "Error: Failed to create download directory"
  exit 1
fi

echo "Downloading human annotations for CNN/Daily Mail dataset to $DOWNLOAD_DIR"

# Download annotations file
echo "Downloading human annotations..."
wget "$ANNOTATION_URL" -O "$DOWNLOAD_DIR/model_annotations.aligned.jsonl"
