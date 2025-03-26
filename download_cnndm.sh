#!/bin/bash
# filepath: /home/foresti/mdlm/download_cnndm.sh

# Default download directory
DOWNLOAD_DIR="./cnndm"
CNN_URL="https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ"
DAILYMAIL_URL="https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs"

# Create download directory if it doesn't exist
mkdir -p "$DOWNLOAD_DIR"
if [ $? -ne 0 ]; then
  echo "Error: Failed to create download directory"
  exit 1
fi

echo "Downloading CNN/Daily Mail dataset to $DOWNLOAD_DIR"

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
  echo "Installing gdown package for Google Drive downloads..."
  pip install gdown
fi

# Download CNN file
echo "Downloading CNN stories..."
gdown "$CNN_URL" -O "$DOWNLOAD_DIR/cnn.tgz"

if [ $? -ne 0 ]; then
  echo "Error: Failed to download CNN stories"
  exit 1
fi

# Download Daily Mail file
echo "Downloading Daily Mail stories..."
gdown "$DAILYMAIL_URL" -O "$DOWNLOAD_DIR/dailymail.tgz"

if [ $? -ne 0 ]; then
  echo "Error: Failed to download Daily Mail stories"
  exit 1
fi

# Extract files
echo "Extracting CNN stories..."
tar -xzf "$DOWNLOAD_DIR/cnn.tgz" -C "$DOWNLOAD_DIR"

echo "Extracting Daily Mail stories..."
tar -xzf "$DOWNLOAD_DIR/dailymail.tgz" -C "$DOWNLOAD_DIR"

echo "Download and extraction completed."
echo "Files are saved in $DOWNLOAD_DIR"