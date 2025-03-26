#!/bin/bash
# filepath: download_data.sh

# Default values
DOWNLOAD_DIR="./data"
PARALLEL=4  # Number of parallel downloads
START_IDX=0
END_IDX=23
BASE_URL="https://storage.googleapis.com/sfr-summarization-repo-research/M"
EXTRACT=false  # Default: don't extract files

# Help message
function show_help {
  echo "Usage: $0 [OPTIONS]"
  echo "Download data archives from specified URL pattern"
  echo ""
  echo "Options:"
  echo "  -d, --dir DIR       Download directory (default: ./data)"
  echo "  -p, --parallel N    Number of parallel downloads (default: 4)"
  echo "  -s, --start N       Starting index (default: 0)"
  echo "  -e, --end N         Ending index (default: 23)"
  echo "  -x, --extract       Extract downloaded tar.gz files"
  echo "  -h, --help          Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -d|--dir)
      DOWNLOAD_DIR="$2"
      shift
      shift
      ;;
    -p|--parallel)
      PARALLEL="$2"
      shift
      shift
      ;;
    -s|--start)
      START_IDX="$2"
      shift
      shift
      ;;
    -e|--end)
      END_IDX="$2"
      shift
      shift
      ;;
    -x|--extract)
      EXTRACT=true
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

# Create download directory if it doesn't exist
if [ ! -d "$DOWNLOAD_DIR" ]; then
  echo "Creating download directory: $DOWNLOAD_DIR"
  mkdir -p "$DOWNLOAD_DIR"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to create download directory"
    exit 1
  fi
fi

echo "Downloading files from $BASE_URL[N].tar.gz to $DOWNLOAD_DIR"
echo "Index range: $START_IDX to $END_IDX"
echo "Using $PARALLEL parallel downloads"
if [ "$EXTRACT" = true ]; then
  echo "Files will be extracted after download"
fi

# Function to extract a tar.gz file
extract_file() {
  local filepath=$1
  local idx=$2
  local extract_dir="${DOWNLOAD_DIR}/M${idx}"
  
  echo "Extracting M${idx}.tar.gz to ${extract_dir}..."
  
  # Create extraction directory if it doesn't exist
  mkdir -p "$extract_dir"
  
  # Extract the file
  tar -xzf "$filepath" -C "$extract_dir"
  
  if [ $? -eq 0 ]; then
    echo "Successfully extracted M${idx}.tar.gz"
    # Mark as extracted by creating a flag file
    touch "${extract_dir}/.extracted"
    return 0
  else
    echo "Failed to extract M${idx}.tar.gz"
    return 1
  fi
}

# Function to download a single file
download_file() {
  local idx=$1
  local url="${BASE_URL}${idx}.tar.gz"
  local output="${DOWNLOAD_DIR}/M${idx}.tar.gz"
  
  echo "Downloading $url to $output..."
  
  # Check if file already exists
  if [ -f "$output" ]; then
    echo "File $output already exists, skipping download..."
    
    # If extraction is enabled, extract the file (even if it was already downloaded)
    if [ "$EXTRACT" = true ]; then
      extract_file "$output" "$idx"
    fi
    
    return 0
  fi
  
  # Use wget with progress bar
  wget -q --show-progress -O "$output" "$url"
  
  if [ $? -eq 0 ]; then
    echo "Successfully downloaded M${idx}.tar.gz"
    
    # Extract if requested
    if [ "$EXTRACT" = true ]; then
      extract_file "$output" "$idx"
    fi
    
    return 0
  else
    echo "Failed to download M${idx}.tar.gz"
    # Remove partially downloaded file
    rm -f "$output"
    return 1
  fi
}

# Track the number of running jobs and success/failure counts
running=0
success=0
failed=0

# Loop through each index and download
for ((i=START_IDX; i<=END_IDX; i++)); do
  # Start download in background
  download_file $i &
  
  # Increment the running count
  ((running++))
  
  # If we've reached the parallel limit or last file, wait for completion
  if [ $running -ge $PARALLEL ] || [ $i -eq $END_IDX ]; then
    wait  # Wait for all background processes to finish
    running=0  # Reset the counter
  fi
done

# Wait for any remaining background jobs
wait

echo "Download process completed."
echo "Files were saved to $DOWNLOAD_DIR"

# Count success/failure by checking files
downloaded_success=0
downloaded_failed=0
extracted_success=0
extracted_failed=0

for ((i=START_IDX; i<=END_IDX; i++)); do
  if [ -f "${DOWNLOAD_DIR}/M${i}.tar.gz" ]; then
    ((downloaded_success++))
    
    # Check extraction status if extraction was enabled
    if [ "$EXTRACT" = true ] && [ -f "${DOWNLOAD_DIR}/M${i}/.extracted" ]; then
      ((extracted_success++))
    elif [ "$EXTRACT" = true ]; then
      ((extracted_failed++))
    fi
  else
    ((downloaded_failed++))
  fi
done

echo "Summary: $downloaded_success files downloaded successfully, $downloaded_failed failed"

if [ $downloaded_failed -gt 0 ]; then
  echo "Failed downloads:"
  for ((i=START_IDX; i<=END_IDX; i++)); do
    if [ ! -f "${DOWNLOAD_DIR}/M${i}.tar.gz" ]; then
      echo "  M${i}.tar.gz"
    fi
  done
fi

# Report extraction status if extraction was enabled
if [ "$EXTRACT" = true ]; then
  echo "Extraction summary: $extracted_success files extracted successfully, $extracted_failed failed"
  
  if [ $extracted_failed -gt 0 ]; then
    echo "Failed extractions:"
    for ((i=START_IDX; i<=END_IDX; i++)); do
      if [ -f "${DOWNLOAD_DIR}/M${i}.tar.gz" ] && [ ! -f "${DOWNLOAD_DIR}/M${i}/.extracted" ]; then
        echo "  M${i}.tar.gz"
      fi
    done
  fi
fi

if [ $downloaded_failed -gt 0 ] || ([ "$EXTRACT" = true ] && [ $extracted_failed -gt 0 ]); then
  exit 1
else
  echo "All operations completed successfully!"
  exit 0
fi