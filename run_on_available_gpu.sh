#!/bin/bash
# filepath: /home/foresti/mutinfo/run_on_available_gpu.sh

function usage {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -s <script>      Single script to run"
    echo "  -d <directory>   Directory containing scripts to run in queue"
    echo "  -t <threshold>   GPU memory threshold in MB (default: 2000)"
    echo "Example: $0 -s run_nishiyama.sh -t 2000"
    echo "Example: $0 -d /path/to/scripts/dir -t 3000"
}

# Process command-line arguments
THRESHOLD=2000
SCRIPT_PATH=""
SCRIPTS_DIR=""

while getopts "s:d:t:h" opt; do
    case $opt in
        s) SCRIPT_PATH="$OPTARG" ;;
        d) SCRIPTS_DIR="$OPTARG" ;;
        t) THRESHOLD="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done

# Check that we have at least one input method
if [[ -z "$SCRIPT_PATH" && -z "$SCRIPTS_DIR" ]]; then
    echo "Error: You must provide either a script (-s) or a directory (-d)"
    usage
    exit 1
fi

# Function to check GPU availability
check_gpu_availability() {
    # Get list of GPU indices with memory info
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | while read -r line; do
        GPU_ID=$(echo "$line" | awk '{print $1}')
        FREE_MEM=$(echo "$line" | awk '{print $2}')
        
        if [ "$FREE_MEM" -gt "$THRESHOLD" ]; then
            echo "$GPU_ID"
            return 0
        fi
    done
    
    return 1
}

# Function to run a single script when GPU is available
run_script_on_gpu() {
    local script_to_run="$1"
    
    # Make sure the script exists and is executable
    if [[ ! -f "$script_to_run" ]]; then
        echo "Error: Script not found - $script_to_run"
        return 1
    fi
    
    # Extract script name without extension for log file
    SCRIPT_NAME=$(basename "$script_to_run" .sh)
    DATE=$(date +"%Y-%m-%d")
    LOG_FILE="${SCRIPT_NAME}_${DATE}.log"
    
    echo "Processing script: $script_to_run"
    echo "Results will be logged to: $LOG_FILE"
    
    # Loop until available GPU is found
    while true; do
        AVAILABLE_GPU=$(check_gpu_availability)
        
        if [ -n "$AVAILABLE_GPU" ]; then
            echo "Found available GPU: $AVAILABLE_GPU"
            echo "Starting $script_to_run on GPU $AVAILABLE_GPU at $(date)" | tee -a "$LOG_FILE"
            
            # Set CUDA_VISIBLE_DEVICES and run the script
            export CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU
            
            # Run the script and log output
            bash "$script_to_run" 2>&1 | tee -a "$LOG_FILE"
            
            echo "Script finished at $(date)" | tee -a "$LOG_FILE"
            return 0
        else
            echo "No GPUs available with at least ${THRESHOLD}MB free memory. Waiting 30 seconds..."
            sleep 30
        fi
    done
}

# Main execution
if [[ -n "$SCRIPT_PATH" ]]; then
    # Run a single script
    run_script_on_gpu "$SCRIPT_PATH"
elif [[ -n "$SCRIPTS_DIR" ]]; then
    # Run all scripts in the directory
    if [[ ! -d "$SCRIPTS_DIR" ]]; then
        echo "Error: Directory not found - $SCRIPTS_DIR"
        exit 1
    fi
    
    echo "Processing all scripts in directory: $SCRIPTS_DIR"
    
    # Find all .sh files in the directory and process them
    find "$SCRIPTS_DIR" -name "*.sh" -type f | sort | while read -r script; do
        echo "Queuing script: $script"
        run_script_on_gpu "$script"
    done
    
    echo "All scripts in queue completed."
fi