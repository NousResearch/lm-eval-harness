#!/bin/bash

# Initialize variables
DTYPE="auto"
GPU_COUNT=1
MODELS=()
MODEL_NAMES=()

# Function to display help message
show_help() {
    echo "Usage: $0 [OPTIONS] MODEL1[:NAME1] [MODEL2[:NAME2] ...]"
    echo ""
    echo "Arguments:"
    echo "  MODEL[:NAME]        Path to model with optional custom name"
    echo "                      Example: 'mistralai/Mistral-7B-v0.1:mistral7b'"
    echo ""
    echo "Optional arguments:"
    echo "  -h, --help         Show this help message and exit"
    echo "  --dtype DTYPE      Data type for all models (default: auto)"
    echo "  --gpu_count COUNT  Number of GPUs to use (default: 1)"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --gpu_count)
            GPU_COUNT="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            # Split model path and name if provided (format: path:name)
            IFS=':' read -r MODEL_PATH MODEL_NAME <<< "$1"
            MODELS+=("$MODEL_PATH")
            # If no name provided, use basename of model path
            if [ -z "$MODEL_NAME" ]; then
                MODEL_NAME="$(basename "$MODEL_PATH")"
            fi
            MODEL_NAMES+=("$MODEL_NAME")
            shift
            ;;
    esac
done

# Check if any models were specified
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Error: No models specified"
    show_help
    exit 1
fi

# Process each model
for i in "${!MODELS[@]}"; do
    echo "====================================="
    echo "Processing model: ${MODELS[i]}"
    echo "Using name: ${MODEL_NAMES[i]}"
    echo "====================================="
    
    # Run the evaluation harness
    ./run_eval_harness.sh "${MODELS[i]}" \
        --dtype "$DTYPE" \
        --gpu_count "$GPU_COUNT" \
        --name "${MODEL_NAMES[i]}"
    
    echo "Completed evaluation for: ${MODELS[i]}"
    echo ""
done