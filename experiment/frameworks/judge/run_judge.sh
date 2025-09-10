#!/bin/bash

# Function to print colored messages
print_status() {
    echo -e "\033[1;36m$1\033[0m"
}

print_error() {
    echo -e "\033[1;31m$1\033[0m"
}

print_success() {
    echo -e "\033[1;32m$1\033[0m"
}

# Function to print usage
print_usage() {
    echo "Usage: $0 -i <input_file> [-o <output_dir>] [-m <model>] [-k <api_key>] [-r <max_retries>] [-c <checkpoint_interval>]"
    echo
    echo "Options:"
    echo "  -i <input_file>    Input JSON file (output from run_oracle_eval.py) (required)"
    echo "  -o <output_dir>    Output directory (default: output)"
    echo "  -m <model>        Model to use for judgment (default: gpt-4o-mini)"
    echo "  -k <api_key>      OpenAI API key (default: from OPENAI_API_KEY env var)"
    echo "  -r <max_retries>  Maximum number of retries for failed judgments (default: 3)"
    echo "  -c <interval>     Interval for saving checkpoints (default: 10)"
    echo "  -h                Show this help message"
    exit 1
}

# Function to validate input file
validate_input_file() {
    if [ ! -f "$1" ]; then
        print_error "Error: Input file '$1' does not exist"
        exit 1
    fi
    
    if ! python3 -c "import json; json.load(open('$1'))" 2>/dev/null; then
        print_error "Error: Input file '$1' is not a valid JSON file"
        exit 1
    fi
    
    # Validate that the input file has the correct structure
    if ! python3 -c "
import json, sys
try:
    with open('$1') as f:
        data = json.load(f)
except Exception as e:
    print(f'Error validating input file: {str(e)}')
    sys.exit(1)
" 2>/dev/null; then
        print_error "Error: Input file '$1' does not have the correct structure"
        exit 1
    fi
}

# Configuration
INPUT_FILE="single_bench_oracle.json"
OUTPUT_DIR="output"
MODEL="gpt-4o-mini"  # Default model
MAX_RETRIES=3
CHECKPOINT_INTERVAL=20
OPENAI_API_KEY="${OPENAI_API_KEY:fake-key}"  # Default API key

# Parse command line arguments
while getopts "i:o:m:k:r:c:h" opt; do
    case $opt in
        i) INPUT_FILE="$OPTARG";;
        o) OUTPUT_DIR="$OPTARG";;
        m) MODEL="$OPTARG";;
        k) OPENAI_API_KEY="$OPTARG";;
        r) MAX_RETRIES="$OPTARG";;
        c) CHECKPOINT_INTERVAL="$OPTARG";;
        h) print_usage;;
        ?) print_usage;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_FILE" ]; then
    print_error "Error: Input file is required"
    print_usage
fi

# Validate input file
validate_input_file "$INPUT_FILE"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get input filename without extension
INPUT_BASENAME=$(basename "$INPUT_FILE")
INPUT_NAME="${INPUT_BASENAME%.*}"

# Set output file path
JUDGE_OUTPUT_FILE="$OUTPUT_DIR/${INPUT_NAME}_judge.json"

# Export environment variables
export OPENAI_API_KEY
export DOCRAG_MODEL="$MODEL"

# Print configuration
print_status "Configuration:"
print_status "Input file: $INPUT_FILE"
print_status "Output directory: $OUTPUT_DIR"
print_status "Model: $MODEL"
print_status "Max retries: $MAX_RETRIES"
print_status "Checkpoint interval: $CHECKPOINT_INTERVAL"
print_status "Output file: $JUDGE_OUTPUT_FILE"
print_status "Using OpenAI API key: ${OPENAI_API_KEY:0:8}..."  # Only show first 8 characters for security

# Run judgment
print_status "\nStarting judgment..."
python3 judge.py \
    --input_file "$INPUT_FILE" \
    --output_file "$JUDGE_OUTPUT_FILE" \
    --model "$MODEL" \
    --max_retries "$MAX_RETRIES" \
    --checkpoint_interval "$CHECKPOINT_INTERVAL"

# Check if judgment was successful
if [ $? -eq 0 ]; then
    print_success "\nJudgment completed successfully!"
    
    # Print judgment statistics
    print_status "\nJudgment Statistics:"
    python3 -c "
import json
with open('$JUDGE_OUTPUT_FILE') as f:
    data = json.load(f)
    stats = data.get('metadata', {}).get('statistics', {})
    print('\033[1;36mModel:\033[0m', data.get('metadata', {}).get('model', 'N/A'))
    print('\033[1;36mTotal Samples:\033[0m', stats.get('total_samples', 'N/A'))
    print('\033[1;36mJudged Samples:\033[0m', stats.get('judged_samples', 'N/A'))
    print('\033[1;36mConsistent Answers:\033[0m', stats.get('yes_count', 'N/A'))
    print('\033[1;36mInconsistent Answers:\033[0m', stats.get('no_count', 'N/A'))
    print('\033[1;36mConsistency Rate:\033[0m', f\"{stats.get('consistency_rate', 0):.2%}\")
    print('\033[1;36mErrors:\033[0m', stats.get('error_count', 'N/A'))
    print('\033[1;36mSkipped:\033[0m', stats.get('skipped_count', 'N/A'))
"
else
    print_error "\nJudgment failed!"
    exit 1
fi
