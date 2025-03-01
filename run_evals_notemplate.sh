#!/bin/bash

# Initialize variables
PRETRAINED_MODEL=""
DTYPE="auto"
GPU_COUNT=1
DATE_STR=$(date '+%Y%m%d_%H%M%S')
NAME="default_${DATE_STR}"

# Function to display help message
show_help() {
  echo "Usage: $0 <PRETRAINED_MODEL> [OPTIONS]"
  echo ""
  echo "Mandatory argument:"
  echo "  PRETRAINED_MODEL      The name of the pretrained model."
  echo ""
  echo "Optional arguments:"
  echo "  -h, --help           Show this help message and exit."
  echo "  --dtype DTYPE        Data type for the model (default is auto)."
  echo "  --gpu_count COUNT    Number of GPUs available (default is 1)."
  echo "  --name NAME          Custom name for this evaluation run (default is 'default_<timestamp>')."
}

# Function to run the lm_eval command with given tasks
run_command() {
  TASKS=$1
  BENCHMARK_NAME=$2

  # Create clean directory name by replacing problematic characters
  CLEAN_NAME=$(echo "${NAME}" | tr '/' '_')
  
  # Create a single results directory for all benchmarks
  OUTPUT_PATH="./benchmark_logs/${NAME}/"
  TEXT_LOG_PATH="./benchmark_logs/${NAME}/benchmarks.txt"
  
  # Create directory first
  mkdir -p "./benchmark_logs/${CLEAN_NAME}"
  
  echo "Running tasks: $TASKS"
  echo "Output path: $OUTPUT_PATH"
  
  lm_eval --model vllm \
    --model_args pretrained=${PRETRAINED_MODEL},dtype=${DTYPE},gpu_memory_utilization=0.96,tensor_parallel_size=${GPU_COUNT} \
    --tasks $TASKS \
    --batch_size auto \
    --output_path "$OUTPUT_PATH" \
    --human_readable_name "${BENCHMARK_NAME}"
}

# Check if at least one argument is given
if [ $# -eq 0 ]; then
  echo "Please specify the pretrained model as the first argument."
  show_help
  exit 1
fi

# First argument is mandatory for pretrained model
PRETRAINED_MODEL=$1
shift

# Parse remaining command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -h|--help)
      show_help
      exit 0
      ;;
    --dtype)
      DTYPE="$2"
      shift # Past argument
      shift # Past value
      ;;
    --gpu_count)
      GPU_COUNT="$2"
      shift # Past argument
      shift # Past value
      ;;
    --name)
      NAME="$2"
      shift # Past argument
      shift # Past value
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Run the tasks
#run_command "truthfulqa_mc" "TruthfulQA"
run_command "openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq" "GPT4All"
run_command "agieval_aqua_rat,agieval_logiqa_en,agieval_lsat_ar,agieval_lsat_lr,agieval_lsat_rc,agieval_sat_en,agieval_sat_en_without_passage,agieval_sat_math" "AGIEval"
run_command "leaderboard_bbh_boolean_expressions,leaderboard_bbh_causal_judgement,leaderboard_bbh_date_understanding,leaderboard_bbh_disambiguation_qa,leaderboard_bbh_formal_fallacies,leaderboard_bbh_geometric_shapes,leaderboard_bbh_hyperbaton,leaderboard_bbh_logical_deduction_five_objects,leaderboard_bbh_logical_deduction_seven_objects,leaderboard_bbh_logical_deduction_three_objects,leaderboard_bbh_movie_recommendation,leaderboard_bbh_navigate,leaderboard_bbh_object_counting,leaderboard_bbh_penguins_in_a_table,leaderboard_bbh_reasoning_about_colored_objects,leaderboard_bbh_ruin_names,leaderboard_bbh_salient_translation_error_detection,leaderboard_bbh_snarks,leaderboard_bbh_sports_understanding,leaderboard_bbh_temporal_sequences,leaderboard_bbh_tracking_shuffled_objects_five_objects,leaderboard_bbh_tracking_shuffled_objects_seven_objects,leaderboard_bbh_tracking_shuffled_objects_three_objects,leaderboard_bbh_web_of_lies" "Bigbenchhard"
run_command "leaderboard_gpqa_extended,leaderboard_gpqa_diamond,leaderboard_gpqa_main" "GPQA"
run_command "leaderboard_math_algebra_hard,leaderboard_math_counting_and_prob_hard,leaderboard_math_geometry_hard,leaderboard_math_intermediate_algebra_hard,leaderboard_math_num_theory_hard,leaderboard_math_prealgebra_hard,leaderboard_math_precalculus_hard" "MATH_hard"
run_command "leaderboard_mmlu_pro" "MMLU_Pro"
