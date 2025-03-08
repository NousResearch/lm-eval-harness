#!/bin/bash

MODEL=${1:-"mistralai/Mistral-7B-Instruct-v0.2"}

lm_eval --model vllm \
    --model_args pretrained="$MODEL",dtype=auto,gpu_memory_utilization=0.8,tensor_parallel_size=8 \
    --tasks agieval_generative \
    --batch_size auto \
    --apply_chat_template \
    --write_out \
    --output_path out/debug/agieval-generative \
    --system_instruction "You are a helpful AI assistant. Think through problems carefully and respond accurately." \
    --human_readable_name agieval-generative-test \
    --log_samples