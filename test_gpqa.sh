#!/bin/bash

# Simplified test script
TASKS="gpqa_test"

lm_eval --model vllm \
    --model_args pretrained="NousResearch/DeepHermes-3-Mistral-24B-Preview",dtype=auto,gpu_memory_utilization=0.8,tensor_parallel_size=8 \
    --tasks $TASKS \
    --batch_size auto \
    --mcq_to_generative \
    --apply_chat_template \
    --write_out \
    --output_path out/debug/test-gpqa \
    --limit 2