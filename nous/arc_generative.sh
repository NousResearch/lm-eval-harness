#!/bin/bash

# Run ARC Challenge and Easy generative evaluations

# Choose which tasks to run
TASKS="arc_challenge_generative,arc_easy_generative"

SYSTEM_PROMPT="You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem. Always provide your multiple choice answer letter in the format: The answer is \\boxed{<LETTER>}."

lm_eval --model vllm \
    --model_args pretrained="NousResearch/DeepHermes-3-Llama-3-1B-Preview",dtype=auto,gpu_memory_utilization=0.8,tensor_parallel_size=8 \
    --tasks $TASKS \
    --batch_size auto \
    --mcq_to_generative \
    --apply_chat_template \
    --write_out \
    --output_path out/debug/arc \
    --system_instruction "$SYSTEM_PROMPT" \
    --human_readable_name deephermes-test \
    --log_samples \
    --limit 3