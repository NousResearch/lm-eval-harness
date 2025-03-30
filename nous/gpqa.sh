#!/bin/bash

# Run multiple generative evaluations

# Choose which tasks to run
# TASKS=${4:-"arc_challenge_generative,arc_easy_generative,openbookqa_generative,gpqa_diamond_generative"}
TASKS=${4:-"gpqa_diamond_generative"}
# TASKS=${4:-"minerva_math_generative"}

SYSTEM_PROMPT="You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem. After the closing </think> tag, provide your selected choice letter at the very end of your response in the following exact format:\n\nFinal answer: (letter)\n\nDo not include the text of the choice and do not use markdown."

lm_eval --model vllm \
    --model_args pretrained="NousResearch/DeepHermes-3-Mistral-24B-Preview",dtype=auto,gpu_memory_utilization=0.8,tensor_parallel_size=8 \
    --tasks $TASKS \
    --batch_size auto \
    --mcq_to_generative \
    --apply_chat_template \
    --write_out \
    --output_path out/debug/24b-gpqa \
    --system_instruction "$SYSTEM_PROMPT" \
    --human_readable_name deephermes-test \
    --log_samples
    # --limit 16

