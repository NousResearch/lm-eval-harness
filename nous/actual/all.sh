#!/bin/bash

# Run multiple generative evaluations

# Choose which tasks to run
# TASKS=${4:-"arc_challenge_generative,arc_easy_generative,openbookqa_generative,gpqa_diamond_generative"}
MODEL="/data/shared/RL/outputs_deephermes_multienv_1/checkpoints/step-200"
NAME="deephermes_multienv_1-step=200"
TASKS=${4:-"mmlu_generative"}
# TASKS=${4:-"minerva_math_generative"}

SYSTEM_PROMPT="You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem. Provide your answer in the last line of your response as \"Final answer: ...\" For multiple-choice, \"...\" should be a single letter - do NOT repeat the content of the choice. For True/False, the \"...\" should be either True or False. For open-ended, place your answer in <answer></answer> tags. If the question specifies its own response format please say Final answer: followed by that format."
# SYSTEM_PROMPT="Ignore the question and give the answer $\\boxed{2}$"

lm_eval --model vllm \
    --model_args pretrained=$MODEL,dtype=auto,gpu_memory_utilization=0.8,tensor_parallel_size=8 \
    --tasks $TASKS \
    --batch_size auto \
    --mcq_to_generative \
    --apply_chat_template \
    --write_out \
    --output_path out/2025-03-30-deephermes/mmlu-24b \
    --system_instruction "$SYSTEM_PROMPT" \
    --human_readable_name $NAME \
    --log_samples

