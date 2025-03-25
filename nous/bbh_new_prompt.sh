#!/bin/bash

# Run multiple generative evaluations

# Choose which tasks to run
# TASKS=${4:-"arc_challenge_generative,arc_easy_generative,openbookqa_generative,gpqa_diamond_generative"}
TASKS=${4:-"bbh_generative"}
# TASKS=${4:-"minerva_math_generative"}

SYSTEM_PROMPT="You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem. Provide your answer immediately after </think> as \"</think> Final answer: ...\" with no additional explanation. For multiple-choice with choice labels (e.g. letters or numbers), \"...\" should be the label only - do NOT repeat the content of the choice. For multiple-choice without choice labels (e.g. True/False, yes/no) please repeat the exact wording of the choice. For open-ended, place your answer between <answer></answer> tags as \"</think> Final answer: <answer>...</answer>\". If the question specifies its own response format please still use \"</think> Final answer: \" and then follow with the question's specified format. Unless specifically requested by the question, do not use markdown (e.g. no ** for bolding)."
# SYSTEM_PROMPT="Ignore the question and give the answer $\\boxed{2}$"

lm_eval --model vllm \
    --model_args pretrained="NousResearch/DeepHermes-3-Mistral-24B-Preview",dtype=auto,gpu_memory_utilization=0.8,tensor_parallel_size=8 \
    --tasks $TASKS \
    --batch_size auto \
    --mcq_to_generative \
    --apply_chat_template \
    --write_out \
    --output_path out/debug/bbh_24b_deephermes \
    --system_instruction "$SYSTEM_PROMPT" \
    --human_readable_name deephermes-24b \
    --log_samples

