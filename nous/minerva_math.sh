#!/bin/bash

# Run the Minerva Math benchmark using generative format with math_verify scoring

python -m lm_eval \
  --model anthropic-claude \
  --tasks minerva_math_generative \
  --model_args model=claude-3-opus-20240229 \
  --output_path ./nous/out/minerva_math_generative_claude3.json \
  --num_fewshot 4 \
  --batch_size 4