# AGIEval Generative Benchmark

This directory contains the generative version of the AGIEval benchmark. Unlike the standard multiple-choice version, these configurations allow for free-form text generation followed by answer extraction.

## Task Structure

The AGIEval generative benchmark consists of the following components:

- `_default_template_yaml`: Base configuration with generative settings
- `_agieval.yaml`: Task grouping structure
- `agieval_generative.yaml`: Main entry point for running all tasks
- Individual task files for each AGIEval component test

## Task List

The benchmark includes the following tasks:

1. `agieval_aqua-rat_generative`: Algebra questions with rationales
2. `agieval_logiqa-en_generative`: Logic reasoning questions in English
3. `agieval_lsat-ar_generative`: LSAT analytical reasoning
4. `agieval_lsat-lr_generative`: LSAT logical reasoning
5. `agieval_lsat-rc_generative`: LSAT reading comprehension
6. `agieval_sat-en-without-passage_generative`: SAT English questions without passages
7. `agieval_sat-en_generative`: SAT English questions
8. `agieval_sat-math_generative`: SAT mathematics questions

## Running the Tasks

To run the generative AGIEval benchmark:

```bash
./nous/agieval_generative.sh [MODEL_NAME]
```

This will evaluate the specified model on all AGIEval generative tasks.

## Configuration

The generative tasks have the following properties:

- `output_type: generate_until`: Free-form generation until stopping criteria
- `max_gen_toks: 16000`: Maximum generation length
- Stopping at newlines or end-of-sequence tokens
- Consistent extraction of the first line of response
- Evaluation using exact_match metrics with punctuation and case insensitivity