# Creating a Custom Filter in Nous

This guide explains how to create and register a custom filter for the Nous evaluation framework.

## Background

Filters in Nous process the raw output of language models to extract or format answers in a specific way. For example, the `regex` filter extracts content matching a regex pattern from the model's response.

## Step 1: Understand the Filter Architecture

All filters inherit from the abstract `Filter` base class in `lm_eval/api/filter.py` and implement the `apply` method:

```python
from lm_eval.api.filter import Filter

class MyCustomFilter(Filter):
    def __init__(self, param1=default1, param2=default2, ...):
        self.param1 = param1
        self.param2 = param2
        # ... other initialization

    def apply(self, resps, docs):
        # Process responses
        # resps: List of model responses
        # docs: Document metadata 
        filtered_resps = [...]  # Your filtering logic here
        return filtered_resps
```

## Step 2: Register Your Filter

Import the registration decorator from `lm_eval.api.registry` and apply it to your filter class:

```python
from lm_eval.api.registry import register_filter

@register_filter("my-custom-filter")
class MyCustomFilter(Filter):
    # implementation as above
```

## Step 3: Add Implementation File

Create a new Python file for your filter in the appropriate location. Based on the filter's function, choose one of:

- `lm_eval/filters/extraction.py`: Filters that extract content from responses
- `lm_eval/filters/selection.py`: Filters that select certain responses
- `lm_eval/filters/transformation.py`: Filters that transform responses
- `lm_eval/filters/custom.py`: Other specialized filters

If your filter doesn't fit an existing category, create a new file and ensure it's imported in `lm_eval/filters/__init__.py`.

## Step 4: Example Implementation

Here's an example of a simple custom filter that removes punctuation from responses:

```python
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter
import re

@register_filter("remove-punctuation")
class RemovePunctuationFilter(Filter):
    def __init__(self, keep_spaces=True):
        self.keep_spaces = keep_spaces
        self.punctuation_pattern = r'[^\w\s]'

    def apply(self, resps, docs):
        filtered_resps = []
        for resp_list in resps:
            filtered_resp_list = []
            for resp in resp_list:
                # Remove punctuation
                filtered_resp = re.sub(self.punctuation_pattern, '', resp)
                # Optionally collapse whitespace
                if not self.keep_spaces:
                    filtered_resp = re.sub(r'\s+', ' ', filtered_resp).strip()
                filtered_resp_list.append(filtered_resp)
            filtered_resps.append(filtered_resp_list)
        return filtered_resps
```

## Step 5: Using Your Filter in Tasks

Once registered, your filter can be used in task definitions:

```yaml
filter_list:
  - name: "process-answers"
    filter:
      - function: "my-custom-filter"
        param1: value1
        param2: value2
      # Chain with other filters as needed
      - function: "take_first"
```

## Step 6: Testing Your Filter

Create unit tests for your filter in the `tests/` directory to ensure it works as expected.

1. Create a test file (e.g., `tests/test_my_custom_filter.py`)
2. Write test cases with sample inputs and expected outputs
3. Run the tests to verify correctness

## Best Practices

1. Make filters configurable with parameters
2. Document parameters clearly
3. Handle edge cases gracefully
4. Return data in the same structure as received
5. Chain filters for complex processing
6. Consider performance for large-scale evaluations