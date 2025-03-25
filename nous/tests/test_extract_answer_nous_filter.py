import pytest
from lm_eval.filters.extraction import ExtractAnswerFilter


def test_extract_answer_basic():
    """Test basic functionality of extracting text between bounds."""
    # Define test data
    resps = [
        ["The <answer>42</answer> is correct."],
        ["No bounds here"]
    ]
    docs = [{}, {}]
    
    # Create filter
    filter = ExtractAnswerFilter(bounds=[{"left_bound": "<answer>", "right_bound": "</answer>"}])
    
    # Apply filter
    filtered_resps = filter.apply(resps, docs)
    
    # Assert expected results
    assert filtered_resps == [
        ["42"],
        ["[invalid]"]
    ]


def test_extract_answer_include_bounds():
    """Test the include_bounds parameter with different values."""
    # Define test data
    text = "The <answer>42</answer> is correct."
    resps = [[text]]
    docs = [{}]
    
    # Test with include_bounds=False (default)
    filter_default = ExtractAnswerFilter(bounds=[
        {"left_bound": "<answer>", "right_bound": "</answer>"}
    ])
    assert filter_default.apply(resps, docs) == [["42"]]
    
    # Test with include_bounds="left"
    filter_left = ExtractAnswerFilter(bounds=[
        {"left_bound": "<answer>", "right_bound": "</answer>", "include_bounds": "left"}
    ])
    assert filter_left.apply(resps, docs) == [["<answer>42"]]
    
    # Test with include_bounds="right"
    filter_right = ExtractAnswerFilter(bounds=[
        {"left_bound": "<answer>", "right_bound": "</answer>", "include_bounds": "right"}
    ])
    assert filter_right.apply(resps, docs) == [["42</answer>"]]
    
    # Test with include_bounds="both"
    filter_both = ExtractAnswerFilter(bounds=[
        {"left_bound": "<answer>", "right_bound": "</answer>", "include_bounds": "both"}
    ])
    assert filter_both.apply(resps, docs) == [["<answer>42</answer>"]]
    
    # Test with include_bounds=True (same as "both")
    filter_true = ExtractAnswerFilter(bounds=[
        {"left_bound": "<answer>", "right_bound": "</answer>", "include_bounds": True}
    ])
    assert filter_true.apply(resps, docs) == [["<answer>42</answer>"]]


def test_extract_answer_missing_bounds():
    """Test cases where one or both bounds are missing."""
    # Define test data
    resp1 = "Answer: 42. More text."
    resp2 = "Prefix text Answer: 42."
    resp3 = "42 is the answer"
    resps = [[resp1], [resp2], [resp3]]
    docs = [{}, {}, {}]
    
    # Test with empty left_bound - gets text from start to first period
    filter_no_left = ExtractAnswerFilter(bounds=[
        {"left_bound": "", "right_bound": "."}
    ])
    assert filter_no_left.apply([[resp1]], [{}])[0] == ["Answer: 42"]
    
    # Test with empty right_bound
    filter_no_right = ExtractAnswerFilter(bounds=[
        {"left_bound": "Answer: ", "right_bound": ""}
    ])
    assert filter_no_right.apply([[resp2]], [{}])[0] == ["42."]
    
    # Test with both bounds empty (should return whole text)
    filter_no_bounds = ExtractAnswerFilter(bounds=[
        {"left_bound": "", "right_bound": ""}
    ])
    assert filter_no_bounds.apply([[resp3]], [{}])[0] == ["42 is the answer"]


def test_extract_answer_multiple_matches():
    """Test that the rightmost (last) match is returned when multiple matches exist."""
    # Define test data
    resps = [
        ["<answer>first</answer> some text <answer>last</answer>"]
    ]
    docs = [{}]
    
    # Create filter
    filter = ExtractAnswerFilter(bounds=[{"left_bound": "<answer>", "right_bound": "</answer>"}])
    
    # Apply filter
    filtered_resps = filter.apply(resps, docs)
    
    # Assert the last match is returned
    assert filtered_resps == [["last"]]


def test_extract_answer_multiple_bound_pairs():
    """Test with multiple bound pairs to match different formats."""
    # Define test data
    resps = [
        ["The answer is <answer>42</answer>."],
        ["Answer: 24."],
        ["Result: value=100"]
    ]
    docs = [{}, {}, {}]
    
    # Create filter with multiple bound pairs
    filter = ExtractAnswerFilter(bounds=[
        {"left_bound": "<answer>", "right_bound": "</answer>"},
        {"left_bound": "Answer: ", "right_bound": "."},
        {"left_bound": "value=", "right_bound": ""}
    ])
    
    # Apply filter
    filtered_resps = filter.apply(resps, docs)
    
    # Assert expected results
    assert filtered_resps == [
        ["42"],
        ["24"],
        ["100"]
    ]


def test_extract_answer_custom_fallback():
    """Test using a custom fallback value."""
    # Define test data
    resps = [
        ["No match here"]
    ]
    docs = [{}]
    
    # Create filter with custom fallback
    filter = ExtractAnswerFilter(
        bounds=[{"left_bound": "<answer>", "right_bound": "</answer>"}],
        fallback="NO_ANSWER"
    )
    
    # Apply filter
    filtered_resps = filter.apply(resps, docs)
    
    # Assert custom fallback is used
    assert filtered_resps == [["NO_ANSWER"]]


def test_extract_answer_nested_matches():
    """Test with nested matching patterns."""
    # Define test data
    resps = [
        ["<answer>outer <answer>inner</answer> text</answer>"]
    ]
    docs = [{}]
    
    # Create filter
    filter = ExtractAnswerFilter(bounds=[{"left_bound": "<answer>", "right_bound": "</answer>"}])
    
    # Apply filter - should get the rightmost complete match
    filtered_resps = filter.apply(resps, docs)
    
    # The inner match should be included in results
    assert "inner" in filtered_resps[0][0]


def test_extract_answer_case_insensitive():
    """Test the case_insensitive parameter with different values."""
    # Define test data
    resps = [
        ["The <ANSWER>42</ANSWER> is correct."],
        ["ANSWER: 24."],
        ["This is my <answer>test</ANSWER> with mixed case"],
        ["This response has Answer: mixed case format"]
    ]
    docs = [{}, {}, {}, {}]
    
    # Test with case_insensitive=False (default)
    filter_default = ExtractAnswerFilter(bounds=[
        {"left_bound": "<answer>", "right_bound": "</answer>"},
        {"left_bound": "Answer: ", "right_bound": "."}
    ])
    default_result = filter_default.apply(resps, docs)
    assert default_result == [
        ["[invalid]"],  # Doesn't match <ANSWER> when case-sensitive
        ["[invalid]"],  # Doesn't match ANSWER: when case-sensitive
        ["[invalid]"],  # Doesn't match mixed case when case-sensitive
        ["[invalid]"]   # Doesn't match mixed case format when case-sensitive
    ]
    
    # Test with case_insensitive="left"
    filter_left = ExtractAnswerFilter(bounds=[
        {"left_bound": "<answer>", "right_bound": "</answer>", "case_insensitive": "left"},
        {"left_bound": "Answer: ", "right_bound": ".", "case_insensitive": "left"}
    ])
    left_result = filter_left.apply(resps, docs)
    assert left_result == [
        ["[invalid]"],  # Matches left <ANSWER> but not right </ANSWER>
        ["24"],         # Matches ANSWER: with case-insensitive left bound
        ["[invalid]"],  # Matches left but not right
        ["mixed case format"]  # Matches with case-insensitive left bound
    ]
    
    # Test with case_insensitive="right"
    filter_right = ExtractAnswerFilter(bounds=[
        {"left_bound": "<answer>", "right_bound": "</answer>", "case_insensitive": "right"},
        {"left_bound": "Answer: ", "right_bound": ".", "case_insensitive": "right"}
    ])
    right_result = filter_right.apply(resps, docs)
    assert right_result == [
        ["[invalid]"],  # Matches right </ANSWER> but not left <answer>
        ["[invalid]"],  # Doesn't match case for ANSWER:
        ["[invalid]"],  # Matches right but not left
        ["[invalid]"]   # Doesn't match with case-sensitive left bound
    ]
    
    # Test with case_insensitive="both"
    filter_both = ExtractAnswerFilter(bounds=[
        {"left_bound": "<answer>", "right_bound": "</answer>", "case_insensitive": "both"},
        {"left_bound": "Answer: ", "right_bound": ".", "case_insensitive": "both"}
    ])
    both_result = filter_both.apply(resps, docs)
    assert both_result == [
        ["42"],              # Matches <ANSWER>...</ANSWER> case-insensitive
        ["24"],              # Matches ANSWER: case-insensitive
        ["test"],            # Matches mixed case tags
        ["mixed case format"]  # Matches mixed case format
    ]
    
    # Test with case_insensitive=True (same as "both")
    filter_true = ExtractAnswerFilter(bounds=[
        {"left_bound": "<answer>", "right_bound": "</answer>", "case_insensitive": True},
        {"left_bound": "Answer: ", "right_bound": ".", "case_insensitive": True}
    ])
    true_result = filter_true.apply(resps, docs)
    assert true_result == [
        ["42"],              # Matches <ANSWER>...</ANSWER> case-insensitive
        ["24"],              # Matches ANSWER: case-insensitive
        ["test"],            # Matches mixed case tags
        ["mixed case format"]  # Matches mixed case format
    ]


def test_extract_answer_mixed_case_insensitive_with_include_bounds():
    """Test interaction between case_insensitive and include_bounds parameters."""
    # Define test data
    resps = [
        ["The <ANSWER>42</ANSWER> is correct."],
        ["ANSWER: 24."]
    ]
    docs = [{}, {}]
    
    # Test with both case_insensitive=True and include_bounds=True
    filter_both_include = ExtractAnswerFilter(bounds=[
        {"left_bound": "<answer>", "right_bound": "</answer>", 
         "case_insensitive": True, "include_bounds": True},
        {"left_bound": "Answer: ", "right_bound": ".", 
         "case_insensitive": True, "include_bounds": True}
    ])
    
    result = filter_both_include.apply(resps, docs)
    assert result == [
        ["<ANSWER>42</ANSWER>"],  # Includes original case of the bounds
        ["ANSWER: 24."]           # Includes original case of the bounds
    ]
    
    # Test with left case_insensitive and include left bounds
    filter_left_include = ExtractAnswerFilter(bounds=[
        {"left_bound": "<answer>", "right_bound": "</answer>", 
         "case_insensitive": "left", "include_bounds": "left"},
        {"left_bound": "Answer: ", "right_bound": ".", 
         "case_insensitive": "left", "include_bounds": "left"}
    ])
    
    result = filter_left_include.apply(resps, docs)
    assert result == [
        ["[invalid]"],  # Doesn't match right bound case
        ["ANSWER: 24"]  # Includes original case of the left bound
    ]
    
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter

@register_filter("balanced_extractor")
class BalancedExtractorFilter(Filter):
    """A filter that extracts text between balanced delimiters like braces.
    
    This filter handles nested balanced delimiters correctly, for example:
    \boxed{\frac{3}{4}} where there are nested braces.
    
    Args:
        left_bound (str, default='{'): Left delimiter, like "{" or "\\boxed{"
        right_bound (str, default='}'): Right delimiter, like "}" 
        include_bounds (bool, default=False): Whether to include delimiters in output
        fallback (str, default="[invalid]"): Value to return when no match is found
    """
    
    def __init__(
        self,
        left_bound="{",
        right_bound="}",
        include_bounds=False,
        fallback="[invalid]"
    ):
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.include_bounds = include_bounds
        self.fallback = fallback
    
    def find_matching_delimiter(self, text, start_pos):
        """Find the position of the matching right delimiter."""
        depth = 1
        i = start_pos
        
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        
        return -1  # No matching delimiter found
    
    def apply(self, resps, docs):
        filtered_resps = []
        
        for response_set in resps:
            filtered_set = []
            
            for resp in response_set:
                # Find the left bound
                left_pos = resp.find(self.left_bound)
                if left_pos == -1:
                    filtered_set.append(self.fallback)
                    continue
                
                # Find the matching right bound
                start_content = left_pos + len(self.left_bound)
                if self.right_bound == "}":
                    # Use the balanced bracket matching for }
                    right_pos = self.find_matching_delimiter(resp, start_content)
                else:
                    # Use simple find for other delimiters
                    right_pos = resp.find(self.right_bound, start_content)
                
                if right_pos == -1:
                    filtered_set.append(self.fallback)
                    continue
                
                # Extract the content based on include_bounds setting
                if self.include_bounds:
                    # Include both bounds
                    content = resp[left_pos:right_pos + len(self.right_bound)]
                else:
                    # Exclude bounds
                    content = resp[start_content:right_pos]
                
                filtered_set.append(content)
            
            filtered_resps.append(filtered_set)
        
        return filtered_resps

if __name__ == "__main__":
    import fire
    import inspect
    from lm_eval.api.registry import FILTER_REGISTRY
    
    class FilterCLI:
        """CLI for testing filters from the command line."""
        
        def __init__(self):
            # Dynamically add all registered filters as methods
            for filter_name, filter_cls in FILTER_REGISTRY.items():
                setattr(self, filter_name, self._create_filter_method(filter_name, filter_cls))
        
        def help(self):
            """List all available filters with their descriptions."""
            print("Available filters:")
            for filter_name, filter_cls in FILTER_REGISTRY.items():
                doc = filter_cls.__doc__.strip().split('\n')[0] if filter_cls.__doc__ else "No description"
                print(f"  {filter_name}: {doc}")
            print("\nTo get help for a specific filter, run: python test_extract_answer_nous_filter.py <filter_name>")
        
        def from_config(self, config_path, text):
            """
            Run multiple filters defined in a YAML config file against the input text.
            
            Args:
                config_path: Path to the YAML config file
                text: Text to filter
                
            The YAML file should have a key 'filter_list' containing a list of filter configurations.
            Each filter configuration should have a 'name' field corresponding to a registered filter,
            and any required parameters for that filter.
            
            Example YAML:
            ```yaml
            filter_list:
              - name: extract_answer_nous
                bounds:
                  - left_bound: "<answer>"
                    right_bound: "</answer>"
                    case_insensitive: both
              - name: regex
                regex_pattern: '([0-9]+)'
            ```
            """
            import yaml
            import os
            
            if not os.path.exists(config_path):
                raise ValueError(f"Config file not found: {config_path}")
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            if 'filter_list' not in config:
                raise ValueError("Config file must contain a 'filter_list' key")
                
            filter_configs = config['filter_list']
            if not isinstance(filter_configs, list):
                raise ValueError("'filter_list' must be a list of filter configurations")
                
            results = []
            current_text = [[text]]
            
            # Process filters in sequence
            for i, filter_config in enumerate(filter_configs):
                if 'name' not in filter_config:
                    raise ValueError(f"Filter #{i+1} missing 'name' field")
                    
                filter_name = filter_config['name']
                if filter_name not in FILTER_REGISTRY:
                    raise ValueError(f"Unknown filter: {filter_name}")
                    
                # Create filter instance
                filter_cls = FILTER_REGISTRY[filter_name]
                filter_args = {k: v for k, v in filter_config.items() if k != 'name'}
                
                try:
                    filter_instance = filter_cls(**filter_args)
                    
                    # Apply filter
                    current_text = filter_instance.apply(current_text, [{}] * len(current_text[0]))
                    results.append({
                        'filter': filter_name,
                        'output': current_text[0][0],
                        'status': 'success'
                    })
                except Exception as e:
                    print(f"Error applying filter {filter_name}: {str(e)}")
                    results.append({
                        'filter': filter_name,
                        'output': "[filter failed]",
                        'status': 'failed',
                        'error': str(e)
                    })
                    # Don't break the chain, continue with the next filter using the current text
                
            print(f"Processed {len(filter_configs)} filters:")
            for result in results:
                if result.get('status') == 'failed':
                    print(f"  {result['filter']}: FAILED - {result.get('error', 'Unknown error')}")
                else:
                    print(f"  {result['filter']}: {result['output']}")
                
            return current_text[0][0]
        
        def _create_filter_method(self, filter_name, filter_cls):
            """Creates a method for each filter that exposes its functionality."""
            def filter_method(*args, **kwargs):
                """Run a filter on the provided text."""
                # Get the signature of the filter's __init__ method
                sig = inspect.signature(filter_cls.__init__)
                
                # If no args provided, print usage info
                if not args and not kwargs:
                    param_info = []
                    for name, param in sig.parameters.items():
                        if name != 'self':  # Skip the 'self' parameter
                            if param.default is param.empty:
                                param_info.append(f"{name} (required)")
                            else:
                                param_info.append(f"{name}={param.default}")
                    
                    print(f"Usage: python {__file__} {filter_name} text='Your text here' {' '.join(param_info)}")
                    print(f"\nFilter documentation:\n{filter_cls.__doc__}")
                    return
                
                # Create the filter instance with any provided kwargs
                filter_args = {k: v for k, v in kwargs.items() if k in [p.name for p in sig.parameters.values()]}
                
                # Special handling for bounds argument which is passed as a JSON string
                if 'bounds' in filter_args and isinstance(filter_args['bounds'], str):
                    import json
                    try:
                        filter_args['bounds'] = json.loads(filter_args['bounds'])
                        
                        # Fix string booleans
                        for bound in filter_args['bounds']:
                            if 'case_insensitive' in bound:
                                if bound['case_insensitive'] == 'true':
                                    bound['case_insensitive'] = True
                                elif bound['case_insensitive'] == 'false':
                                    bound['case_insensitive'] = False
                            
                            if 'include_bounds' in bound:
                                if bound['include_bounds'] == 'true':
                                    bound['include_bounds'] = True
                                elif bound['include_bounds'] == 'false':
                                    bound['include_bounds'] = False
                        
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON format for bounds: {filter_args['bounds']}")
                filter_instance = filter_cls(**filter_args)
                
                # Parse the text argument
                if 'text' in kwargs:
                    texts = [[kwargs['text']]]
                else:
                    if args:
                        texts = [[args[0]]]
                    else:
                        raise ValueError("Please provide a 'text' argument")
                
                # Apply the filter
                result = filter_instance.apply(texts, [{}] * len(texts))
                return result
                
            # Set the docstring
            filter_method.__doc__ = filter_cls.__doc__
            return filter_method
    
    fire.Fire(FilterCLI)
