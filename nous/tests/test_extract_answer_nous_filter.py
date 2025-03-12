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