import pytest
from lm_eval.filters.extraction import RegexFallbackToOriginalFilter


def test_regex_fallback_basic():
    """Test basic functionality of regex matching with fallback."""
    # Define test data
    resps = [
        ["The answer is 42."],
        ["No match here"]
    ]
    docs = [{}, {}]
    
    # Create filter
    filter = RegexFallbackToOriginalFilter(regex_pattern=r"answer is (\d+)")
    
    # Apply filter
    filtered_resps = filter.apply(resps, docs)
    
    # Assert expected results
    assert filtered_resps == [
        ["42"],
        ["No match here"]  # Returns original text since no match found
    ]


def test_regex_fallback_group_select():
    """Test selecting different capture groups."""
    # Define test data
    resps = [
        ["Name: John, Age: 30, City: New York"],
    ]
    docs = [{}]
    
    # Test with default group_select=-1 (last group)
    filter_default = RegexFallbackToOriginalFilter(
        regex_pattern=r"Name: (\w+), Age: (\d+), City: (.+)"
    )
    assert filter_default.apply(resps, docs) == [["New York"]]
    
    # Test with group_select=0 (first group)
    filter_first = RegexFallbackToOriginalFilter(
        regex_pattern=r"Name: (\w+), Age: (\d+), City: (.+)",
        group_select=0
    )
    assert filter_first.apply(resps, docs) == [["John"]]
    
    # Test with group_select=1 (second group)
    filter_second = RegexFallbackToOriginalFilter(
        regex_pattern=r"Name: (\w+), Age: (\d+), City: (.+)",
        group_select=1
    )
    assert filter_second.apply(resps, docs) == [["30"]]


def test_regex_fallback_without_groups():
    """Test regex pattern without any capture groups."""
    # Define test data
    resps = [
        ["The answer is 42."],
        ["No match here"]
    ]
    docs = [{}, {}]
    
    # Create filter with regex that has no capture groups
    filter = RegexFallbackToOriginalFilter(regex_pattern=r"answer is \d+")
    
    # Apply filter
    filtered_resps = filter.apply(resps, docs)
    
    # Assert expected results - should return the entire match
    assert filtered_resps == [
        ["answer is 42"],
        ["No match here"]  # Returns original text since no match found
    ]


def test_regex_fallback_multiple_matches():
    """Test behavior with multiple regex matches in the same text."""
    # Define test data
    resps = [
        ["Number 1: 42, Number 2: 100"]
    ]
    docs = [{}]
    
    # Create filter
    filter = RegexFallbackToOriginalFilter(regex_pattern=r"Number \d+: (\d+)")
    
    # Apply filter - should find the first match due to using search() not findall()
    filtered_resps = filter.apply(resps, docs)
    
    # Assert expected results
    assert filtered_resps == [["42"]]  # Gets the first match


def test_regex_fallback_out_of_bounds_group():
    """Test behavior when the selected group index is out of bounds."""
    # Define test data
    resps = [
        ["Name: John, Age: 30"]
    ]
    docs = [{}]
    
    # Create filter with group_select that's out of bounds
    filter = RegexFallbackToOriginalFilter(
        regex_pattern=r"Name: (\w+)",
        group_select=5  # There's only one group
    )
    
    # Apply filter
    filtered_resps = filter.apply(resps, docs)
    
    # Assert it returns the original text when group index is out of bounds
    assert filtered_resps == [["Name: John, Age: 30"]]


def test_regex_fallback_complex_pattern():
    """Test with a more complex regex pattern."""
    # Define test data
    resps = [
        ["The answer to the question is: 42!"],
        ["I think the solution might be 3.14159"]
    ]
    docs = [{}, {}]
    
    # Create filter with a complex pattern
    filter = RegexFallbackToOriginalFilter(
        regex_pattern=r"(?:answer|solution)(?:.+?)(?:is|be)[:\s]+([0-9\.]+)"
    )
    
    # Apply filter
    filtered_resps = filter.apply(resps, docs)
    
    # Assert expected results
    assert filtered_resps == [
        ["42"],
        ["3.14159"]
    ]