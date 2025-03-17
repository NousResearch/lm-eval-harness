from lm_eval.filters.extraction import ExtractAnswerFilter
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter

@register_filter("balanced_extractor")
class BalancedExtractorFilter(Filter):
    """A filter that extracts text between balanced delimiters like braces.
    
    This filter handles nested balanced delimiters correctly, for example:
    \boxed{\frac{3}{4}} where there are nested braces.
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
    
    def find_matching_delimiter(self, text, start_pos, left_delim, right_delim):
        """Find the position of the matching right delimiter."""
        depth = 1
        pos = start_pos
        
        while pos < len(text):
            # Check for the left delimiter (increase depth)
            if pos + len(left_delim) <= len(text) and text[pos:pos+len(left_delim)] == left_delim:
                depth += 1
                pos += len(left_delim)
            # Check for the right delimiter (decrease depth)
            elif pos + len(right_delim) <= len(text) and text[pos:pos+len(right_delim)] == right_delim:
                depth -= 1
                if depth == 0:
                    return pos  # Found the matching delimiter
                pos += len(right_delim)
            else:
                pos += 1
        
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
                right_pos = self.find_matching_delimiter(
                    resp, start_content, self.left_bound, self.right_bound
                )
                
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

# Test with our specific example
input_text = r"here's my answer: \boxed{\frac{3}{4}}."
print(f"Input text: {repr(input_text)}")

# Create an instance of the regular filter
regular_filter = ExtractAnswerFilter(bounds=[
    {"left_bound": r"\boxed{", "right_bound": "}", "include_bounds": True}
])
regular_result = regular_filter.apply([[input_text]], [{}])
print(f"Regular filter result: {regular_result}")

# Create an instance of our balanced filter
balanced_filter = BalancedExtractorFilter(
    left_bound=r"\boxed{",
    right_bound="}",
    include_bounds=True
)
balanced_result = balanced_filter.apply([[input_text]], [{}])
print(f"Balanced filter result: {balanced_result}")

# Try with more complex nested example
complex_input = r"The final answer is \boxed{\sqrt{2} + \sum_{i=1}^{3} i^2}."
print(f"\nComplex input: {repr(complex_input)}")
complex_result = balanced_filter.apply([[complex_input]], [{}])
print(f"Balanced filter with complex input: {complex_result}")