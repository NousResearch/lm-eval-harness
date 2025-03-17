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

# Test with our specific example
input_text = r"here's my answer: \boxed{\frac{3}{4}}."

# Raw version to see actual string
print(f"Input text: {repr(input_text)}")
print(f"Characters: {[c for c in input_text]}")

# Try a direct approach in the test script
def extract_boxed(text):
    left_bound = r"\boxed{"
    start = text.find(left_bound)
    if start == -1:
        return "[not found]"
    
    start_content = start + len(left_bound)
    
    # Find balanced closing brace
    depth = 1
    i = start_content
    
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                break
        i += 1
    
    if i >= len(text):
        return "[no matching brace]"
    
    return text[start:i+1]

result = extract_boxed(input_text)
print(f"Direct extraction result: {result}")

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

complex_direct = extract_boxed(complex_input)
print(f"Direct extraction complex result: {complex_direct}")

complex_result = balanced_filter.apply([[complex_input]], [{}])
print(f"Balanced filter with complex input: {complex_result}")