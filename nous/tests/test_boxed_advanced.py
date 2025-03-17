from lm_eval.filters.extraction import ExtractAnswerFilter
import re

# Input text
input_text = r"here's my answer: \boxed{\frac{3}{4}}."
print(f"Input text: {repr(input_text)}")

# Try using a custom approach with regex
pattern = r"\\boxed\{(.*?)\}"
match = re.search(pattern, input_text)
if match:
    print(f"Regex match: {match.group(0)}")
else:
    print("No regex match")

# Test with different right bounds
tests = [
    {"left_bound": r"\boxed{", "right_bound": "}.", "include_bounds": True},
    {"left_bound": r"\boxed{", "right_bound": "}}", "include_bounds": True},
    {"left_bound": r"\boxed{", "right_bound": "}", "include_bounds": True},
    {"left_bound": r"\boxed", "right_bound": ".", "include_bounds": True},
]

for i, test in enumerate(tests):
    filter = ExtractAnswerFilter(bounds=[test])
    result = filter.apply([[input_text]], [{}])
    print(f"Test {i+1} with {test}: {result}")

# Try a non-greedy approach - create a custom filter function
def custom_extract(text):
    pattern = r"\\boxed\{(.*?)\}"
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    return "[not found]"

print(f"Custom extract: {custom_extract(input_text)}")

# Try a right-to-left search for the last closing brace
def find_matching_brace(text, start_pos):
    depth = 1
    for i in range(start_pos, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return -1

def extract_boxed(text):
    left_bound = r"\boxed{"
    start = text.find(left_bound)
    if start == -1:
        return "[not found]"
    
    start_content = start + len(left_bound)
    end = find_matching_brace(text, start_content)
    
    if end == -1:
        return "[no matching brace]"
    
    return text[start:end+1]

print(f"Balanced brace extract: {extract_boxed(input_text)}")