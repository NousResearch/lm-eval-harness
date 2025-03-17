from lm_eval.filters.extraction import ExtractAnswerFilter

# Create a raw string for the input
input_text = r"here's my answer: \boxed{\frac{3}{4}}."
print(f"Input text: {repr(input_text)}")

# Create the filter
filter = ExtractAnswerFilter(bounds=[
    {"left_bound": r"\boxed{", "right_bound": "}", "include_bounds": True}
])

# Apply the filter
result = filter.apply([[input_text]], [{}])
print(f"Result: {result}")

# Try with a real backslash in the text
real_input = "here's my answer: \\boxed{\\frac{3}{4}}."
print(f"Real input: {repr(real_input)}")
real_result = filter.apply([[real_input]], [{}])
print(f"Real result: {real_result}")

# Try with different escape handling
filter2 = ExtractAnswerFilter(bounds=[
    {"left_bound": "\\boxed{", "right_bound": "}", "include_bounds": True}
])
result2 = filter2.apply([[real_input]], [{}])
print(f"Result with double backslash: {result2}")

# Try with even more escaping 
filter3 = ExtractAnswerFilter(bounds=[
    {"left_bound": "\\\\boxed{", "right_bound": "}", "include_bounds": True}
])
result3 = filter3.apply([[real_input]], [{}])
print(f"Result with quadruple backslash: {result3}")