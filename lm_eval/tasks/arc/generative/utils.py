import re

def extract_boxed_answer(text):
    """Extract the answer from a boxed format: \boxed{X}"""
    pattern = r"\\boxed\{([A-D])\}"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None