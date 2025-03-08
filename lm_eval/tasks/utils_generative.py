import re

def extract_boxed_answer(text):
    """Extract the answer from a boxed format: \boxed{X}"""
    pattern = r"\\boxed\{([A-Za-z0-9-]+)\}"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def extract_boxed_yes_no(text):
    """Extract yes/no answer from boxed format: \boxed{yes} or \boxed{no}"""
    pattern = r"\\boxed\{(yes|no)\}"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None

def extract_boxed_number(text):
    """Extract numeric answer from boxed format: \boxed{X}"""
    pattern = r"\\boxed\{(\d+(?:\.\d+)?)\}"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None