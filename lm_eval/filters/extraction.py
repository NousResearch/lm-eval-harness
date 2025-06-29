import re
import string
from typing import Union, List, Dict, Optional, Any, Iterable, Pattern

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


@register_filter("regex")
class RegexFilter(Filter):
    """A filter that extracts values from text using regex pattern matching.

    This filter applies a regex pattern to each model response and extracts matched values.
    If no match is found, returns a fallback value. Useful for extracting structured data
    (like numbers) from unstructured model outputs.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
        fallback_regex: list[str] = None,
        fallback_regex_group_select: list[int] = None,
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.fallback_regex = (
            [re.compile(r) for r in fallback_regex] if fallback_regex else None
        )
        self.fallback_regex_group_select = (
            fallback_regex_group_select
            if fallback_regex_group_select
            else group_select * len(fallback_regex)
            if fallback_regex
            else None
        )
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)
        def process_match(
            match: Union[list[str], list[tuple[str, ...]], tuple[str, ...]],
            group_select: int = self.group_select,
            fallback: str = self.fallback,
        ) -> str:
            """Helper function to process regex match results"""
            if not match:
                return fallback

            match = match[group_select]
            if isinstance(match, tuple):
                # Filter out empty strings and get first non-empty match if it exists
                valid_matches = [m for m in match if m]
                return valid_matches[0].strip() if valid_matches else fallback
            return match.strip()

        def try_fallback_regex(resp: str) -> str:
            """Helper function to attempt fallback regex patterns"""
            for regex, group_select in zip(
                self.fallback_regex, self.fallback_regex_group_select
            ):
                match = regex.findall(resp)
                if match:
                    return process_match(match, group_select)
            return self.fallback

        def filter_response(resp: str) -> str:
            """Process a single response string"""
            # Try primary regex first
            match = self.regex.findall(resp)
            if match:
                return process_match(match)

            # If primary regex fails and fallback_regex exists, try those
            return try_fallback_regex(resp) if self.fallback_regex else self.fallback

        return [
            [filter_response(resp) for resp in response_set] for response_set in resps
        ]


@register_filter("remove_whitespace")
class WhitespaceFilter(Filter):
    """Filters out leading whitespace from responses."""

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        def filter_set(inst):
            filtered_resp = []
            for resp in inst:
                resp = resp.lstrip()
                filtered_resp.append(resp)
            return filtered_resp

        filtered_resps = [filter_set(resp) for resp in resps]

        return filtered_resps


@register_filter("multi_choice_regex")
class MultiChoiceRegexFilter(RegexFilter):
    """A filter for extracting multiple choice answers from text responses.

    This filter processes responses in the following order:
    1. Full text matches of answer choices (e.g., "The earth is round" -> "A")
    2. Letter-based answers in various formats (e.g., "(A)", "A:", "Answer: A")

    Args:
        regex_pattern (str, optional): Custom regex pattern for matching. If None, uses default pattern.
        group_select (int, default=0): Which regex group to select from matches.
        fallback (str, default="[invalid]"): Value to return when no match is found.
        ignore_case (bool, default=True): Whether to ignore case when matching.
        ignore_punctuation (bool, default=False): Whether to ignore punctuation when matching.
        regexes_to_ignore (list, optional): List of regex patterns to remove from text before matching.
        max_choices (int, default=4): Maximum number of choices to consider (A-D).
        choices_field (str, default="choices"): Field name or dot path to get choices from document.
        format_style (str, default="plain"): Output format style ("plain" for "A", "parens" for "(A)").

    Examples:
        >>> filter = MultiChoiceRegexFilter(format_style="parens", choices_field="choices")
        >>> doc = {"choices": ["The earth is round", "The earth is flat"]}
        >>> responses = ["The earth is round", "Answer: B", "(A)"]
        >>> filter.apply([responses], [doc])
        [[["(A)", "(B)", "(A)"]]]

        # With nested choices
        >>> doc = {"metadata": {"question": {"choices": ["True", "False"]}}}
        >>> filter = MultiChoiceRegexFilter(choices_field="metadata.question.choices")

        # With custom format
        >>> filter = MultiChoiceRegexFilter(format_style="plain")  # Returns "A" instead of "(A)"
    """

    def __init__(
        self,
        regex_pattern: str = None,
        group_select: int = 0,
        fallback: str = "[invalid]",
        ignore_case: bool = True,
        ignore_punctuation: bool = False,
        regexes_to_ignore: list = None,
        max_choices: int = 4,  # A-Z
        choices_field: str = "choices",
        format_style: str = "plain",
    ) -> None:
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore
        self.max_choices = max_choices
        self.choices_field = choices_field
        self.format_style = format_style

        # If no custom pattern, create comprehensive letter pattern
        if regex_pattern is None:
            letters = "".join([chr(ord("A") + i) for i in range(max_choices)])
            # Matches (A), A:, Answer: A etc.
            regex_pattern = rf"(?:\(([{letters}])\))|(?:(?:answer|choice|option)?:?\s*([{letters}])(?:\s|$))"

        super().__init__(regex_pattern, group_select, fallback)

    def _format_letter(self, letter: str) -> str:
        """Format a letter based on format_style setting"""
        if self.format_style == "parens":
            return f"({letter})"
        elif self.format_style == "plain":
            return letter
        # Add more format styles here as needed:
        # elif self.format_style == "brackets":
        #     return f"[{letter}]"
        # elif self.format_style == "numbered":
        #     return f"{ord(letter) - ord('A') + 1}"
        else:
            return f"({letter})"

    def _filter_text(self, text: str) -> str:
        """Apply text filtering rules (case, punctuation, regex ignores)"""
        if self.regexes_to_ignore is not None:
            for pattern in self.regexes_to_ignore:
                text = re.sub(pattern, "", text)

        if self.ignore_case:
            text = text.lower()

        if self.ignore_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        return text.strip()

    def _build_choice_patterns(self, choices: list[str]) -> tuple:
        """
        Build regex patterns and conversion maps for both full text
        and letter-based answers.
        """
        # For matching full text of choices
        choice_patterns = []
        choice_to_letter = {}

        # For matching letter answers
        letter_map = {}  # Maps raw letters to (A) format

        for i, choice in enumerate(choices):
            if i >= self.max_choices:
                break

            # Get the letter for this choice (A, B, C, etc)
            letter = chr(ord("A") + i)
            formatted_letter = self._format_letter(letter)

            # Process the choice text
            processed_choice = self._filter_text(choice)

            # Add to full text matching
            choice_patterns.append(re.escape(processed_choice))
            choice_to_letter[processed_choice] = formatted_letter

            # Add to letter matching
            letter_map[letter] = formatted_letter

        # Create regex for full text matches
        full_text_pattern = "|".join(choice_patterns) if choice_patterns else "(?!)"

        # Create regex for letter matches (: A, (A), etc)
        # If no choices given, use default A-Z range based on max_choices
        if not letter_map:
            letters = "".join([chr(ord("A") + i) for i in range(self.max_choices)])
        else:
            letters = "".join(letter_map.keys())

        letter_pattern = rf"(?:\(([{letters}])\))|(?:(?:answer|choice|option)?:?\s*([{letters}])(?:\s|$))"

        return (
            re.compile(full_text_pattern),
            re.compile(letter_pattern),
            choice_to_letter,
            letter_map,
        )

    def _get_choices(self, doc: dict) -> list:
        """
        Safely extract choices from the document using the specified field name.
        Handles nested fields using dot notation (e.g., "metadata.choices").
        Returns empty list if:
        - doc is None or not a dict
        - field doesn't exist
        - field value is None
        - field value is not a list
        """
        if doc is None or not isinstance(doc, dict):
            return []

        if "." in self.choices_field:
            # Handle nested fields
            fields = self.choices_field.split(".")
            value = doc
            for field in fields:
                if not isinstance(value, dict) or field not in value:
                    return []
                value = value[field]
                if value is None:
                    return []
            assert isinstance(value, list)
            return value
        else:
            # Direct field access
            value = doc.get(self.choices_field)
            if value is None:
                return []
            assert isinstance(value, list)
            return value

    def _find_match(
        self, regex, text: str, conversion_map: dict = None
    ) -> Union[str, None]:
        """Find regex matches and convert using the provided map if any."""
        matches = regex.findall(text)
        if not matches:
            return None

        # Handle both single matches and tuple groups
        match = matches[self.group_select]
        if isinstance(match, tuple):
            # Take first non-empty group
            match = next((m for m in match if m), None)

        if match and conversion_map:
            return conversion_map.get(match, match)

        return match

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        filtered_resps = []

        for responses, doc in zip(resps, docs):
            choices = self._get_choices(doc)

            # Build patterns for both full text and letter matching
            full_text_re, letter_re, choice_to_letter, letter_map = (
                self._build_choice_patterns(choices)
            )

            filtered = []
            for resp in responses:
                match = None

                # Try the custom regex pattern first (if provided)
                if self.regex_pattern != letter_re.pattern:
                    match = self._find_match(self.regex, resp)

                if not match:
                    # Try matching full text of choices
                    processed_resp = self._filter_text(resp)
                    match = self._find_match(
                        full_text_re, processed_resp, choice_to_letter
                    )

                    if not match:
                        # Try matching letter patterns
                        if self.ignore_case:
                            resp = resp.upper()
                        match = self._find_match(letter_re, resp, letter_map)

                filtered.append(match if match else self.fallback)

            filtered_resps.append(filtered)

        return filtered_resps

@register_filter("regex_fallback_to_original_nous")
class RegexFallbackToOriginalFilter(Filter):
    """A filter that extracts text using regex pattern matching with fallback to original text.
    
    This filter searches for a regex pattern in the text and returns the specified group if found.
    If no match is found, it returns the original text instead of a fallback string.
    
    Args:
        regex_pattern (str): The regex pattern to search for
        group_select (int, default=-1): The group index to extract from the match 
                                        (-1 means the last group)
    
    Examples:
        >>> filter = RegexFallbackToOriginalFilter(regex_pattern=r"Answer: (.*)")
        >>> filter.apply([["Answer: 42"]], [{}])
        [["42"]]
        
        >>> filter = RegexFallbackToOriginalFilter(regex_pattern=r"Not found: (.*)")
        >>> filter.apply([["Answer: 42"]], [{}])
        [["Answer: 42"]]  # Returns original text since regex didn't match
    """
    
    def __init__(
        self,
        regex_pattern: str,
        group_select: int = -1,
    ) -> None:
        """Initialize the regex filter with fallback to original text.
        
        Args:
            regex_pattern: Regular expression pattern to search for
            group_select: Which regex capture group to return (-1 for last group)
        """
        self.regex_pattern = regex_pattern
        try:
            self.regex = re.compile(regex_pattern)
        except Exception as e:
            print("failed to compile pattern:", regex_pattern)
            raise e
        self.group_select = group_select
        
    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        """Apply the regex filter to responses with fallback to original text.
        
        Args:
            resps: List of response lists
            docs: List of document dictionaries
            
        Returns:
            List of filtered response lists
        """
        def process_response(resp: str) -> str:
            # Try to match the regex
            match = self.regex.search(resp)
            if match:
                # If we have a match, extract the specified group
                groups = match.groups()
                if not groups:
                    # No capture groups, return the entire match
                    return match.group(0)
                
                # Select the appropriate group
                group_idx = self.group_select
                if group_idx < 0:
                    # Convert negative index to positive
                    group_idx = len(groups) + group_idx
                
                # Ensure group index is within bounds
                if 0 <= group_idx < len(groups):
                    group = groups[group_idx]
                    return group if group is not None else resp
                
                # If the group index is out of bounds, return the original text
                return resp
            
            # If no match is found, return the original text
            return resp
            
        # Process each response
        return [
            [process_response(resp) for resp in response_set]
            for response_set in resps
        ]


@register_filter("extract_answer_nous")
class ExtractAnswerFilter(Filter):
    """A filter that extracts text between specified bounds.
    
    This filter extracts text between left and right bounds in a response.
    It can handle multiple bound pairs and returns the rightmost (last) match.
    If a left bound is missing, it uses the start of the string.
    If a right bound is missing, it uses the end of the string.
    
    Args:
        bounds (List[Dict[str, str]], optional): A list of bound dictionaries with keys:
            - "left_bound" (str): The left boundary text (start of string if empty)
            - "right_bound" (str): The right boundary text (end of string if empty)
            - "include_bounds" (Union[bool, str], optional): Controls which bounds to include:
              - False: Don't include either bound (default)
              - True/"both": Include both bounds
              - "left": Include only the left bound
              - "right": Include only the right bound
            - "case_insensitive" (Union[bool, str], optional): Controls case sensitivity:
              - False: Case-sensitive matching (default)
              - True/"both": Case-insensitive for both bounds
              - "left": Case-insensitive only for left bound
              - "right": Case-insensitive only for right bound
        fallback (str, default="[invalid]"): Value to return when no match is found

    Examples:
        >>> filter = ExtractAnswerFilter(bounds=[{"left_bound": "<answer>", "right_bound": "</answer>"}])
        >>> filter.apply([["The <answer>42</answer> is correct."]], [{}])
        [["42"]]
        
        >>> filter = ExtractAnswerFilter(bounds=[
        ...     {"left_bound": "", "right_bound": ".", "include_bounds": "right"}, 
        ...     {"left_bound": "Answer: ", "right_bound": "", "include_bounds": "left"}
        ... ])
        >>> filter.apply([["Answer: 42"]], [{}])
        [["Answer: 42"]]
        
        >>> # Without specifying bounds, it returns the entire text
        >>> filter = ExtractAnswerFilter()
        >>> filter.apply([["This is the complete response."]], [{}])
        [["This is the complete response."]]
    """

    def __init__(self, bounds: Optional[List[Dict[str, Any]]] = None, fallback: str = "[invalid]", **kwargs) -> None:
        """
        Initializes the ExtractAnswerFilter.

        Args:
            bounds (Optional[List[Dict[str, Any]]]): List of bound dictionaries.
            fallback (str): Fallback value if extraction fails.
        """
        self.bounds = bounds
        self.fallback = fallback

    def _extract_with_bound(self, text: str, bound: Dict[str, Any], strip: bool = True) -> Optional[str]:
        """
        Extract text from the given text using a single bound dictionary.

        Args:
            text (str): The text to extract from.
            bound (Dict[str, Any]): A bound dictionary containing keys:
                - "left_bound" (str)
                - "right_bound" (str)
                - "include_bounds" (Union[bool, str])
                - "case_insensitive" (Union[bool, str])

        Returns:
            Optional[str]: The extracted text if the bounds match; otherwise, None.
        """
        left_bound = bound.get("left_bound", "")
        right_bound = bound.get("right_bound", "")
        include_bounds = bound.get("include_bounds", False)
        case_insensitive = bound.get("case_insensitive", False)

        # Determine left index.
        if left_bound == "":
            left_index = 0
        else:
            if case_insensitive in [True, "both", "left"]:
                left_index = text.lower().rfind(left_bound.lower())
            else:
                left_index = text.rfind(left_bound)
            if left_index == -1:
                return None
            if not (include_bounds in [True, "both", "left"]):
                left_index += len(left_bound)

        # Determine right index.
        if right_bound == "":
            right_index = len(text)
        else:
            # For case_insensitive in [True, "both"], use forgiving search.
            if case_insensitive in [True, "both"]:
                found = text.lower().find(right_bound.lower(), left_index)
                if found == -1:
                    right_index = len(text)
                else:
                    right_index = found
                    if include_bounds in [True, "both", "right"]:
                        right_index += len(right_bound)
            else:
                # For strict matching (covers case_insensitive == False or "left" or "right")
                found = text.find(right_bound, left_index)
                if found == -1:
                    if case_insensitive == "left":
                        # Try a case-insensitive search to decide if the right bound exists in a different case.
                        found_ci = text.lower().find(right_bound.lower(), left_index)
                        if found_ci == -1:
                            # Truly missing: forgive and use end-of-string.
                            right_index = len(text)
                        else:
                            # Right bound exists but case differs → treat as no match.
                            return None
                    else:
                        return None
                else:
                    right_index = found
                    if include_bounds in [True, "both", "right"]:
                        right_index += len(right_bound)
        out = text[left_index:right_index]
        if strip:
            return out.strip()
        return out

    def apply(self, resps: Union[List, Iterable], docs: List[dict]) -> Iterable:
        """
        Applies the extraction filter on the responses.

        For each response string in each instance, this method iterates over the provided bounds.
        If a bound yields a match, it updates the extraction so that the last successful extraction
        (i.e. the rightmost match) is used. If no bounds match, the fallback value is returned.
        When no bounds are provided, the entire response is returned.
        The output mirrors the input structure: one list per instance, each containing a filtered string.

        Args:
            resps (Union[List, Iterable]): A list of response lists (each a list of response strings).
            docs (List[dict]): A list of corresponding documents (unused in this filter).

        Returns:
            Iterable: The filtered responses (list of lists of strings) in the same order as input.
        """
        filtered_resps = []
        for instance_resps in resps:
            instance_filtered = []
            for resp in instance_resps:
                extraction = None
                if self.bounds:
                    for bound in self.bounds:
                        candidate = self._extract_with_bound(resp, bound)
                        if candidate is not None and candidate != "":
                            extraction = candidate
                    if extraction is None:
                        extraction = self.fallback
                else:
                    extraction = resp
                instance_filtered.append(extraction)
            filtered_resps.append(instance_filtered)
        return filtered_resps


@register_filter("nous_lighteval_mc")
class NousLightevalMCFilter(Filter):
    """A filter for extracting multiple choice answers from LightEval-formatted responses.
    
    This filter processes multiple-choice responses in the format used by LightEval for 
    benchmarks like MMLU and GPQA. It supports both letter-based and full text matching,
    and handles the LightEval formatting standards.
    
    Args:
        fallback (str, default="[invalid]"): Value to return when no match is found.
        ignore_case (bool, default=True): Whether to ignore case when matching.
        ignore_punctuation (bool, default=False): Whether to ignore punctuation when matching.
        format_style (str, default="plain"): Output format style ("plain" for "A", "parens" for "(A)").
        max_choices (int, default=26): Maximum number of choices to consider (A-Z).
        answer_word (str, default="Answer"): The word preceding the answer (e.g., "Answer:").
        
    Examples:
        >>> filter = NousLightevalMCFilter()
        >>> docs = [{"choices": ["Paris", "London", "Berlin", "Rome"]}]
        >>> filter.apply([["Answer: A", "The capital of France is Paris"]], docs)
        [[["A", "A"]]]
    """
    
    def __init__(
        self, 
        fallback: str = "[invalid]",
        ignore_case: bool = True,
        ignore_punctuation: bool = False,
        format_style: str = "plain", 
        max_choices: int = 26,
        answer_word: str = "Answer",
    ) -> None:
        """Initialize the LightEval multiple choice filter."""
        self.fallback = fallback
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.format_style = format_style
        self.max_choices = max_choices
        self.answer_word = answer_word
        
        # Prepare letter patterns (standard format like used in MMLU and GPQA)
        self.letters = "".join([chr(ord("A") + i) for i in range(max_choices)])
        
        # Core regex patterns for letter matching
        # This handles formats like:
        # - "A" (plain letter)
        # - "(A)" (parenthesized letter)
        # - "A:" (letter with colon)
        # - "Answer: A" (word followed by letter)
        # - "The answer is A" (phrase containing letter)
        self.letter_pattern = rf"(?:\(([{self.letters}])\))|(?:(?:{answer_word}|choice|option)?:?\s*([{self.letters}])(?:\s|$|\.|\,|\)|\]|\}}|:|;))"
        self.letter_regex = re.compile(self.letter_pattern, re.IGNORECASE if ignore_case else 0)
        
        # For matching numerical responses like "1", "2", "3" (common in some benchmarks)
        self.number_pattern = r"(?:\(([1-9][0-9]*)\))|(?:(?:answer|choice|option)?:?\s*([1-9][0-9]*)(?:\s|$|\.|\,|\)|\]|\}}|:|;))"
        self.number_regex = re.compile(self.number_pattern)
    
    def _format_letter(self, letter: str) -> str:
        """Format a letter based on format_style setting."""
        if self.format_style == "parens":
            return f"({letter})"
        elif self.format_style == "plain":
            return letter
        # Additional formats could be added here
        else:
            return letter
            
    def _filter_text(self, text: str) -> str:
        """Apply text filtering rules (case, punctuation)."""
        if self.ignore_case:
            text = text.lower()
            
        if self.ignore_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))
            
        return text.strip()
    
    def _build_choice_patterns(self, choices: list[str]) -> tuple:
        """Build regex patterns and mapping for both full text and letter-based answers."""
        # For matching full text of choices
        choice_patterns = []
        choice_to_letter = {}
        
        # For matching letter answers
        letter_map = {}  # Maps raw letters to desired format
        
        for i, choice in enumerate(choices):
            if i >= self.max_choices:
                break
                
            # Get letter for this choice (A, B, C, etc.)
            letter = chr(ord("A") + i)
            formatted_letter = self._format_letter(letter)
            
            # Process choice text
            processed_choice = self._filter_text(choice)
            
            # Add to full text matching
            if processed_choice:
                choice_patterns.append(re.escape(processed_choice))
                choice_to_letter[processed_choice] = formatted_letter
            
            # Add to letter matching
            letter_map[letter.upper()] = formatted_letter
            letter_map[letter.lower()] = formatted_letter
            
            # Add numerical mapping (1-based)
            number = str(i + 1)
            letter_map[number] = formatted_letter
            
        # Create regex for full text matches
        full_text_pattern = "|".join(choice_patterns) if choice_patterns else "(?!)"
        full_text_regex = re.compile(full_text_pattern, re.IGNORECASE if self.ignore_case else 0)
        
        return full_text_regex, choice_to_letter, letter_map
        
    def _get_choices(self, doc: dict) -> list:
        """Extract choices from the document, handle both direct and 'query_choices' field."""
        if doc is None or not isinstance(doc, dict):
            return []
            
        # Try standard 'choices' field first
        choices = doc.get("choices")
        if isinstance(choices, list) and choices:
            return choices
            
        # Try LightEval-style field name
        for field in ["query_choices", "choices_list", "options"]:
            choices = doc.get(field)
            if isinstance(choices, list) and choices:
                return choices
                
        return []
    
    def _find_letter_match(self, text: str, letter_map: dict) -> Union[str, None]:
        """Find letter/number matches in the response."""
        # Try letter pattern first (A, B, C)
        letter_matches = self.letter_regex.findall(text)
        if letter_matches:
            for match in letter_matches:
                if isinstance(match, tuple):
                    # Take first non-empty group
                    match = next((m for m in match if m), None)
                if match and match.upper() in letter_map:
                    return letter_map[match.upper()]
        
        # Try number pattern (1, 2, 3)
        number_matches = self.number_regex.findall(text)
        if number_matches:
            for match in number_matches:
                if isinstance(match, tuple):
                    match = next((m for m in match if m), None)
                if match and match in letter_map:
                    return letter_map[match]
                    
        return None
    
    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        """Apply the filter to extract multiple choice answers from responses."""
        filtered_resps = []
        
        for responses, doc in zip(resps, docs):
            choices = self._get_choices(doc)
            full_text_re, choice_to_letter, letter_map = self._build_choice_patterns(choices)
            
            filtered = []
            for resp in responses:
                # Skip empty responses
                if not resp or resp.isspace():
                    filtered.append(self.fallback)
                    continue
                    
                match = None
                
                # Try matching full text of choices
                processed_resp = self._filter_text(resp)
                for choice_text, letter in choice_to_letter.items():
                    if choice_text in processed_resp:
                        match = letter
                        break
                
                # If no full text match, try letter/number matching
                if not match:
                    match = self._find_letter_match(resp, letter_map)
                    
                filtered.append(match if match else self.fallback)
                
            filtered_resps.append(filtered)
            
        return filtered_resps


@register_filter("comprehensive_answer_extraction")
class ComprehensiveAnswerExtractionFilter(Filter):
    """A truly comprehensive filter for extracting answers from any LLM response format.
    
    This filter uses a multi-stage hierarchical approach to handle ALL possible answer formats:
    - Mathematical containers: \\boxed{...}, \\text{...}
    - Answer indicators: Final answer, the answer is, best answer
    - XML/HTML tags: <answer>...</answer>
    - Structural markers: **, (), [], etc.
    - Special cases: Dyck paths, mixed combinations
    - Nested formats: \\boxed{**Final answer: A)**}
    
    The filter works by:
    1. Extracting content from containers (\\boxed, <answer>, etc.)
    2. Recursively parsing extracted content for answer indicators
    3. Classifying and cleaning final answers
    4. Multiple fallback layers for edge cases
    
    Args:
        fallback (str, default="[invalid]"): Value to return when no match is found
    """
    
    def __init__(self, fallback: str = "[invalid]") -> None:
        """Initialize the comprehensive answer extraction filter."""
        self.fallback = fallback
        
        # Container extraction patterns (highest priority)
        self.container_patterns = [
            # LaTeX containers - capture everything inside
            (re.compile(r'\\boxed\{(.*?)\}', re.IGNORECASE | re.DOTALL), 1),
            (re.compile(r'\\text\{(.*?)\}', re.IGNORECASE | re.DOTALL), 1),
            
            # XML/HTML tags
            (re.compile(r'<answer>(.*?)</answer>', re.IGNORECASE | re.DOTALL), 1),
            (re.compile(r'<solution>(.*?)</solution>', re.IGNORECASE | re.DOTALL), 1),
            
            # Markdown/structured containers
            (re.compile(r'\*\*(.*?)\*\*'), 1),  # **content**
            (re.compile(r'`(.*?)`'), 1),        # `content`
        ]
        
        # Answer indicator patterns (medium priority)
        self.indicator_patterns = [
            # Final answer variants
            (re.compile(r'final\s+answer\s*:?\s*(.*?)(?:\s*$|\s*\.|$)', re.IGNORECASE | re.MULTILINE), 1),
            (re.compile(r'the\s+final\s+answer\s+is\s*(.*?)(?:\s*$|\s*\.|$)', re.IGNORECASE | re.MULTILINE), 1),
            
            # "The answer is" variants
            (re.compile(r'the\s+answer\s+is\s*:?\s*(.*?)(?:\s*$|\s*\.|$)', re.IGNORECASE | re.MULTILINE), 1),
            (re.compile(r'answer\s*:?\s*(.*?)(?:\s*$|\s*\.|$)', re.IGNORECASE | re.MULTILINE), 1),
            
            # "Best answer" variants
            (re.compile(r'best\s+answer\s+is\s*:?\s*(.*?)(?:\s*$|\s*\.|$)', re.IGNORECASE | re.MULTILINE), 1),
            (re.compile(r'the\s+best\s+answer\s+is\s*:?\s*(.*?)(?:\s*$|\s*\.|$)', re.IGNORECASE | re.MULTILINE), 1),
            
            # Choice/Option indicators
            (re.compile(r'choice\s*:?\s*(.*?)(?:\s*$|\s*\.|$)', re.IGNORECASE | re.MULTILINE), 1),
            (re.compile(r'option\s*:?\s*(.*?)(?:\s*$|\s*\.|$)', re.IGNORECASE | re.MULTILINE), 1),
            (re.compile(r'select\s*:?\s*(.*?)(?:\s*$|\s*\.|$)', re.IGNORECASE | re.MULTILINE), 1),
            
            # Solution/Result indicators
            (re.compile(r'solution\s*:?\s*(.*?)(?:\s*$|\s*\.|$)', re.IGNORECASE | re.MULTILINE), 1),
            (re.compile(r'result\s*:?\s*(.*?)(?:\s*$|\s*\.|$)', re.IGNORECASE | re.MULTILINE), 1),
        ]
        
        # Direct pattern matching for simple cases
        self.direct_patterns = [
            # Parenthesized letters/symbols
            (re.compile(r'\(([A-Za-z0-9]+(?:\)|\])*)\)'), 1),
            (re.compile(r'\[([A-Za-z0-9]+(?:\)|\])*)\]'), 1),
            
            # Letters followed by punctuation
            (re.compile(r'\b([A-Za-z])\s*\)\s*\.?\s*$', re.MULTILINE), 1),  # A) at end of line
            (re.compile(r'\b([A-Za-z])\s*\.?\s*$', re.MULTILINE), 1),       # A. at end of line
            (re.compile(r'^([A-Za-z])\s*[\.\):\-]', re.MULTILINE), 1),       # A: at start of line
            
            # Special symbols (Dyck paths, etc.)
            (re.compile(r'([)\]]+)'), 1),    # Closing brackets/parens
            (re.compile(r'([\(\[]+)'), 1),   # Opening brackets/parens
            (re.compile(r'([)\]\(\[]+)'), 1), # Mixed brackets
        ]
    
    def _clean_extracted_content(self, content: str) -> str:
        """Clean extracted content by removing common noise."""
        if not content:
            return ""
            
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Remove common noise patterns
        content = re.sub(r'^\s*[-*•]\s*', '', content)  # Remove bullet points
        content = re.sub(r'^\s*\d+\.\s*', '', content)  # Remove numbering
        
        return content
    
    def _extract_from_content(self, content: str, depth: int = 0) -> Optional[str]:
        """Extract answer from already-extracted content using recursive parsing."""
        if not content or depth > 5:  # Prevent infinite recursion
            return None
            
        content = self._clean_extracted_content(content)
        
        # First try to extract from containers within the content
        for pattern, group_idx in self.container_patterns:
            match = pattern.search(content)
            if match:
                inner_content = match.group(group_idx)
                # Recursively extract from inner content
                inner_result = self._extract_from_content(inner_content, depth + 1)
                if inner_result:
                    return inner_result
        
        # Then try answer indicators
        for pattern, group_idx in self.indicator_patterns:
            match = pattern.search(content)
            if match:
                answer_content = match.group(group_idx).strip()
                if answer_content:
                    # Recursively process the found content
                    recursive_result = self._extract_from_content(answer_content, depth + 1)
                    if recursive_result:
                        return recursive_result
                    # If no recursive match, clean and return the content
                    return self._normalize_answer(answer_content)
        
        # Finally try direct patterns
        for pattern, group_idx in self.direct_patterns:
            match = pattern.search(content)
            if match:
                return self._normalize_answer(match.group(group_idx))
        
        # If content looks like a simple answer, return it normalized
        normalized = self._normalize_answer(content)
        if normalized and len(normalized) <= 20:  # Reasonable answer length
            return normalized
            
        return None
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize extracted answer to standard format."""
        if not answer:
            return ""
            
        answer = answer.strip()
        
        # Remove common punctuation and formatting
        answer = re.sub(r'^\s*[*_`"\']*(.*?)[*_`"\']*\s*$', r'\1', answer)
        answer = re.sub(r'^\s*[-•]\s*', '', answer) # Remove bullets
        
        # Handle parentheses and brackets
        if answer.startswith('(') and answer.endswith(')'):
            answer = answer[1:-1].strip()
        if answer.startswith('[') and answer.endswith(']'):
            answer = answer[1:-1].strip()
            
        # Split on common separators and take first part for choice questions
        parts = re.split(r'[)\]\s]\s*', answer, 1)
        if len(parts) > 1 and len(parts[0]) <= 3:  # Likely a choice letter
            answer = parts[0]
        
        # Clean up remaining punctuation
        answer = re.sub(r'[^\w)\]\(\[]+$', '', answer)  # Remove trailing punctuation except brackets
        answer = re.sub(r'^[^\w)\]\(\[]+', '', answer)  # Remove leading punctuation except brackets
        
        if not answer:
            return ""
        
        # Normalize single letters to uppercase
        if len(answer) == 1 and answer.isalpha():
            return answer.upper()
            
        # Keep special symbols (Dyck paths) as-is
        if all(c in '()[]' for c in answer):
            return answer
            
        return answer
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract answer using comprehensive multi-stage approach."""
        if not text:
            return None
            
        # Clean input text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Pre-processing: Split on </think> and take the last part
        if '</think>' in text:
            parts = text.split('</think>')
            text = parts[-1].strip()  # Take everything after the last </think>
        
        # Stage 1: Try container extraction first
        for pattern, group_idx in self.container_patterns:
            matches = pattern.findall(text)
            if matches:
                # Try each match (take the last/rightmost one as it's usually final)
                for match in reversed(matches):
                    result = self._extract_from_content(match)
                    if result:
                        return result
        
        # Stage 2: Try answer indicators on full text
        for pattern, group_idx in self.indicator_patterns:
            matches = pattern.findall(text)
            if matches:
                for match in reversed(matches):
                    result = self._extract_from_content(match)
                    if result:
                        return result
        
        # Stage 3: Try direct patterns on full text
        for pattern, group_idx in self.direct_patterns:
            matches = pattern.findall(text)
            if matches:
                for match in reversed(matches):
                    normalized = self._normalize_answer(match)
                    if normalized:
                        return normalized
        
        # Stage 4: Last resort - look for isolated single letters
        single_letter_matches = re.findall(r'\b([A-Za-z])\b', text)
        if single_letter_matches:
            return single_letter_matches[-1].upper()
            
        return None
    
    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        """Apply comprehensive answer extraction to responses."""
        import json
        import os
        from datetime import datetime
        
        # Log all raw generations before filtering (expensive inference preservation)
        log_dir = "filter_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_log_file = os.path.join(log_dir, f"raw_generations_{timestamp}.jsonl")
        
        try:
            with open(raw_log_file, 'w') as f:
                for idx, (responses, doc) in enumerate(zip(resps, docs)):
                    # Main response (first response for recompute script compatibility)
                    main_response = responses[0] if responses else ""
                    
                    # Extract target/answer from doc
                    target = None
                    for key in ['target', 'answer', 'label', 'gold', 'choices']:
                        if key in doc:
                            target = doc[key]
                            break
                    
                    # For multiple choice, extract the correct answer letter
                    if target is None and 'choices' in doc and 'gold' in doc:
                        choices = doc['choices']
                        gold_idx = doc['gold']
                        if isinstance(gold_idx, int) and 0 <= gold_idx < len(choices):
                            # Convert to letter format (A, B, C, D)
                            target = chr(ord('A') + gold_idx)
                    
                    log_entry = {
                        # Compatible with recompute script format
                        "response": main_response,
                        "target": target,
                        "doc": doc,
                        
                        # Additional metadata
                        "idx": idx,
                        "doc_id": doc.get("doc_id", f"doc_{idx}"),
                        "task": doc.get("task", "unknown"),
                        "raw_responses": responses,  # All responses for completeness
                        "timestamp": datetime.now().isoformat()
                    }
                    f.write(json.dumps(log_entry) + "\n")
            
            print(f"[FILTER LOG] Saved {len(resps)} raw generations to {raw_log_file}")
        except Exception as e:
            print(f"[FILTER LOG ERROR] Failed to save raw generations: {e}")
        
        filtered_resps = []
        filter_errors = []
        
        for idx, responses in enumerate(resps):
            filtered = []
            for resp_idx, resp in enumerate(responses):
                try:
                    answer = self._extract_answer(resp)
                    filtered.append(answer if answer is not None else self.fallback)
                except Exception as e:
                    # Log filter errors but don't crash
                    error_info = {
                        "doc_idx": idx,
                        "resp_idx": resp_idx,
                        "error": str(e),
                        "raw_response": resp[:500] + "..." if len(resp) > 500 else resp
                    }
                    filter_errors.append(error_info)
                    filtered.append(self.fallback)
            filtered_resps.append(filtered)
        
        # Log any filter errors
        if filter_errors:
            error_log_file = os.path.join(log_dir, f"filter_errors_{timestamp}.json")
            try:
                with open(error_log_file, 'w') as f:
                    json.dump(filter_errors, f, indent=2)
                print(f"[FILTER LOG] Logged {len(filter_errors)} filter errors to {error_log_file}")
            except Exception as e:
                print(f"[FILTER LOG ERROR] Failed to save filter errors: {e}")
            
        return filtered_resps
