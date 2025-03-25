import re
import string
from typing import Union, List, Dict, Optional, Any, Iterable

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
                        if candidate is not None:
                            extraction = candidate
                    if extraction is None:
                        extraction = self.fallback
                else:
                    extraction = resp
                instance_filtered.append(extraction)
            filtered_resps.append(instance_filtered)
        return filtered_resps
