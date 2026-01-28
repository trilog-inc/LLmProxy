"""Utility functions for LLM proxy."""


def clean_tool_call_arguments(arguments: str, request_id: str = None) -> tuple[str, bool]:
    """
    Clean tool call arguments by removing problematic markers that can leak through from upstream.

    Args:
        arguments: JSON string containing tool call arguments
        request_id: Optional request ID for logging context

    Returns:
        Tuple of (cleaned_arguments, bool_was_cleaned)
    """
    if not arguments:
        return arguments, False

    # Define markers as they appear in the source logs
    PROBLEMATIC_MARKERS = [
        "<|tool_calls_section_begin|>",      # First marker from logs
        "<|tool_calls_section_end|>",     # Second marker from logs  
        "<|tool_call_begin|>",    # Third marker from logs
        "<|tool_call_end|>",   # Fourth marker from logs
        "<|tool_call_argument_begin|>",  # Fifth marker from logs
    ]
    
    original_arguments = arguments
    cleaned_arguments = arguments
    was_cleaned = False
    
    # Remove each problematic marker from arguments
    for marker in PROBLEMATIC_MARKERS:
        if marker in cleaned_arguments:
            cleaned_arguments = cleaned_arguments.replace(marker, "")
            was_cleaned = True

    # If we cleaned markers, log it
    if was_cleaned and request_id:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"[Request {request_id}] Cleaned problematic markers from tool call arguments"
        )

    return cleaned_arguments, was_cleaned