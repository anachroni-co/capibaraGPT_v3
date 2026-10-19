"""
Enhanced Markdown Formatting Utilities for Model Responses

This module provides structured formatting with validation and improved type handling.
"""

from typing import List, Optional, Union
from dataclasses import dataclass, field

import logging
logger = logging.getLogger(__name__)


@dataclass
class MarkdownSection:
    """Base model for Markdown section validation."""
    content: Union[str, List[str]]
    enabled: bool = True


@dataclass
class MarkdownResponse:
    """Model for complete Markdown response validation."""
    sections: List[MarkdownSection]
    metadata: Optional[dict] = None


def handle_error(error_class):
    """Decorator for error handling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                raise error_class(str(e)) from e
        return wrapper
    return decorator


class DataProcessingError(Exception):
    """Exception for data processing errors."""
    pass


@handle_error(DataProcessingError)
def format_markdown_response(
    content: Union[str, List[str]],
    metadata: Optional[dict] = None
) -> str:
    """
    Format a response in Markdown with validation.

    Args:
        content: Content to format
        metadata: Additional metadata

    Returns:
        Formatted response in Markdown
    """
    # Create section
    section = MarkdownSection(content=content)

    # Create response
    response = MarkdownResponse(
        sections=[section],
        metadata=metadata
    )

    # Format
    formatted = []
    for section in response.sections:
        if section.enabled:
            if isinstance(section.content, list):
                formatted.extend(section.content)
            else:
                formatted.append(section.content)

    return "\n\n".join(formatted)


@handle_error(DataProcessingError)
def validate_markdown_response(response: str) -> bool:
    """
    Validate a Markdown response.

    Args:
        response: Response to validate

    Returns:
        True if the response is valid
    """
    try:
        # Try to create response object
        MarkdownResponse(sections=[MarkdownSection(content=response)])
        return True
    except Exception:
        return False


# Example Usage
if __name__ == "__main__":
    try:
        formatted = format_markdown_response(
            content="Adaptive Computing Basics",
            metadata={
                "title": "An Introductory Overview",
                "paragraphs": [
                    "Adaptive computing leverages adaptive mechanical phenomena to perform computations.",
                    "Qubits can exist in superposition states enabling parallel processing."
                ],
                "summary": "Fundamental concepts of adaptive computation",
                "important_points": [
                    "Uses qubits instead of classical bits",
                    "Employs superposition and entanglement",
                    "Enables exponential computational speedups for certain problems"
                ],
                "final_summary": "Adaptive computing represents a paradigm shift in computational theory"
            }
        )

        logger.info("Formatted Markdown:\n")
        logger.info(formatted)

    except DataProcessingError as e:
        logger.error(f"Error: {e}")
