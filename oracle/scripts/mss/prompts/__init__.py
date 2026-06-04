"""Prompt loading and management for MSS oracle."""

import re
from pathlib import Path
from typing import Dict, Optional

PROMPTS_DIR = Path(__file__).parent

# Cache for loaded prompts
_prompt_cache: Dict[str, str] = {}


def load_prompt(name: str) -> str:
    """Load prompt template by name.

    Args:
        name: Prompt name (without .md extension)

    Returns:
        Prompt template content with YAML frontmatter stripped

    Raises:
        ValueError: If prompt file not found
    """
    prompt_file = PROMPTS_DIR / f"{name}.md"
    if not prompt_file.exists():
        raise ValueError(f"Prompt not found: {name}")

    content = prompt_file.read_text()

    # Strip YAML frontmatter if present
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2]

    return content.strip()


def get_prompt(name: str = "mss_verification") -> str:
    """Get prompt template by name (with caching).

    Args:
        name: Prompt name (default: mss_verification)

    Returns:
        Prompt template content
    """
    if name not in _prompt_cache:
        _prompt_cache[name] = load_prompt(name)
    return _prompt_cache[name]


def list_prompts() -> list:
    """List all available prompt templates.

    Returns:
        List of prompt names (without .md extension)
    """
    return [p.stem for p in PROMPTS_DIR.glob("*.md")]


def get_prompt_metadata(name: str) -> Optional[Dict]:
    """Get metadata from prompt YAML frontmatter.

    Args:
        name: Prompt name

    Returns:
        Dictionary of metadata or None if no frontmatter
    """
    prompt_file = PROMPTS_DIR / f"{name}.md"
    if not prompt_file.exists():
        return None

    content = prompt_file.read_text()

    if not content.startswith("---"):
        return None

    parts = content.split("---", 2)
    if len(parts) < 3:
        return None

    # Parse simple YAML frontmatter
    frontmatter = parts[1].strip()
    metadata = {}

    for line in frontmatter.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            metadata[key] = value

    return metadata
