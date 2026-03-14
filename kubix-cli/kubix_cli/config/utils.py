"""
Config utilities - Deep merge, validation, etc.
"""

from typing import Dict, Any
import copy


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    The override dict takes precedence over base.
    For nested dicts, recursively merge.
    For lists, override replaces base.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = deep_merge(result[key], value)
        else:
            # Override value
            result[key] = copy.deepcopy(value)

    return result


def validate_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration dictionary.

    Args:
        config_dict: Configuration to validate

    Returns:
        Validated config dict

    Raises:
        ValueError: If config is invalid
    """
    # Check for required fields
    # Add validation logic as needed

    return config_dict


def normalize_paths(config_dict: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    """
    Normalize relative paths in config to absolute paths.

    Args:
        config_dict: Configuration dictionary
        base_dir: Base directory for relative paths

    Returns:
        Config with normalized paths
    """
    # Add path normalization logic as needed
    return config_dict


def merge_bash_permissions(
    base: Dict[str, str],
    override: Dict[str, str]
) -> Dict[str, str]:
    """
    Special merge logic for bash permissions.

    If override provides granular permissions, merge with base.
    If override provides wildcard only, use it as default.

    Args:
        base: Base bash permissions
        override: Override bash permissions

    Returns:
        Merged bash permissions
    """
    result = copy.deepcopy(base)

    # If override has only wildcard, replace default
    if "*" in override and len(override) == 1:
        result["*"] = override["*"]
    else:
        # Merge granular permissions
        result.update(override)

    return result
