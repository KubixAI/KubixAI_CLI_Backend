"""
Wildcard Pattern Matcher for Permission System

Matches commands against wildcard patterns to determine permissions.
Inspired by OpenCode's wildcard matching logic.
"""

import re
from typing import Dict, Optional, Tuple, List


class WildcardMatcher:
    """
    Matches strings against wildcard patterns.

    Patterns can contain:
    - * (asterisk) - matches any sequence of characters
    - ? (question mark) - matches any single character (future)

    More specific patterns take precedence over general ones.
    Specificity is determined by:
    1. Length of literal (non-wildcard) parts
    2. Position of wildcards (later is more specific)
    """

    @staticmethod
    def pattern_to_regex(pattern: str) -> str:
        """
        Convert wildcard pattern to regex.

        Args:
            pattern: Wildcard pattern (e.g., "rm *")

        Returns:
            Regex pattern string

        Examples:
            "rm *" -> "rm .*"
            "*" -> ".*"
            "git push *" -> "git push .*"
        """
        # Escape special regex characters except * and ?
        escaped = re.escape(pattern)

        # Replace escaped wildcards with regex equivalents
        regex = escaped.replace(r'\*', '.*')  # * matches anything
        regex = regex.replace(r'\?', '.')   # ? matches one char

        # Anchor at start and end for exact matching
        regex = f'^{regex}$'

        return regex

    @staticmethod
    def calculate_specificity(pattern: str) -> int:
        """
        Calculate specificity score for a pattern.
        Higher score = more specific.

        Specificity factors:
        1. Length of literal parts (non-wildcards)
        2. Position of first wildcard (later = more specific)

        Args:
            pattern: Wildcard pattern

        Returns:
            Specificity score (higher = more specific)

        Examples:
            "*" -> 0 (least specific)
            "rm *" -> 3 (length of "rm ")
            "rm -rf *" -> 8 (length of "rm -rf ")
            "exact_match" -> 999999 (most specific - no wildcards)
        """
        # If no wildcards, this is an exact match - highest specificity
        if '*' not in pattern and '?' not in pattern:
            return 999999

        # Count literal characters before first wildcard
        first_wildcard_pos = len(pattern)
        for i, char in enumerate(pattern):
            if char in ('*', '?'):
                first_wildcard_pos = i
                break

        # Specificity = length of literal prefix
        # This ensures "rm -rf *" is more specific than "rm *"
        return first_wildcard_pos

    @staticmethod
    def matches(command: str, pattern: str) -> bool:
        """
        Check if command matches pattern.

        Args:
            command: Command string to check
            pattern: Wildcard pattern

        Returns:
            True if command matches pattern

        Examples:
            matches("rm file.txt", "rm *") -> True
            matches("rm file.txt", "git *") -> False
            matches("git push origin", "git push *") -> True
        """
        # Normalize whitespace
        command = ' '.join(command.split())
        pattern = ' '.join(pattern.split())

        # Convert to regex and match
        regex = WildcardMatcher.pattern_to_regex(pattern)
        return bool(re.match(regex, command))

    @staticmethod
    def find_best_match(
        command: str,
        patterns: Dict[str, str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Find the most specific matching pattern for a command.

        Args:
            command: Command to match
            patterns: Dict of {pattern: permission}

        Returns:
            Tuple of (matched_pattern, permission) or (None, None)

        Algorithm:
        1. Find all matching patterns
        2. Sort by specificity (highest first)
        3. Return most specific match

        Examples:
            command: "rm -rf file.txt"
            patterns: {"*": "allow", "rm *": "ask", "rm -rf *": "deny"}
            result: ("rm -rf *", "deny")  # Most specific match
        """
        # Normalize command
        command = ' '.join(command.split())

        # Find all matching patterns
        matches: List[Tuple[str, str, int]] = []

        for pattern, permission in patterns.items():
            if WildcardMatcher.matches(command, pattern):
                specificity = WildcardMatcher.calculate_specificity(pattern)
                matches.append((pattern, permission, specificity))

        # No matches found
        if not matches:
            return None, None

        # Sort by specificity (highest first)
        matches.sort(key=lambda x: x[2], reverse=True)

        # Return most specific match
        best_pattern, best_permission, _ = matches[0]
        return best_pattern, best_permission

    @staticmethod
    def match_all(
        command: str,
        patterns: Dict[str, str]
    ) -> List[Tuple[str, str, int]]:
        """
        Find all matching patterns for a command with their specificity.

        Useful for debugging and understanding which patterns match.

        Args:
            command: Command to match
            patterns: Dict of {pattern: permission}

        Returns:
            List of (pattern, permission, specificity) sorted by specificity
        """
        command = ' '.join(command.split())

        matches: List[Tuple[str, str, int]] = []

        for pattern, permission in patterns.items():
            if WildcardMatcher.matches(command, pattern):
                specificity = WildcardMatcher.calculate_specificity(pattern)
                matches.append((pattern, permission, specificity))

        # Sort by specificity (highest first)
        matches.sort(key=lambda x: x[2], reverse=True)

        return matches


# Convenience functions for common use cases

def matches_bash_pattern(command: str, patterns: Dict[str, str]) -> Optional[str]:
    """
    Check if bash command matches any pattern and return permission.

    Args:
        command: Bash command
        patterns: Permission patterns for bash

    Returns:
        Permission string or None if no match

    Example:
        patterns = {"*": "allow", "rm *": "deny"}
        matches_bash_pattern("rm file.txt", patterns) -> "deny"
    """
    _, permission = WildcardMatcher.find_best_match(command, patterns)
    return permission


def is_dangerous_command(command: str) -> bool:
    """
    Check if command is potentially dangerous.

    Dangerous commands include:
    - rm with -rf flags
    - chmod/chown
    - dd
    - mkfs
    - shutdown/reboot
    - curl/wget with pipe to shell

    Args:
        command: Command to check

    Returns:
        True if command is dangerous
    """
    command_lower = command.lower().strip()

    # Dangerous patterns
    dangerous_patterns = [
        'rm -rf',
        'rm -fr',
        'dd if=',
        'mkfs',
        '>dev/',
        'chmod 777',
        'chmod -r 777',
        'curl.*|.*bash',
        'wget.*|.*bash',
        'curl.*|.*sh',
        'wget.*|.*sh',
        ':(){:|:&};:',  # Fork bomb
        'shutdown',
        'reboot',
        'poweroff',
        'mkfs',
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, command_lower):
            return True

    return False
