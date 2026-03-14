"""
Session Data Models

Defines Session and SessionMetadata structures for conversation persistence.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
import time


@dataclass
class SessionMetadata:
    """
    Session metadata and tracking information.

    Stores metadata about a session without loading all messages.
    Used for quick lookups, filtering, and session lists.
    """

    # Identity
    name: str                           # User-friendly name
    description: str = ""               # Optional description

    # Timestamps (ISO 8601 format strings for JSON serialization)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())

    # Context
    agent_name: str = "general"         # Which agent (optimizer, debugger, etc.)
    working_dir: str = ""               # Working directory

    # Statistics
    message_count: int = 0              # Total messages
    user_messages: int = 0              # User message count
    assistant_messages: int = 0         # Assistant message count
    total_tokens: int = 0               # Approximate token count

    # Organization
    tags: List[str] = field(default_factory=list)
    pinned: bool = False                # Pin to top of list

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionMetadata':
        """Create from dictionary."""
        return cls(**data)

    def update_timestamp(self):
        """Update the updated_at and last_accessed timestamps."""
        now = datetime.now().isoformat()
        self.updated_at = now
        self.last_accessed = now

    def estimate_tokens(self, messages: List[Any]) -> int:
        """
        Estimate token count from messages.

        Rough estimation: ~4 characters per token
        """
        total_chars = sum(len(msg.content) for msg in messages if hasattr(msg, 'content'))
        return total_chars // 4


@dataclass
class Session:
    """
    A conversation session.

    Contains all messages and metadata for a complete conversation.
    Can be saved to disk and loaded later.
    """

    # Identity
    id: str                             # Unique session ID (sess_YYYYMMDD_HHMMSS_random)
    metadata: SessionMetadata           # Session metadata

    # Messages (from openrouter_v2.Message)
    messages: List[Any] = field(default_factory=list)

    # Forking support
    parent_id: Optional[str] = None     # Parent session ID if forked
    fork_point: Optional[int] = None    # Message index where forked

    # Additional data
    tags: List[str] = field(default_factory=list)
    notes: str = ""                     # User notes about session

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary for JSON serialization.

        Returns:
            Dictionary representation
        """
        # Convert messages to dicts
        messages_data = []
        for msg in self.messages:
            if hasattr(msg, '__dict__'):
                # Message object with attributes
                msg_dict = {
                    'role': msg.role,
                    'content': msg.content,
                }
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    msg_dict['tool_calls'] = msg.tool_calls
                if hasattr(msg, 'name') and msg.name:
                    msg_dict['name'] = msg.name
                messages_data.append(msg_dict)
            elif isinstance(msg, dict):
                # Already a dict
                messages_data.append(msg)

        return {
            'id': self.id,
            'metadata': self.metadata.to_dict(),
            'messages': messages_data,
            'parent_id': self.parent_id,
            'fork_point': self.fork_point,
            'tags': self.tags,
            'notes': self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """
        Create session from dictionary.

        Args:
            data: Dictionary from JSON

        Returns:
            Session instance
        """
        # Reconstruct metadata
        metadata = SessionMetadata.from_dict(data['metadata'])

        # Messages are stored as dicts, they'll be converted to Message objects when loaded into agent
        return cls(
            id=data['id'],
            metadata=metadata,
            messages=data.get('messages', []),
            parent_id=data.get('parent_id'),
            fork_point=data.get('fork_point'),
            tags=data.get('tags', []),
            notes=data.get('notes', '')
        )

    @staticmethod
    def generate_id() -> str:
        """
        Generate unique session ID.

        Format: sess_YYYYMMDD_HHMMSS_RANDOM

        Returns:
            Unique session ID
        """
        import random
        import string

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

        return f"sess_{timestamp}_{random_suffix}"

    def add_message(self, message: Any):
        """
        Add message to session and update metadata.

        Args:
            message: Message object or dict
        """
        self.messages.append(message)

        # Update metadata
        self.metadata.message_count = len(self.messages)

        # Count by role
        self.metadata.user_messages = sum(
            1 for m in self.messages
            if (hasattr(m, 'role') and m.role == 'user') or (isinstance(m, dict) and m.get('role') == 'user')
        )
        self.metadata.assistant_messages = sum(
            1 for m in self.messages
            if (hasattr(m, 'role') and m.role == 'assistant') or (isinstance(m, dict) and m.get('role') == 'assistant')
        )

        # Update token estimate
        self.metadata.total_tokens = self.metadata.estimate_tokens(self.messages)

        # Update timestamp
        self.metadata.update_timestamp()

    def get_messages_from_point(self, fork_point: int) -> List[Any]:
        """
        Get messages up to a specific point (for forking).

        Args:
            fork_point: Message index

        Returns:
            List of messages up to fork point
        """
        return self.messages[:fork_point + 1]

    def get_summary(self) -> str:
        """
        Get short summary of session.

        Returns:
            Human-readable summary
        """
        return (
            f"{self.metadata.name} ({self.metadata.agent_name}): "
            f"{self.metadata.message_count} messages, "
            f"updated {self._format_timestamp(self.metadata.updated_at)}"
        )

    @staticmethod
    def _format_timestamp(iso_timestamp: str) -> str:
        """
        Format ISO timestamp to human-readable relative time.

        Args:
            iso_timestamp: ISO 8601 timestamp string

        Returns:
            Relative time string (e.g., "2 hours ago")
        """
        try:
            dt = datetime.fromisoformat(iso_timestamp)
            now = datetime.now()
            delta = now - dt

            if delta.days > 30:
                return f"{delta.days // 30} month{'s' if delta.days // 30 > 1 else ''} ago"
            elif delta.days > 0:
                return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
            elif delta.seconds > 3600:
                hours = delta.seconds // 3600
                return f"{hours} hour{'s' if hours > 1 else ''} ago"
            elif delta.seconds > 60:
                minutes = delta.seconds // 60
                return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
            else:
                return "just now"
        except:
            return "unknown"

    def __str__(self) -> str:
        """String representation."""
        return self.get_summary()

    def __repr__(self) -> str:
        """Debug representation."""
        return f"<Session id={self.id} name={self.metadata.name} messages={self.metadata.message_count}>"
