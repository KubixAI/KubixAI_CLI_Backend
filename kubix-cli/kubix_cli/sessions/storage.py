"""
Session Storage

Handles persistent storage of sessions to disk.
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from .models import Session, SessionMetadata


class SessionStorage:
    """
    Handles session persistence to disk.

    Storage structure:
    .kubix/
    ├── sessions/
    │   ├── sess_*.json         # Individual session files
    │   └── active.json         # Current active session
    └── session-index.json      # Quick lookup index
    """

    def __init__(self, storage_dir: Path):
        """
        Initialize session storage.

        Args:
            storage_dir: Base directory (.kubix/)
        """
        self.storage_dir = Path(storage_dir)
        self.sessions_dir = self.storage_dir / "sessions"
        self.index_file = self.storage_dir / "session-index.json"

        # Create directories
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Load or create index
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        """Load session index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load session index: {e}")

        # Return empty index
        return {
            "sessions": {},
            "last_active": None
        }

    def _save_index(self):
        """Save session index to disk."""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save session index: {e}")

    def _session_file_path(self, session_id: str) -> Path:
        """Get file path for session ID."""
        return self.sessions_dir / f"{session_id}.json"

    def save(self, session: Session) -> bool:
        """
        Save session to disk.

        Args:
            session: Session to save

        Returns:
            True if successful, False otherwise
        """
        try:
            # Save session file
            session_file = self._session_file_path(session.id)
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2)

            # Update index
            self.index["sessions"][session.id] = {
                "name": session.metadata.name,
                "agent": session.metadata.agent_name,
                "created_at": session.metadata.created_at,
                "updated_at": session.metadata.updated_at,
                "message_count": session.metadata.message_count,
                "pinned": session.metadata.pinned,
                "tags": session.tags
            }
            self._save_index()

            return True

        except Exception as e:
            print(f"Error saving session {session.id}: {e}")
            return False

    def load(self, session_id: str) -> Optional[Session]:
        """
        Load session from disk.

        Args:
            session_id: Session ID to load

        Returns:
            Session object or None if not found
        """
        try:
            session_file = self._session_file_path(session_id)

            if not session_file.exists():
                return None

            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            session = Session.from_dict(data)

            # Update last accessed time
            session.metadata.update_timestamp()

            return session

        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None

    def delete(self, session_id: str) -> bool:
        """
        Delete session from disk.

        Args:
            session_id: Session ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete file
            session_file = self._session_file_path(session_id)
            if session_file.exists():
                session_file.unlink()

            # Remove from index
            if session_id in self.index["sessions"]:
                del self.index["sessions"][session_id]
                self._save_index()

            return True

        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")
            return False

    def list_all(self) -> List[str]:
        """
        List all session IDs.

        Returns:
            List of session IDs
        """
        return list(self.index["sessions"].keys())

    def list_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all sessions without loading full sessions.

        Returns:
            Dict of {session_id: metadata_dict}
        """
        return self.index["sessions"]

    def get_by_name(self, name: str) -> Optional[str]:
        """
        Get session ID by name.

        Args:
            name: Session name

        Returns:
            Session ID or None if not found
        """
        for session_id, metadata in self.index["sessions"].items():
            if metadata["name"] == name:
                return session_id
        return None

    def search(self, query: str) -> List[str]:
        """
        Search sessions by name or tags.

        Args:
            query: Search query

        Returns:
            List of matching session IDs
        """
        query_lower = query.lower()
        matching = []

        for session_id, metadata in self.index["sessions"].items():
            # Search in name
            if query_lower in metadata["name"].lower():
                matching.append(session_id)
                continue

            # Search in tags
            if any(query_lower in tag.lower() for tag in metadata.get("tags", [])):
                matching.append(session_id)

        return matching

    def filter_by_agent(self, agent_name: str) -> List[str]:
        """
        Filter sessions by agent name.

        Args:
            agent_name: Agent name

        Returns:
            List of session IDs
        """
        return [
            session_id
            for session_id, metadata in self.index["sessions"].items()
            if metadata["agent"] == agent_name
        ]

    def filter_pinned(self) -> List[str]:
        """
        Get all pinned sessions.

        Returns:
            List of pinned session IDs
        """
        return [
            session_id
            for session_id, metadata in self.index["sessions"].items()
            if metadata.get("pinned", False)
        ]

    def set_active(self, session_id: str):
        """
        Set active session.

        Args:
            session_id: Session ID to mark as active
        """
        self.index["last_active"] = session_id
        self._save_index()

    def get_active(self) -> Optional[str]:
        """
        Get active session ID.

        Returns:
            Active session ID or None
        """
        return self.index.get("last_active")

    def exists(self, session_id: str) -> bool:
        """
        Check if session exists.

        Args:
            session_id: Session ID

        Returns:
            True if exists, False otherwise
        """
        return session_id in self.index["sessions"]

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with stats
        """
        total_sessions = len(self.index["sessions"])
        total_size = 0

        for session_id in self.index["sessions"].keys():
            session_file = self._session_file_path(session_id)
            if session_file.exists():
                total_size += session_file.stat().st_size

        return {
            "total_sessions": total_sessions,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "storage_dir": str(self.storage_dir)
        }
