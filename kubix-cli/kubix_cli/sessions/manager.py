"""
Session Manager

High-level interface for session management operations.
"""

import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from .models import Session, SessionMetadata
from .storage import SessionStorage
from .exporter import SessionExporter

console = Console()


class SessionManager:
    """
    Manages session lifecycle and operations.

    Provides high-level interface for:
    - Creating and managing sessions
    - Saving and loading sessions
    - Forking sessions
    - Session cleanup
    - Auto-save
    """

    def __init__(self, working_dir: Path, agent_name: str = "general"):
        """
        Initialize session manager.

        Args:
            working_dir: Working directory
            agent_name: Current agent name
        """
        self.working_dir = Path(working_dir)
        self.agent_name = agent_name

        # Storage
        config_dir = self.working_dir / ".kubix"
        self.storage = SessionStorage(config_dir)

        # Current session
        self.current_session: Optional[Session] = None

        # Auto-save
        self.auto_save_enabled = False
        self.auto_save_interval = 60  # seconds
        self.auto_save_thread: Optional[threading.Thread] = None
        self._stop_auto_save = threading.Event()

    def create_session(
        self,
        name: Optional[str] = None,
        messages: Optional[List[Any]] = None,
        description: str = ""
    ) -> Session:
        """
        Create new session.

        Args:
            name: Session name (auto-generated if None)
            messages: Initial messages
            description: Optional description

        Returns:
            New session
        """
        # Generate ID
        session_id = Session.generate_id()

        # Auto-generate name if not provided
        if not name:
            name = self._generate_name(messages)

        # Create metadata
        metadata = SessionMetadata(
            name=name,
            description=description,
            agent_name=self.agent_name,
            working_dir=str(self.working_dir)
        )

        # Create session
        session = Session(
            id=session_id,
            metadata=metadata,
            messages=messages or []
        )

        # Update metadata from messages
        if messages:
            for msg in messages:
                session.add_message(msg)

        self.current_session = session
        return session

    def _generate_name(self, messages: Optional[List[Any]]) -> str:
        """Generate session name from first user message or timestamp."""
        if messages:
            # Find first user message
            for msg in messages:
                role = msg.role if hasattr(msg, 'role') else msg.get('role')
                if role == 'user':
                    content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                    # Take first 30 chars and slugify
                    words = content[:50].split()[:5]
                    name = '-'.join(words).lower()
                    # Remove special chars
                    name = ''.join(c if c.isalnum() or c == '-' else '' for c in name)
                    return name[:40] or "untitled"

        # Fallback: timestamp
        return datetime.now().strftime('session-%Y%m%d-%H%M%S')

    def save_current(self, name: Optional[str] = None) -> bool:
        """
        Save current session.

        Args:
            name: Optional new name for session

        Returns:
            True if successful
        """
        if not self.current_session:
            console.print("[yellow]No active session to save[/yellow]")
            return False

        # Update name if provided
        if name:
            self.current_session.metadata.name = name

        # Update timestamp
        self.current_session.metadata.update_timestamp()

        # Save to storage
        success = self.storage.save(self.current_session)

        if success:
            self.storage.set_active(self.current_session.id)
            return True

        return False

    def load_session(self, identifier: str) -> Optional[Session]:
        """
        Load session by ID or name.

        Args:
            identifier: Session ID or name

        Returns:
            Loaded session or None
        """
        # Try as session ID first
        session = self.storage.load(identifier)

        if not session:
            # Try as name
            session_id = self.storage.get_by_name(identifier)
            if session_id:
                session = self.storage.load(session_id)

        if session:
            self.current_session = session
            self.storage.set_active(session.id)
            return session

        return None

    def delete_session(self, identifier: str) -> bool:
        """
        Delete session by ID or name.

        Args:
            identifier: Session ID or name

        Returns:
            True if successful
        """
        # Try as session ID first
        if self.storage.exists(identifier):
            return self.storage.delete(identifier)

        # Try as name
        session_id = self.storage.get_by_name(identifier)
        if session_id:
            return self.storage.delete(session_id)

        return False

    def list_sessions(
        self,
        pinned_only: bool = False,
        agent_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all sessions.

        Args:
            pinned_only: Show only pinned sessions
            agent_filter: Filter by agent name

        Returns:
            List of session metadata dicts
        """
        all_metadata = self.storage.list_all_metadata()

        results = []
        for session_id, metadata in all_metadata.items():
            # Apply filters
            if pinned_only and not metadata.get("pinned", False):
                continue

            if agent_filter and metadata.get("agent") != agent_filter:
                continue

            results.append({
                "id": session_id,
                **metadata
            })

        # Sort by updated_at (most recent first)
        results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        return results

    def display_sessions(
        self,
        pinned_only: bool = False,
        agent_filter: Optional[str] = None
    ):
        """
        Display sessions in a nice table.

        Args:
            pinned_only: Show only pinned sessions
            agent_filter: Filter by agent name
        """
        sessions = self.list_sessions(pinned_only=pinned_only, agent_filter=agent_filter)

        if not sessions:
            console.print("\n[dim]No sessions found[/dim]\n")
            return

        # Separate pinned and regular
        pinned = [s for s in sessions if s.get("pinned", False)]
        regular = [s for s in sessions if not s.get("pinned", False)]

        console.print(f"\n[bold cyan]📁 Your Sessions[/bold cyan] ([dim]{len(sessions)} total[/dim])\n")

        # Show pinned first
        if pinned and not pinned_only:
            console.print("[bold]📌 Pinned:[/bold]\n")
            self._display_session_table(pinned)
            console.print()

        # Show regular
        if regular and not pinned_only:
            if pinned:
                console.print("[bold]Recent:[/bold]\n")
            self._display_session_table(regular[:10])  # Show 10 most recent

            if len(regular) > 10:
                console.print(f"\n[dim]...and {len(regular) - 10} more. Use /search to find specific sessions.[/dim]")

        console.print()

    def _display_session_table(self, sessions: List[Dict[str, Any]]):
        """Display sessions as a table."""
        table = Table(show_header=True, box=box.SIMPLE, padding=(0, 2))
        table.add_column("Name", style="cyan", width=30)
        table.add_column("Agent", style="yellow", width=12)
        table.add_column("Messages", style="blue", width=10)
        table.add_column("Updated", style="dim", width=15)
        table.add_column("Tags", style="green")

        for session in sessions:
            # Format relative time
            updated = session.get("updated_at", "")
            try:
                dt = datetime.fromisoformat(updated)
                now = datetime.now()
                delta = now - dt

                if delta.days > 0:
                    time_str = f"{delta.days}d ago"
                elif delta.seconds > 3600:
                    time_str = f"{delta.seconds // 3600}h ago"
                elif delta.seconds > 60:
                    time_str = f"{delta.seconds // 60}m ago"
                else:
                    time_str = "just now"
            except:
                time_str = "unknown"

            # Format tags
            tags = session.get("tags", [])
            tags_str = " ".join(f"#{tag}" for tag in tags[:3])
            if len(tags) > 3:
                tags_str += f" +{len(tags) - 3}"

            table.add_row(
                session.get("name", "untitled")[:28],
                session.get("agent", "unknown"),
                str(session.get("message_count", 0)),
                time_str,
                tags_str
            )

        console.print(table)

    def display_session_info(self, identifier: Optional[str] = None):
        """
        Display detailed session info.

        Args:
            identifier: Session ID or name (current session if None)
        """
        if identifier:
            session = self.load_session(identifier)
        else:
            session = self.current_session

        if not session:
            console.print("[yellow]No session found[/yellow]")
            return

        # Create info panel
        info_lines = []
        info_lines.append(f"[bold]ID:[/bold] {session.id}")
        info_lines.append(f"[bold]Agent:[/bold] {session.metadata.agent_name}")
        info_lines.append(f"[bold]Created:[/bold] {session.metadata.created_at}")
        info_lines.append(f"[bold]Updated:[/bold] {session.metadata.updated_at}")
        info_lines.append(f"[bold]Working Dir:[/bold] {session.metadata.working_dir}")
        info_lines.append("")
        info_lines.append(f"[bold]Messages:[/bold] {session.metadata.message_count} ({session.metadata.user_messages} user, {session.metadata.assistant_messages} assistant)")
        info_lines.append(f"[bold]Tokens (est):[/bold] ~{session.metadata.total_tokens:,}")

        if session.tags:
            info_lines.append(f"[bold]Tags:[/bold] {', '.join(f'#{tag}' for tag in session.tags)}")

        info_lines.append(f"[bold]Pinned:[/bold] {'Yes' if session.metadata.pinned else 'No'}")

        if session.notes:
            info_lines.append("")
            info_lines.append(f"[bold]Notes:[/bold]")
            info_lines.append(session.notes)

        panel = Panel(
            "\n".join(info_lines),
            title=f"📄 Session: {session.metadata.name}",
            border_style="cyan"
        )

        console.print()
        console.print(panel)
        console.print()

    def rename_session(self, identifier: str, new_name: str) -> bool:
        """
        Rename session.

        Args:
            identifier: Session ID or name
            new_name: New name

        Returns:
            True if successful
        """
        session = self.load_session(identifier)
        if not session:
            return False

        session.metadata.name = new_name
        return self.storage.save(session)

    def fork_session(
        self,
        new_name: str,
        fork_point: Optional[int] = None
    ) -> Optional[Session]:
        """
        Fork current session.

        Args:
            new_name: Name for forked session
            fork_point: Message index to fork from (None = current point)

        Returns:
            New forked session or None
        """
        if not self.current_session:
            console.print("[yellow]No active session to fork[/yellow]")
            return None

        # Determine fork point
        if fork_point is None:
            fork_point = len(self.current_session.messages) - 1

        # Get messages up to fork point
        forked_messages = self.current_session.get_messages_from_point(fork_point)

        # Create new session
        new_session = self.create_session(
            name=new_name,
            messages=forked_messages,
            description=f"Forked from {self.current_session.metadata.name}"
        )

        # Set fork metadata
        new_session.parent_id = self.current_session.id
        new_session.fork_point = fork_point

        # Save forked session
        self.storage.save(new_session)

        return new_session

    def export_session(
        self,
        filepath: str,
        identifier: Optional[str] = None,
        format: str = "auto"
    ) -> bool:
        """
        Export session to file.

        Args:
            filepath: Output file path
            identifier: Session ID or name (current if None)
            format: Format (auto, json, md, txt, html)

        Returns:
            True if successful
        """
        if identifier:
            session = self.load_session(identifier)
        else:
            session = self.current_session

        if not session:
            console.print("[yellow]No session to export[/yellow]")
            return False

        try:
            SessionExporter.save_to_file(session, filepath, format)
            return True
        except Exception as e:
            console.print(f"[red]Export failed: {e}[/red]")
            return False

    # Auto-save functionality

    def enable_auto_save(self, interval: int = 60):
        """
        Enable auto-save.

        Args:
            interval: Save interval in seconds
        """
        if self.auto_save_enabled:
            return

        self.auto_save_enabled = True
        self.auto_save_interval = interval
        self._stop_auto_save.clear()

        # Start background thread
        self.auto_save_thread = threading.Thread(
            target=self._auto_save_worker,
            daemon=True
        )
        self.auto_save_thread.start()

        console.print(f"[green]✓ Auto-save enabled (every {interval}s)[/green]")

    def disable_auto_save(self):
        """Disable auto-save."""
        if not self.auto_save_enabled:
            return

        self.auto_save_enabled = False
        self._stop_auto_save.set()

        if self.auto_save_thread:
            self.auto_save_thread.join(timeout=2)

        console.print("[dim]Auto-save disabled[/dim]")

    def _auto_save_worker(self):
        """Background worker for auto-save."""
        while not self._stop_auto_save.wait(self.auto_save_interval):
            if self.current_session:
                self.save_current()

    def cleanup_old_sessions(self, days: int = 30) -> int:
        """
        Delete sessions older than N days (unless pinned).

        Args:
            days: Age threshold in days

        Returns:
            Number of sessions deleted
        """
        cutoff = datetime.now() - timedelta(days=days)
        deleted = 0

        for session_id in self.storage.list_all():
            # Load metadata only
            metadata_dict = self.storage.index["sessions"].get(session_id)
            if not metadata_dict:
                continue

            # Skip pinned
            if metadata_dict.get("pinned", False):
                continue

            # Check age
            try:
                last_accessed = datetime.fromisoformat(metadata_dict.get("updated_at", ""))
                if last_accessed < cutoff:
                    self.storage.delete(session_id)
                    deleted += 1
            except:
                pass

        return deleted

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return self.storage.get_storage_stats()
