"""
Session Exporter

Export sessions to various formats (JSON, Markdown, Text, HTML).
"""

import json
from typing import Any
from datetime import datetime

from .models import Session


class SessionExporter:
    """Export sessions to different formats."""

    @staticmethod
    def to_json(session: Session, pretty: bool = True) -> str:
        """
        Export session as JSON.

        Args:
            session: Session to export
            pretty: Pretty print JSON

        Returns:
            JSON string
        """
        indent = 2 if pretty else None
        return json.dumps(session.to_dict(), indent=indent)

    @staticmethod
    def to_markdown(session: Session) -> str:
        """
        Export session as Markdown.

        Args:
            session: Session to export

        Returns:
            Markdown string
        """
        lines = []

        # Header
        lines.append(f"# {session.metadata.name}")
        lines.append("")
        lines.append(f"**Agent:** {session.metadata.agent_name}")
        lines.append(f"**Created:** {session.metadata.created_at}")
        lines.append(f"**Messages:** {session.metadata.message_count}")

        if session.metadata.description:
            lines.append(f"**Description:** {session.metadata.description}")

        if session.tags:
            lines.append(f"**Tags:** {', '.join(f'`{tag}`' for tag in session.tags)}")

        if session.notes:
            lines.append("")
            lines.append("## Notes")
            lines.append("")
            lines.append(session.notes)

        lines.append("")
        lines.append("---")
        lines.append("")

        # Messages
        lines.append("## Conversation")
        lines.append("")

        for i, msg in enumerate(session.messages, 1):
            role = msg.role if hasattr(msg, 'role') else msg.get('role', 'unknown')
            content = msg.content if hasattr(msg, 'content') else msg.get('content', '')

            if role == "system":
                lines.append(f"### System Prompt")
                lines.append("")
                lines.append(f"> {content}")
                lines.append("")
            elif role == "user":
                lines.append(f"### User")
                lines.append("")
                lines.append(content)
                lines.append("")
            elif role == "assistant":
                lines.append(f"### Assistant")
                lines.append("")
                lines.append(content)
                lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"*Exported from Kubix CLI on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(lines)

    @staticmethod
    def to_text(session: Session) -> str:
        """
        Export session as plain text.

        Args:
            session: Session to export

        Returns:
            Plain text string
        """
        lines = []

        # Header
        lines.append("=" * 70)
        lines.append(f"Session: {session.metadata.name}")
        lines.append("=" * 70)
        lines.append(f"Agent: {session.metadata.agent_name}")
        lines.append(f"Created: {session.metadata.created_at}")
        lines.append(f"Messages: {session.metadata.message_count}")

        if session.metadata.description:
            lines.append(f"Description: {session.metadata.description}")

        if session.tags:
            lines.append(f"Tags: {', '.join(session.tags)}")

        if session.notes:
            lines.append("")
            lines.append("Notes:")
            lines.append(session.notes)

        lines.append("=" * 70)
        lines.append("")

        # Messages
        for i, msg in enumerate(session.messages, 1):
            role = msg.role if hasattr(msg, 'role') else msg.get('role', 'unknown')
            content = msg.content if hasattr(msg, 'content') else msg.get('content', '')

            lines.append(f"[{i}] {role.upper()}")
            lines.append("-" * 70)
            lines.append(content)
            lines.append("")

        # Footer
        lines.append("=" * 70)
        lines.append(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)

        return "\n".join(lines)

    @staticmethod
    def to_html(session: Session) -> str:
        """
        Export session as HTML.

        Args:
            session: Session to export

        Returns:
            HTML string
        """
        lines = []

        # HTML header
        lines.append("<!DOCTYPE html>")
        lines.append("<html>")
        lines.append("<head>")
        lines.append("    <meta charset='utf-8'>")
        lines.append(f"    <title>{session.metadata.name} - Kubix CLI</title>")
        lines.append("    <style>")
        lines.append("        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; background: #f5f5f5; }")
        lines.append("        .header { background: #2d3748; color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }")
        lines.append("        .header h1 { margin: 0 0 10px 0; }")
        lines.append("        .metadata { opacity: 0.9; font-size: 14px; }")
        lines.append("        .message { background: white; padding: 20px; margin-bottom: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }")
        lines.append("        .message.system { background: #f7fafc; border-left: 4px solid #4299e1; }")
        lines.append("        .message.user { background: #fff; border-left: 4px solid #48bb78; }")
        lines.append("        .message.assistant { background: #fff; border-left: 4px solid #ed8936; }")
        lines.append("        .role { font-weight: bold; margin-bottom: 10px; text-transform: uppercase; font-size: 12px; color: #718096; }")
        lines.append("        .content { line-height: 1.6; white-space: pre-wrap; }")
        lines.append("        .tag { background: #e2e8f0; padding: 4px 12px; border-radius: 12px; font-size: 12px; margin-right: 5px; }")
        lines.append("        .footer { text-align: center; color: #718096; margin-top: 40px; font-size: 14px; }")
        lines.append("    </style>")
        lines.append("</head>")
        lines.append("<body>")

        # Header
        lines.append("    <div class='header'>")
        lines.append(f"        <h1>{session.metadata.name}</h1>")
        lines.append(f"        <div class='metadata'>")
        lines.append(f"            <strong>Agent:</strong> {session.metadata.agent_name} &nbsp;&nbsp;|&nbsp;&nbsp; ")
        lines.append(f"            <strong>Created:</strong> {session.metadata.created_at} &nbsp;&nbsp;|&nbsp;&nbsp; ")
        lines.append(f"            <strong>Messages:</strong> {session.metadata.message_count}")
        lines.append(f"        </div>")

        if session.tags:
            lines.append(f"        <div style='margin-top: 15px;'>")
            for tag in session.tags:
                lines.append(f"            <span class='tag'>{tag}</span>")
            lines.append(f"        </div>")

        lines.append("    </div>")

        # Messages
        for i, msg in enumerate(session.messages, 1):
            role = msg.role if hasattr(msg, 'role') else msg.get('role', 'unknown')
            content = msg.content if hasattr(msg, 'content') else msg.get('content', '')

            # HTML escape content
            content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            lines.append(f"    <div class='message {role}'>")
            lines.append(f"        <div class='role'>{role}</div>")
            lines.append(f"        <div class='content'>{content}</div>")
            lines.append(f"    </div>")

        # Footer
        lines.append("    <div class='footer'>")
        lines.append(f"        Exported from Kubix CLI on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("    </div>")

        lines.append("</body>")
        lines.append("</html>")

        return "\n".join(lines)

    @staticmethod
    def save_to_file(session: Session, filepath: str, format: str = "auto"):
        """
        Save session to file.

        Args:
            session: Session to export
            filepath: Output file path
            format: Format (auto, json, md, txt, html)
        """
        from pathlib import Path

        filepath = Path(filepath)

        # Auto-detect format from extension
        if format == "auto":
            ext = filepath.suffix.lower()
            if ext == ".json":
                format = "json"
            elif ext in [".md", ".markdown"]:
                format = "md"
            elif ext == ".html":
                format = "html"
            else:
                format = "txt"

        # Export to format
        if format == "json":
            content = SessionExporter.to_json(session)
        elif format == "md":
            content = SessionExporter.to_markdown(session)
        elif format == "html":
            content = SessionExporter.to_html(session)
        else:
            content = SessionExporter.to_text(session)

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
