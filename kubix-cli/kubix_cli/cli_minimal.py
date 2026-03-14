"""
Kubix CLI - Minimal Clean Interface

Clean, fast, beautiful CLI with NVIDIA theme.
"""

import os
import sys
import argparse
from pathlib import Path
import requests

from .agents import AgentOrchestrator
from .ui.theme import console, show_logo, show_success, show_error, show_warning
from .ui.modern_input import get_modern_input, get_minimal_input
from .ui.layout import create_agent_table, create_command_help_table, create_status_panel
from .ui.interactive_help import show_interactive_help, show_options_menu
from .ui.model_selector import show_models, select_model
from .ui.setup_wizard import run_setup_wizard, show_compiler_status
from .openrouter_v2 import OpenRouterClientV2
from .sessions.manager import SessionManager
from rich.table import Table
from rich import box


def list_sessions_interactive(working_dir: Path) -> str:
    """
    Show interactive session list and let user select one.

    Returns:
        Selected session ID or empty string if cancelled
    """
    session_manager = SessionManager(working_dir)
    sessions = session_manager.list_sessions()

    if not sessions:
        show_warning("No saved sessions found")
        return ""

    # Create table
    table = Table(title="📂 Saved Sessions", box=box.ROUNDED, border_style="green")
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="bold green")
    table.add_column("Agent", style="cyan")
    table.add_column("Messages", justify="right", style="yellow")
    table.add_column("Last Modified", style="dim")

    # Sort by modified date (newest first)
    sorted_sessions = sorted(sessions, key=lambda s: s.metadata.modified_at, reverse=True)

    # Add rows
    for idx, session in enumerate(sorted_sessions[:20], 1):  # Show max 20
        name = session.metadata.name[:30]
        agent = session.metadata.agent_name
        msg_count = session.metadata.message_count
        modified = session.metadata.modified_at.strftime("%Y-%m-%d %H:%M")
        table.add_row(str(idx), name, agent, str(msg_count), modified)

    console.print("\n")
    console.print(table)
    console.print("\n[dim]Enter session number (or press Enter to cancel):[/dim]")

    try:
        choice = input("  ▸ ").strip()

        if not choice:
            return ""

        idx = int(choice) - 1
        if 0 <= idx < len(sorted_sessions):
            selected_session = sorted_sessions[idx]
            show_success(f"Loading session: {selected_session.metadata.name}")
            return selected_session.id
        else:
            show_warning("Invalid selection")
            return ""

    except (ValueError, KeyboardInterrupt):
        return ""


def load_latest_session(working_dir: Path) -> str:
    """
    Get the latest session ID.

    Returns:
        Session ID or empty string if none found
    """
    session_manager = SessionManager(working_dir)
    sessions = session_manager.list_sessions()

    if not sessions:
        show_warning("No saved sessions found")
        return ""

    # Sort by modified date
    sorted_sessions = sorted(sessions, key=lambda s: s.metadata.modified_at, reverse=True)
    latest = sorted_sessions[0]

    show_success(f"Continuing session: {latest.metadata.name}")
    console.print(f"[dim]Last modified: {latest.metadata.modified_at.strftime('%Y-%m-%d %H:%M')}[/dim]")
    console.print(f"[dim]Messages: {latest.metadata.message_count}[/dim]\n")

    return latest.id


def detect_gpu_info():
    """Detect GPU and CUDA information."""
    try:
        import subprocess

        # Try nvidia-smi to get GPU name
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split('\n')[0]  # First GPU
            return gpu_name, True
        else:
            return None, False
    except:
        return None, False


def detect_cuda_version():
    """Detect CUDA toolkit version."""
    try:
        import subprocess

        # Try nvcc --version
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode == 0:
            # Extract version from output
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    # Example: "Cuda compilation tools, release 12.0, V12.0.140"
                    parts = line.split('release')
                    if len(parts) > 1:
                        version = parts[1].split(',')[0].strip()
                        return version
            return "Installed"
        return None
    except:
        return None


def show_minimal_banner(orchestrator: AgentOrchestrator):
    """Show clean banner once."""
    console.clear()
    show_logo()

    agent_name = orchestrator.get_current_agent().display_name

    # Detect GPU and CUDA
    gpu_name, gpu_available = detect_gpu_info()
    cuda_version = detect_cuda_version()

    # Build info line
    if gpu_name:
        # Shorten GPU name if too long
        if len(gpu_name) > 30:
            gpu_name = gpu_name[:27] + "..."
        gpu_info = f"[nvidia]{gpu_name}[/nvidia]"
    else:
        gpu_info = f"[nvidia_dim]No GPU detected[/nvidia_dim]"

    if cuda_version:
        cuda_info = f"[success]CUDA {cuda_version}[/success]"
    else:
        cuda_info = f"[nvidia_dim]CUDA not found[/nvidia_dim]"

    console.print(f"\n[nvidia_dim]Agent:[/nvidia_dim] [bold]{agent_name}[/bold]")
    console.print(f"[nvidia_dim]GPU:[/nvidia_dim] {gpu_info}  [nvidia_dim]│[/nvidia_dim]  {cuda_info}")

    # Show API key status hint if needed
    if orchestrator.needs_api_key or orchestrator.api_key == "sk-temp-placeholder":
        console.print("\n[yellow]⚠ No API key configured[/yellow] - Use [green]/setkey[/green] to set up OpenRouter")
    console.print()


def show_gpu_status():
    """Show GPU status information."""
    console.print("\n[nvidia]GPU STATUS[/nvidia]\n")

    try:
        import subprocess

        # Try nvidia-smi for detailed info
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=3
        )

        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            for idx, line in enumerate(lines):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    name, mem_total, mem_used, mem_free, temp, util = parts[:6]

                    console.print(f"[nvidia]GPU {idx}:[/nvidia] {name}")
                    console.print(f"  Memory: {mem_used}/{mem_total} MB ({float(mem_used)/float(mem_total)*100:.1f}% used)")
                    console.print(f"  Temperature: {temp}°C")
                    console.print(f"  Utilization: {util}%")
                    console.print()
        else:
            # Fallback to basic detection
            gpu_name, gpu_available = detect_gpu_info()
            cuda_version = detect_cuda_version()

            if gpu_name:
                console.print(f"[nvidia]GPU:[/nvidia] {gpu_name}")
            else:
                console.print("[nvidia_dim]No GPU detected[/nvidia_dim]")

            if cuda_version:
                console.print(f"[nvidia]CUDA:[/nvidia] Version {cuda_version}")
            else:
                console.print("[nvidia_dim]CUDA not found[/nvidia_dim]")
            console.print()

    except Exception as e:
        console.print("[yellow]Could not retrieve GPU status[/yellow]")
        console.print(f"[dim]Error: {str(e)[:100]}[/dim]\n")


def handle_command(orchestrator: AgentOrchestrator, cmd: str) -> bool:
    """
    Handle slash commands (simplified to essential commands only).

    Returns:
        True to continue, False to exit
    """
    parts = cmd[1:].split()
    if not parts:
        return True

    command = parts[0].lower()

    # Essential Commands Only

    # Model selection
    if command == "models":
        if len(parts) < 2:
            # Show models if no argument provided
            try:
                current_agent = orchestrator.get_current_agent()
                show_models(current_agent.model, OpenRouterClientV2.MODEL_CATALOG)
            except Exception as e:
                # Handle any error in showing models
                import re
                error_msg = str(e)[:200]
                # Remove all rich markup tags
                error_msg = re.sub(r'\[.*?\]', '', error_msg)
                show_error(f"Failed to display models: {error_msg}")
                console.print("[dim]Try /setkey to update your API key[/dim]\n")
        else:
            choice = parts[1]
            selected = select_model(choice, OpenRouterClientV2.MODEL_CATALOG, OpenRouterClientV2.DEFAULT_MODEL)

            if selected:
                # Check if model is free or premium
                is_free_model = selected.endswith(":free")

                # Check if we need an API key (all models require one)
                current_api_key = orchestrator.cache_manager.get_api_key()

                if not current_api_key or current_api_key == "sk-temp-placeholder":
                    # Need to get API key from user
                    console.print("\n[nvidia]API Key Required[/nvidia]")
                    console.print("All models (including free ones) require an OpenRouter API key.\n")
                    console.print("[dim]Get your FREE API key from: https://openrouter.ai[/dim]")
                    console.print("[dim](Sign up with Google/GitHub for instant access)[/dim]\n")

                    from rich.prompt import Prompt
                    try:
                        api_key = Prompt.ask("[nvidia]Paste your API key[/nvidia]", password=True)

                        if api_key and api_key.strip():
                            # Save the API key
                            orchestrator.cache_manager.save_api_key(api_key.strip())
                            orchestrator.api_key = api_key.strip()
                            orchestrator.needs_api_key = False

                            # Update all agents with new API key
                            for agent in orchestrator.agents.values():
                                agent.api_key = api_key.strip()
                                agent.client.api_key = api_key.strip()

                            show_success("API key saved! You can now use all models.")
                        else:
                            show_warning("No API key provided. Get one free at: https://openrouter.ai")
                            return True  # Return to main loop
                    except (KeyboardInterrupt, EOFError):
                        show_warning("Setup cancelled")
                        return True  # Return to main loop

                # Update current agent's model
                current_agent = orchestrator.get_current_agent()
                current_agent.model = selected
                current_agent.client.model = selected

                model_info = OpenRouterClientV2.MODEL_CATALOG[selected]
                if selected == OpenRouterClientV2.DEFAULT_MODEL:
                    console.print(f"[green]✓ Reset to default: [bold]{model_info['name']}[/bold][/green]\n")
                else:
                    price_info = ""
                    if is_free_model:
                        price_info = " (FREE)"
                    console.print(f"[green]✓ Switched to: [bold]{model_info['name']}[/bold]{price_info}[/green]\n")
            else:
                show_warning(f"Invalid choice: {choice}")

    # GPU Status
    elif command == "gpu":
        show_gpu_status()

    # Compiler Setup
    elif command == "setup":
        try:
            run_setup_wizard(orchestrator.config_manager)
        except Exception as e:
            import re
            error_msg = str(e)[:200]
            error_msg = re.sub(r'\[.*?\]', '', error_msg)
            show_error(f"Setup failed: {error_msg}")
            console.print()

    # Show Compiler Status
    elif command in ["compilers", "compiler"]:
        try:
            show_compiler_status(orchestrator.config_manager)
        except Exception as e:
            import re
            error_msg = str(e)[:200]
            error_msg = re.sub(r'\[.*?\]', '', error_msg)
            show_error(f"Status check failed: {error_msg}")
            console.print()

    # Set API Key
    elif command == "setkey":
        console.print("\n[nvidia]Update OpenRouter API Key[/nvidia]")
        console.print("Get your FREE API key from: [cyan]https://openrouter.ai[/cyan]")
        console.print("[dim](Sign up with Google/GitHub for instant access)[/dim]\n")

        from rich.prompt import Prompt
        try:
            api_key = Prompt.ask("[nvidia]Enter your new API key[/nvidia]", password=True)

            if api_key and api_key.strip():
                # Save the API key
                orchestrator.cache_manager.save_api_key(api_key.strip())
                orchestrator.api_key = api_key.strip()
                orchestrator.needs_api_key = False

                # Update all agents with new API key
                for agent in orchestrator.agents.values():
                    agent.api_key = api_key.strip()
                    agent.client.api_key = api_key.strip()

                show_success("API key updated successfully! You can now use all models.")
            else:
                show_warning("No API key provided. Get one free at: https://openrouter.ai")
        except (KeyboardInterrupt, EOFError):
            show_warning("Operation cancelled")

    # Clear conversation
    elif command == "clear":
        orchestrator.get_current_agent().clear_conversation()
        show_minimal_banner(orchestrator)
        show_success("Conversation cleared")

    # Help
    elif command == "help":
        # Show simplified help
        console.print("\n[nvidia]AVAILABLE COMMANDS[/nvidia]\n")
        console.print("  [green]/models[/green]     - List or switch AI models")
        console.print("  [green]/setkey[/green]     - Set or update OpenRouter API key")
        console.print("  [green]/gpu[/green]        - Show GPU status")
        console.print("  [green]/setup[/green]      - Configure compiler paths (nvcc, cl.exe, etc.)")
        console.print("  [green]/compilers[/green]  - Show current compiler configuration")
        console.print("  [green]/clear[/green]      - Clear conversation history")
        console.print("  [green]/help[/green]       - Show this help menu")
        console.print("  [green]/quit[/green]       - Exit Kubix CLI")
        console.print()

    # Exit
    elif command in ["quit", "exit", "q"]:
        console.print("\n[nvidia_dim]Goodbye! 👋[/nvidia_dim]\n")
        return False

    else:
        show_warning(f"Unknown command: /{command}")
        console.print("[dim]Type /help for available commands[/dim]\n")

    return True


def show_help():
    """Show command help."""
    console.print()
    table = create_command_help_table()
    console.print(table)
    console.print()


def show_agents(orchestrator: AgentOrchestrator):
    """Show agents list."""
    console.print()
    table = create_agent_table(orchestrator.agents, orchestrator.current_agent)
    console.print(table)
    console.print()


def show_status(orchestrator: AgentOrchestrator):
    """Show system status."""
    console.print()
    panel = create_status_panel(
        agent=orchestrator.get_current_agent().display_name,
        model=orchestrator.get_current_agent().model,
        routing=orchestrator.auto_routing,
        working_dir=str(orchestrator.working_dir)
    )
    console.print(panel)
    console.print()


def save_session(orchestrator: AgentOrchestrator, parts: list):
    """Save session."""
    from .ui.theme import show_session_saved

    name = parts[1] if len(parts) > 1 else None
    current_agent = orchestrator.get_current_agent()

    if not orchestrator.session_manager.current_session:
        orchestrator.session_manager.create_session(name=name, messages=current_agent.conversation)

    if orchestrator.session_manager.save_current(name=name):
        show_session_saved(orchestrator.session_manager.current_session.metadata.name)
    else:
        show_warning("Save failed")


def load_session(orchestrator: AgentOrchestrator, parts: list):
    """Load session."""
    from .ui.theme import show_session_loaded

    if len(parts) < 2:
        show_warning("Usage: /load <name>")
        return

    session = orchestrator.session_manager.load_session(parts[1])
    if session:
        orchestrator.get_current_agent().conversation = session.messages
        show_session_loaded(session.metadata.name, session.metadata.message_count)
    else:
        show_warning("Session not found")


def export_session(orchestrator: AgentOrchestrator, parts: list):
    """Export session."""
    if len(parts) < 2:
        show_warning("Usage: /export <file>")
        return

    if orchestrator.session_manager.export_session(parts[1]):
        show_success(f"Exported: {parts[1]}")
    else:
        show_warning("Export failed")


def fork_session(orchestrator: AgentOrchestrator, parts: list):
    """Fork session."""
    if len(parts) < 2:
        show_warning("Usage: /fork <name>")
        return

    forked = orchestrator.session_manager.fork_session(parts[1])
    if forked:
        show_success(f"Forked: {parts[1]}")
    else:
        show_warning("Fork failed")


def show_exit_confirmation() -> bool:
    """
    Show exit confirmation dialog.

    Returns:
        True to exit, False to continue
    """
    from prompt_toolkit import prompt
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.styles import Style

    style = Style.from_dict({
        'prompt': '#76B900 bold',
    })

    console.print("[yellow]⚠️  Exit Kubix CLI?[/yellow]")
    console.print()

    while True:
        try:
            # Simple y/n prompt
            answer = prompt(
                HTML('<prompt>  Exit? (y/n): </prompt>'),
                style=style
            ).strip().lower()

            if answer in ['y', 'yes']:
                return True
            elif answer in ['n', 'no', '']:
                return False
            else:
                console.print("[dim]Please enter 'y' for yes or 'n' for no[/dim]")

        except (KeyboardInterrupt, EOFError):
            # Second Ctrl+C or EOF = force exit
            return True


def interactive_mode(orchestrator: AgentOrchestrator):
    """
    Production-ready interactive loop with comprehensive error handling.
    Rock-solid stability for cross-platform deployment.
    """
    # Show banner once (only if not already shown)
    try:
        show_minimal_banner(orchestrator)
    except Exception as e:
        # Banner failed, but continue
        pass  # Silent fail - banner is not critical

    # Error recovery counters
    consecutive_errors = 0
    max_consecutive_errors = 5
    total_errors = 0
    max_total_errors = 50

    # Main loop with recovery
    while True:
        try:
            # Ensure clean terminal state before input
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except:
                pass

            # Get input with fallback
            try:
                user_input = get_modern_input()  # Use modern input with autocomplete
            except (KeyboardInterrupt, EOFError):
                raise  # Re-raise these for proper handling
            except Exception as input_error:
                # Input failed - try fallback to minimal input (no autocomplete)
                console.print("[yellow]Autocomplete unavailable, using basic input...[/yellow]")
                try:
                    user_input = get_minimal_input()
                except (KeyboardInterrupt, EOFError):
                    raise
                except:
                    # Ultimate fallback - plain input
                    try:
                        user_input = input("  > ").strip()
                    except (KeyboardInterrupt, EOFError):
                        raise
                    except:
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            console.print("[red]Too many consecutive errors. Restarting...[/red]")
                            consecutive_errors = 0
                        continue

            # Reset error counter on successful input
            if user_input:
                consecutive_errors = 0

            # Skip empty input
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith('/'):
                try:
                    should_continue = handle_command(orchestrator, user_input)
                    if not should_continue:
                        break
                except Exception as cmd_error:
                    # Strip any rich formatting from error message
                    import re
                    error_msg = str(cmd_error)[:100]
                    # Remove all rich markup tags
                    error_msg = re.sub(r'\[.*?\]', '', error_msg)
                    show_error(f"Command error: {error_msg}")
                    console.print()
                    total_errors += 1
            else:
                # Process query with comprehensive error handling
                try:
                    orchestrator.chat(user_input)
                except KeyboardInterrupt:
                    # User interrupted during processing
                    console.print("\n[yellow]Processing interrupted[/yellow]\n")
                except requests.exceptions.RequestException as req_error:
                    # Network errors
                    show_error(f"Network error: {str(req_error)[:100]}")
                    console.print("[dim]Check your internet connection and try again[/dim]\n")
                    total_errors += 1
                except ValueError as val_error:
                    # Validation errors (e.g., API key issues)
                    # Clean the error message to avoid markup issues
                    error_msg = str(val_error)[:100]
                    # Remove any markup tags from error message
                    error_msg = error_msg.replace('[', '').replace(']', '')
                    show_error(f"Configuration error: {error_msg}")

                    # If it's an API key error, offer to re-enter it immediately
                    if "API key" in str(val_error):
                        console.print("\n[yellow]Your API key appears to be invalid or expired.[/yellow]")
                        console.print("[dim]Use /setkey to update it or answer below:[/dim]\n")

                        from rich.prompt import Prompt
                        try:
                            choice = Prompt.ask("[nvidia]Enter new key? (y/n)[/nvidia]", choices=["y", "n"], default="y")

                            if choice.lower() == "y":
                                console.print("\nGet your FREE API key from: [cyan]https://openrouter.ai[/cyan]")
                                console.print("[dim](Sign up with Google/GitHub for instant access)[/dim]\n")

                                api_key = Prompt.ask("[nvidia]Paste your API key[/nvidia]", password=True)

                                if api_key and api_key.strip():
                                    # Save the API key
                                    orchestrator.cache_manager.save_api_key(api_key.strip())
                                    orchestrator.api_key = api_key.strip()
                                    orchestrator.needs_api_key = False

                                    # Update all agents with new API key
                                    for agent in orchestrator.agents.values():
                                        agent.api_key = api_key.strip()
                                        agent.client.api_key = api_key.strip()

                                    show_success("API key updated! Please try your request again.")
                                    consecutive_errors = 0  # Reset error counter since we fixed the issue
                                else:
                                    console.print("[dim]You can set your API key later with /setkey[/dim]\n")
                            else:
                                console.print("[dim]You can set your API key later with /setkey[/dim]\n")
                        except (KeyboardInterrupt, EOFError):
                            console.print("[dim]You can set your API key later with /setkey[/dim]\n")
                    total_errors += 1
                except Exception as e:
                    # General errors
                    error_msg = str(e)[:200] if str(e) else "Unknown error"
                    show_error(f"Processing error: {error_msg}")
                    console.print()
                    total_errors += 1

                    # Check for critical errors
                    if total_errors >= max_total_errors:
                        console.print("[red]Too many errors. Please restart the application.[/red]")
                        break

        except KeyboardInterrupt:
            # Show exit confirmation with error handling
            try:
                console.print("\n")
                if show_exit_confirmation():
                    console.print("\n[nvidia_dim]Goodbye! 👋[/nvidia_dim]\n")
                    break
                else:
                    console.print("[dim]Continuing...[/dim]\n")
            except:
                # Exit confirmation failed - just exit
                console.print("\n[nvidia_dim]Goodbye! 👋[/nvidia_dim]\n")
                break

        except EOFError:
            # Clean exit on EOF
            console.print("\n\n[nvidia_dim]Goodbye! 👋[/nvidia_dim]\n")
            break

        except Exception as e:
            # Catch-all error handler
            consecutive_errors += 1
            total_errors += 1

            # Log error
            try:
                error_msg = str(e)[:200] if str(e) else "Unknown error"
                show_error(f"Unexpected error: {error_msg}")
                console.print()
            except:
                print(f"Error: {e}")

            # Check if we should continue
            if consecutive_errors >= max_consecutive_errors:
                console.print("[red]Too many consecutive errors. Exiting...[/red]")
                break

            if total_errors >= max_total_errors:
                console.print("[red]Too many total errors. Exiting...[/red]")
                break

    # Cleanup on exit
    try:
        # Import cleanup function
        from .ui.thinking_spinner import cleanup_global_spinner
        cleanup_global_spinner()
    except:
        pass


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Kubix CLI - CUDA AI Assistant")

    parser.add_argument("--working-dir", type=Path, default=None, help="Working directory")
    parser.add_argument("--model", type=str, default=None, help="Override model")
    parser.add_argument("--agent", type=str, choices=["optimizer", "debugger", "analyzer", "general"], default="general", help="Starting agent")
    parser.add_argument("--no-routing", action="store_true", help="Disable auto-routing")
    parser.add_argument("--continue", dest="continue_session", action="store_true", help="Continue latest session")
    parser.add_argument("--resume", action="store_true", help="Show list of sessions to resume from")

    args = parser.parse_args()

    # Working directory
    working_dir = args.working_dir or Path.cwd()

    # Handle session resumption
    session_id_to_load = None

    if args.continue_session:
        # Load latest session
        session_id_to_load = load_latest_session(working_dir)
        if not session_id_to_load:
            console.print("[yellow]Starting new session instead...[/yellow]\n")

    elif args.resume:
        # Interactive session selection
        session_id_to_load = list_sessions_interactive(working_dir)
        if not session_id_to_load:
            console.print("[yellow]Starting new session instead...[/yellow]\n")

    # Config
    config_overrides = {}
    if args.model:
        config_overrides["model"] = args.model

    # Initialize
    try:
        orchestrator = AgentOrchestrator(working_dir=working_dir, config_overrides=config_overrides)
    except Exception as e:
        show_error(f"Init failed: {e}")
        sys.exit(1)

    # Restore session if requested
    if session_id_to_load:
        if not orchestrator.restore_session(session_id_to_load):
            show_warning("Failed to restore session, starting fresh")
    else:
        # Setup fresh session
        if args.agent != "general":
            orchestrator.switch_agent(args.agent)

        if args.no_routing:
            orchestrator.toggle_auto_routing()

    # Run
    interactive_mode(orchestrator)


if __name__ == "__main__":
    main()
