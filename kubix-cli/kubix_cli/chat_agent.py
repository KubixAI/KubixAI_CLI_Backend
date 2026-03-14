"""
Kubix CLI - Agentic AI Chat Interface

Claude Code-style conversational AI for CUDA development.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re
import requests
import shutil

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import box
from rich.prompt import Prompt
from rich.live import Live
from rich.spinner import Spinner

console = Console()

# Top 5 OpenRouter models for CUDA work
MODELS = {
    "1": {
        "name": "GPT-4 Turbo",
        "id": "openai/gpt-4-turbo-preview",
        "description": "Best for complex optimizations and deep explanations"
    },
    "2": {
        "name": "Claude 3.5 Sonnet",
        "id": "anthropic/claude-3.5-sonnet",
        "description": "Excellent at code analysis and refactoring (recommended)"
    },
    "3": {
        "name": "GPT-4o",
        "id": "openai/gpt-4o",
        "description": "Fast, accurate, great for benchmarking analysis"
    },
    "4": {
        "name": "Claude 3 Opus",
        "id": "anthropic/claude-3-opus",
        "description": "Most capable, best for difficult problems"
    },
    "5": {
        "name": "GPT-3.5 Turbo",
        "id": "openai/gpt-3.5-turbo",
        "description": "Fast and economical for quick tasks"
    }
}


class CUDAChatAgent:
    """Agentic AI chat interface for CUDA development."""

    def __init__(self):
        self.current_dir = Path.cwd()
        self.conversation = []
        self.compact = True  # Default compact mode

        # Set default model (Claude 3.5 Sonnet - best for code)
        self.current_model = MODELS["2"]

        # Import CUDA tools
        from .cache import CacheManager
        from .kernel_analyzer import KernelAnalyzer
        from .compiler import CUDACompiler
        from .bench import Benchmarker
        from .profiler import CUDAProfiler

        self.cache_manager = CacheManager()
        self.analyzer = KernelAnalyzer()
        self.compiler = CUDACompiler()
        self.benchmarker = Benchmarker()
        self.profiler = CUDAProfiler()

        # Check API key
        self.api_key = self.cache_manager.get_api_key()
        if not self.api_key:
            console.print("\n[yellow]⚠️  No API key found[/yellow]")
            self._setup_api_key()
            self.api_key = self.cache_manager.get_api_key()
            if not self.api_key:
                console.print("[red]Cannot continue without API key[/red]")
                sys.exit(1)

    def start(self):
        """Start chat interface - Claude Code style."""
        # Show banner
        self._show_banner()

        # Show status with current model
        console.print(f"\n[dim]Using {self.current_model['name']} • /help for commands • /model to switch • Ctrl+C to exit[/dim]\n")

        # Chat loop
        while True:
            try:
                # Show ASCII input box
                self._show_input_box()
                user_input = input().strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                else:
                    # Natural language chat
                    self._handle_chat(user_input)

            except KeyboardInterrupt:
                console.print("\n\n[dim]👋 Goodbye![/dim]\n")
                break
            except EOFError:
                console.print("\n\n[dim]👋 Goodbye![/dim]\n")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]\n")
                import traceback
                traceback.print_exc()

    def _show_banner(self):
        """Show ASCII banner."""
        banner = """
    ██████╗ ██╗ ██████╗ ██╗  ██╗████████╗███╗   ██╗ ██████╗ ██╗    ██╗
    ██╔══██╗██║██╔════╝ ██║  ██║╚══██╔══╝████╗  ██║██╔═══██╗██║    ██║
    ██████╔╝██║██║  ███╗███████║   ██║   ██╔██╗ ██║██║   ██║██║ █╗ ██║
    ██╔══██╗██║██║   ██║██╔══██║   ██║   ██║╚██╗██║██║   ██║██║███╗██║
    ██║  ██║██║╚██████╔╝██║  ██║   ██║   ██║ ╚████║╚██████╔╝╚███╔███╔╝
    ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═══╝ ╚═════╝  ╚══╝╚══╝
        """
        console.print(f"[bold cyan]{banner}[/bold cyan]")
        console.print("[bold]CUDA AI Agent[/bold] • Agentic AI for GPU Development")
        console.print(f"[dim]📂 {self.current_dir}[/dim]")

    def _get_terminal_width(self):
        """Get current terminal width."""
        try:
            return shutil.get_terminal_size().columns
        except:
            return 80  # Default fallback

    def _show_input_box(self):
        """Show dynamic ASCII input box that adapts to terminal width."""
        width = self._get_terminal_width()
        # Make box 2 chars less than terminal width for padding
        box_width = max(40, width - 2)

        # Create top border
        top_border = "┌" + "─" * (box_width - 2) + "┐"
        # Bottom border will be shown after input

        console.print(f"\n[dim]{top_border}[/dim]")
        console.print("[dim]│[/dim] [cyan]>[/cyan] ", end="")

    def select_model(self):
        """Let user select AI model."""
        console.print("\n[bold cyan]Select AI Model:[/bold cyan]\n")

        for key, model in MODELS.items():
            console.print(f"  [bold]{key}.[/bold] {model['name']}")
            console.print(f"      [dim]{model['description']}[/dim]\n")

        choice = Prompt.ask("Select model [1/2/3/4/5]", choices=list(MODELS.keys()), default="2")
        self.current_model = MODELS[choice]

        console.print(f"\n[green]✓ Using: {self.current_model['name']}[/green]")

    def _handle_command(self, cmd: str):
        """Handle slash commands."""
        # Close the input box dynamically
        width = self._get_terminal_width()
        box_width = max(40, width - 2)
        bottom_border = "└" + "─" * (box_width - 2) + "┘"

        console.print("[dim]│[/dim]")
        console.print(f"[dim]{bottom_border}[/dim]")

        parts = cmd[1:].split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if command == "help":
            self._cmd_help()
        elif command == "exit" or command == "quit":
            console.print("\n[dim]👋 Goodbye![/dim]\n")
            sys.exit(0)
        elif command == "compact":
            self.compact = not self.compact
            status = "ON" if self.compact else "OFF"
            console.print(f"\n[green]✓ Compact mode: {status}[/green]\n")
        elif command == "model":
            console.print()
            self.select_model()
            console.print()
        elif command == "clear":
            self.conversation = []
            os.system('cls' if os.name == 'nt' else 'clear')
            self._show_banner()
            console.print(f"\n[green]✓ Conversation cleared[/green]")
            console.print(f"[dim]Using {self.current_model['name']} • /help for commands • /model to switch[/dim]\n")
        elif command == "list":
            self._cmd_list()
        elif command == "info":
            self._cmd_info()
        else:
            console.print(f"\n[yellow]Unknown command: /{command}[/yellow]")
            console.print("[dim]Type /help for available commands[/dim]\n")

    def _handle_chat(self, message: str):
        """Handle natural language chat with AI."""
        # Close the input box dynamically
        width = self._get_terminal_width()
        box_width = max(40, width - 2)
        bottom_border = "└" + "─" * (box_width - 2) + "┘"

        console.print("[dim]│[/dim]")
        console.print(f"[dim]{bottom_border}[/dim]")

        # Add to conversation
        self.conversation.append({
            "role": "user",
            "content": message
        })

        # Build system prompt
        system_prompt = self._build_system_prompt()

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation
        ]

        # Show response (no "AI:" label)
        console.print()

        try:
            # Call API with streaming
            response_text = self._call_ai_streaming(messages)

            # Execute any tools in the response
            response_text = self._execute_tools(response_text)

            # Add to conversation
            self.conversation.append({
                "role": "assistant",
                "content": response_text
            })

            console.print()  # New line after response

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")
            # Remove failed user message
            self.conversation.pop()

    def _execute_tools(self, response_text: str) -> str:
        """Execute tools found in AI response."""
        # Find all tool calls in response
        tool_pattern = r'<TOOL>(.*?)</TOOL>'
        tools = re.findall(tool_pattern, response_text, re.DOTALL)

        if not tools:
            return response_text

        # Execute each tool
        for tool_call in tools:
            try:
                parts = tool_call.split('|', 2)
                tool_name = parts[0].strip()

                console.print(f"\n[cyan]⚙️  Executing: {tool_name}[/cyan]")

                if tool_name == "WRITE_FILE":
                    filename = parts[1].strip()
                    content = parts[2].strip() if len(parts) > 2 else ""
                    self._tool_write_file(filename, content)

                elif tool_name == "COMPILE":
                    filename = parts[1].strip()
                    self._tool_compile(filename)

                elif tool_name == "BENCHMARK":
                    filename = parts[1].strip()
                    self._tool_benchmark(filename)

                elif tool_name == "PROFILE":
                    filename = parts[1].strip()
                    self._tool_profile(filename)

                elif tool_name == "ANALYZE":
                    filename = parts[1].strip()
                    self._tool_analyze(filename)

                elif tool_name == "OPTIMIZE":
                    filename = parts[1].strip()
                    self._tool_optimize(filename)

                elif tool_name == "READ_FILE":
                    filename = parts[1].strip()
                    self._tool_read_file(filename)

                elif tool_name == "LIST_FILES":
                    self._tool_list_files()

            except Exception as e:
                console.print(f"[red]✗ Tool error: {e}[/red]")

        # Remove tool markers from response
        clean_response = re.sub(tool_pattern, '', response_text, flags=re.DOTALL)
        return clean_response

    def _tool_write_file(self, filename: str, content: str):
        """Write a CUDA file."""
        filepath = self.current_dir / filename
        filepath.write_text(content)
        console.print(f"[green]✓ Created {filename}[/green]")

    def _tool_compile(self, filename: str):
        """Compile a CUDA file."""
        filepath = self.current_dir / filename
        if not filepath.exists():
            console.print(f"[red]✗ File not found: {filename}[/red]")
            return

        try:
            result = self.compiler.compile(str(filepath))
            if result.get('success'):
                console.print(f"[green]✓ Compiled successfully[/green]")
                if result.get('warnings'):
                    console.print(f"[yellow]Warnings: {len(result['warnings'])}[/yellow]")
            else:
                console.print(f"[red]✗ Compilation failed[/red]")
                if result.get('error'):
                    console.print(f"[red]{result['error']}[/red]")
        except Exception as e:
            console.print(f"[red]✗ Compile error: {e}[/red]")

    def _tool_benchmark(self, filename: str):
        """Benchmark a CUDA file."""
        filepath = self.current_dir / filename
        if not filepath.exists():
            console.print(f"[red]✗ File not found: {filename}[/red]")
            return

        try:
            console.print("[dim]Running benchmark...[/dim]")
            result = self.benchmarker.benchmark_kernel(str(filepath))
            console.print(f"[green]✓ Benchmark complete:[/green]")
            console.print(f"  Execution time: {result.get('time', 'N/A')}ms")
            console.print(f"  Throughput: {result.get('throughput', 'N/A')} GB/s")
        except Exception as e:
            console.print(f"[red]✗ Benchmark error: {e}[/red]")

    def _tool_profile(self, filename: str):
        """Profile a CUDA file."""
        filepath = self.current_dir / filename
        if not filepath.exists():
            console.print(f"[red]✗ File not found: {filename}[/red]")
            return

        try:
            console.print("[dim]Profiling GPU metrics...[/dim]")
            result = self.profiler.profile_kernel(str(filepath))
            console.print(f"[green]✓ Profile complete:[/green]")
            console.print(f"  Occupancy: {result.get('occupancy', 'N/A')}%")
            console.print(f"  Registers/thread: {result.get('registers', 'N/A')}")
            console.print(f"  Shared memory: {result.get('shared_mem', 'N/A')}KB")
        except Exception as e:
            console.print(f"[red]✗ Profile error: {e}[/red]")

    def _tool_analyze(self, filename: str):
        """Analyze a CUDA file."""
        filepath = self.current_dir / filename
        if not filepath.exists():
            console.print(f"[red]✗ File not found: {filename}[/red]")
            return

        try:
            console.print("[dim]Analyzing code...[/dim]")
            result = self.analyzer.analyze(str(filepath))
            console.print(f"[green]✓ Analysis complete:[/green]")
            console.print(f"  Complexity: {result.get('complexity', 'N/A')}")
            console.print(f"  Opportunities: {result.get('opportunities', 0)}")
        except Exception as e:
            console.print(f"[red]✗ Analysis error: {e}[/red]")

    def _tool_optimize(self, filename: str):
        """Optimize a CUDA file."""
        filepath = self.current_dir / filename
        if not filepath.exists():
            console.print(f"[red]✗ File not found: {filename}[/red]")
            return

        try:
            console.print("[dim]Optimizing with AI...[/dim]")
            # Use the existing optimizer
            from .optimizer import CUDAOptimizer
            optimizer = CUDAOptimizer(self.cache_manager)
            result = optimizer.optimize(str(filepath))
            console.print(f"[green]✓ Optimization complete:[/green]")
            if result.get('output_file'):
                console.print(f"  Saved to: {result['output_file']}")
            if result.get('speedup'):
                console.print(f"  Speedup: {result['speedup']}x")
        except Exception as e:
            console.print(f"[red]✗ Optimization error: {e}[/red]")

    def _tool_read_file(self, filename: str):
        """Read a file."""
        filepath = self.current_dir / filename
        if not filepath.exists():
            console.print(f"[red]✗ File not found: {filename}[/red]")
            return

        try:
            content = filepath.read_text()
            console.print(f"[green]✓ Read {filename}:[/green]")
            # Show first 20 lines
            lines = content.split('\n')[:20]
            for i, line in enumerate(lines, 1):
                console.print(f"  {i:3d} | {line}")
            if len(content.split('\n')) > 20:
                console.print(f"  ... ({len(content.split('\n')) - 20} more lines)")
        except Exception as e:
            console.print(f"[red]✗ Read error: {e}[/red]")

    def _tool_list_files(self):
        """List CUDA files."""
        self._cmd_list()

    def _strip_markdown_realtime(self, chunk: str) -> str:
        """
        Strip markdown formatting in real-time during streaming.
        Uses a simpler, more robust approach.
        """
        # For simplicity and robustness, do immediate pattern replacement
        # This avoids complex buffering issues

        # Strip bold markers
        chunk = re.sub(r'\*\*([^*]+)\*\*', r'\1', chunk)  # **text**
        chunk = re.sub(r'__([^_]+)__', r'\1', chunk)      # __text__

        # Strip italic markers (single * or _)
        chunk = re.sub(r'(?<!\*)\*(?!\*)([^*]+)\*(?!\*)', r'\1', chunk)  # *text*
        chunk = re.sub(r'(?<!_)_(?!_)([^_]+)_(?!_)', r'\1', chunk)       # _text_

        # Strip code blocks with language specifier
        chunk = re.sub(r'```[a-zA-Z]*\n', '\n', chunk)  # ```python or ```cuda etc
        chunk = re.sub(r'```\n?', '', chunk)            # ``` alone

        # Strip inline code
        chunk = re.sub(r'`([^`]+)`', r'\1', chunk)

        # Strip headers but keep text
        chunk = re.sub(r'^#{1,6}\s+', '', chunk, flags=re.MULTILINE)

        # Convert markdown lists to simple dashes (avoid Unicode issues)
        chunk = re.sub(r'^[\s]*[-*+]\s+', '- ', chunk, flags=re.MULTILINE)

        # Strip blockquotes
        chunk = re.sub(r'^>\s+', '  ', chunk, flags=re.MULTILINE)

        # Handle strikethrough
        chunk = re.sub(r'~~([^~]+)~~', r'\1', chunk)

        # Handle images FIRST (before links) ![alt](url) -> [Image: alt]
        chunk = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[Image: \1]', chunk)

        # Handle links [text](url) -> text (after images)
        chunk = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', chunk)

        # Handle horizontal rules (use ASCII dashes)
        chunk = re.sub(r'^[-*_]{3,}$', '-' * 40, chunk, flags=re.MULTILINE)

        return chunk

    def _call_ai_streaming(self, messages: List[Dict]) -> str:
        """Call OpenRouter API with streaming and robust error handling."""
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://kubixai.co",
                        "X-Title": "Kubix CLI"
                    },
                    json={
                        "model": self.current_model['id'],
                        "messages": messages,
                        "stream": True,
                        "temperature": 0.7,
                        "max_tokens": 2000
                    },
                    stream=True,
                    timeout=60
                )

                if response.status_code != 200:
                    error_text = response.text
                    try:
                        error_json = json.loads(error_text)
                        error_msg = error_json.get('error', {}).get('message', error_text)
                    except:
                        error_msg = error_text

                    # If rate limit, retry
                    if response.status_code == 429:
                        retry_count += 1
                        if retry_count < max_retries:
                            console.print(f"\n[yellow]Rate limited, retrying ({retry_count}/{max_retries})...[/yellow]", end="")
                            import time
                            time.sleep(2 ** retry_count)  # Exponential backoff
                            continue

                    raise Exception(f"API error {response.status_code}: {error_msg}")

                # Stream response
                full_response = ""
                try:
                    for line in response.iter_lines():
                        if line:
                            try:
                                line_str = line.decode('utf-8', errors='ignore')
                                if line_str.startswith('data: '):
                                    data = line_str[6:].strip()
                                    if data == '[DONE]':
                                        break
                                    if not data:
                                        continue
                                    try:
                                        chunk = json.loads(data)
                                        if 'choices' in chunk and len(chunk['choices']) > 0:
                                            delta = chunk['choices'][0].get('delta', {})
                                            content = delta.get('content', '')
                                            if content:
                                                # Strip markdown before displaying
                                                clean_content = self._strip_markdown_realtime(content)
                                                # Use sys.stdout.write for immediate output
                                                try:
                                                    sys.stdout.write(clean_content)
                                                    sys.stdout.flush()
                                                except:
                                                    # Fallback if stdout fails
                                                    print(clean_content, end='')
                                                full_response += content  # Keep original for history
                                    except json.JSONDecodeError as e:
                                        # Skip malformed JSON chunks
                                        continue
                            except Exception as e:
                                # Skip problematic lines but continue streaming
                                continue
                except Exception as e:
                    # If we got partial response, return it
                    if full_response:
                        return full_response
                    raise

                return full_response

            except requests.exceptions.Timeout:
                retry_count += 1
                if retry_count < max_retries:
                    console.print(f"\n[yellow]Timeout, retrying ({retry_count}/{max_retries})...[/yellow]", end="")
                    continue
                raise Exception("Request timed out after multiple attempts. Please try again.")

            except requests.exceptions.ConnectionError:
                retry_count += 1
                if retry_count < max_retries:
                    console.print(f"\n[yellow]Connection error, retrying ({retry_count}/{max_retries})...[/yellow]", end="")
                    import time
                    time.sleep(1)
                    continue
                raise Exception("Connection error after multiple attempts. Please check your internet connection.")

            except Exception as e:
                # Don't retry for other exceptions
                raise Exception(f"API call failed: {str(e)}")

        raise Exception("Max retries exceeded")

    def _build_system_prompt(self) -> str:
        """Build system prompt for AI with tool calling."""
        cuda_files = list(self.current_dir.glob("*.cu")) + list(self.current_dir.glob("*.cuh"))
        file_list = [f.name for f in cuda_files[:10]]

        return f"""You are an AGENTIC CUDA development AI assistant with the ability to USE TOOLS.

CRITICAL FORMATTING RULES FOR TERMINAL OUTPUT:
- DO NOT use ANY markdown formatting whatsoever
- NO asterisks for bold (**text** or *text*)
- NO underscores for emphasis (__text__ or _text_)
- NO backticks for code (`code` or ```code```)
- NO hash symbols for headers (### Header)
- Use PLAIN TEXT ONLY for all responses
- For emphasis, use UPPERCASE letters
- For code, use simple indentation (4 spaces)
- For sections, use simple text like "Section Name:" followed by content
- For lists, use simple dashes or numbers

CONTEXT:
- Working directory: {self.current_dir}
- CUDA files: {', '.join(file_list) if file_list else 'None found'}
- Compact mode: {'ON' if self.compact else 'OFF'}

TOOLS AVAILABLE (you can and SHOULD use these):
You can use tools by writing special markers in your response:

1. WRITE_FILE: Create a new CUDA file
   Format: <TOOL>WRITE_FILE|filename.cu|code content here</TOOL>
   Example: <TOOL>WRITE_FILE|kernel.cu|__global__ void add(int* a){{...}}</TOOL>

2. COMPILE: Compile a CUDA file
   Format: <TOOL>COMPILE|filename.cu</TOOL>

3. BENCHMARK: Benchmark a kernel
   Format: <TOOL>BENCHMARK|filename.cu</TOOL>

4. PROFILE: Profile GPU metrics
   Format: <TOOL>PROFILE|filename.cu</TOOL>

5. ANALYZE: Deep code analysis
   Format: <TOOL>ANALYZE|filename.cu</TOOL>

6. OPTIMIZE: AI-powered optimization
   Format: <TOOL>OPTIMIZE|filename.cu</TOOL>

7. READ_FILE: Read a file's contents
   Format: <TOOL>READ_FILE|filename.cu</TOOL>

8. LIST_FILES: List all CUDA files
   Format: <TOOL>LIST_FILES</TOOL>

AGENTIC BEHAVIOR:
- When user asks to create a kernel, USE the WRITE_FILE tool
- When user wants optimization, USE the OPTIMIZE tool
- When user wants performance data, USE the BENCHMARK tool
- When user asks about GPU metrics, USE the PROFILE tool
- When user wants code analysis, USE the ANALYZE tool
- ALWAYS use tools when appropriate - don't just explain, DO IT!

INSTRUCTIONS:
- Be proactive - use tools to accomplish tasks
- Show tool output to user
- Explain what tools you're using
- Be concise {'(compact mode is ON)' if self.compact else ''}
- Use technical terms correctly

IMPORTANT: You are AGENTIC - when user asks you to do something, USE THE TOOLS to actually do it!
Don't just describe what to do - DO IT using the tools above.

Example:
User: "create a vector add kernel"
YOU: "I'll create a vector add kernel for you.

<TOOL>WRITE_FILE|vector_add.cu|__global__ void vectorAdd(float *a, float *b, float *c, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{
        c[i] = a[i] + b[i];
    }}
}}</TOOL>

✓ Created vector_add.cu

Would you like me to compile and benchmark it?"

Respond naturally and USE TOOLS when appropriate."""

    def _cmd_help(self):
        """Show help."""
        help_text = """
[bold cyan]Commands:[/bold cyan]

[bold]Chat:[/bold]
  Just type naturally to chat with AI
  Ask about CUDA, optimizations, or debugging

[bold]Commands:[/bold]
  /help       Show this help
  /model      Switch AI model
  /compact    Toggle compact mode (currently: {})
  /info       Show current model info
  /list       List CUDA files in directory
  /clear      Clear conversation history
  /exit       Exit Kubix CLI

[bold]Examples:[/bold]
  "How can I optimize my kernel?"
  "Explain shared memory in CUDA"
  "What's wrong with my code?"
  "Help me reduce memory usage"
""".format("ON" if self.compact else "OFF")
        console.print(help_text)

    def _cmd_list(self):
        """List CUDA files."""
        files = list(self.current_dir.glob("*.cu")) + list(self.current_dir.glob("*.cuh"))

        if not files:
            console.print("\n[yellow]No CUDA files found in current directory[/yellow]\n")
            return

        console.print("\n[bold cyan]CUDA Files:[/bold cyan]\n")
        for f in files:
            size = f.stat().st_size / 1024
            size_str = f"{size:.1f}KB" if size < 1024 else f"{size/1024:.1f}MB"
            console.print(f"  • {f.name:30s} {size_str:>8s}")
        console.print()

    def _cmd_info(self):
        """Show current model info."""
        console.print(f"\n[bold cyan]Current Configuration:[/bold cyan]\n")
        console.print(f"  Model: [bold]{self.current_model['name']}[/bold]")
        console.print(f"  Description: [dim]{self.current_model['description']}[/dim]")
        console.print(f"  Compact mode: [bold]{'ON' if self.compact else 'OFF'}[/bold]")
        console.print(f"  Working directory: [dim]{self.current_dir}[/dim]")
        console.print()

    def _setup_api_key(self):
        """Setup API key."""
        console.print("\n[yellow]Get your API key from: https://openrouter.ai[/yellow]\n")
        api_key = Prompt.ask("Enter API key", password=True)
        if api_key:
            self.cache_manager.save_api_key(api_key)
            console.print("[green]✓ API key saved[/green]")
        else:
            console.print("[red]No API key provided[/red]")
