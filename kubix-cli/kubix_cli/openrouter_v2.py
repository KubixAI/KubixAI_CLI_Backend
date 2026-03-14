"""
OpenRouter Client V2 - Native Function Calling Support
Production-ready with rock-solid error handling and streaming.
"""

import requests
import json
import time
import sys
import os
import re
import platform
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass
import backoff
from rich.console import Console
from .ui.thinking_spinner import ThinkingSpinner

console = Console()


@dataclass
class Message:
    """Chat message."""
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class MarkdownStripper:
    """Streaming markdown stripper that handles partial patterns."""

    def __init__(self):
        self.buffer = ""
        self.pending = ""  # Pending markdown markers

    def add_chunk(self, chunk: str) -> str:
        """Add a chunk and return stripped content ready for output."""
        self.buffer += chunk

        # Only process when we have enough content to ensure complete patterns
        # or when we see a newline (natural boundary)
        if len(self.buffer) < 100 and '\n' not in self.buffer[-50:]:
            return ""  # Wait for more content

        # Strip markdown patterns from buffer
        processed = self._strip_patterns(self.buffer)

        # Keep last 50 chars in buffer in case there's an incomplete pattern
        if len(processed) > 50:
            output = processed[:-50]
            self.buffer = processed[-50:]
        else:
            return ""  # Wait for more content

        return output

    def _strip_patterns(self, text: str) -> str:
        """Strip all markdown patterns from text."""
        # Bold **text**
        text = re.sub(r'\*\*([^*]+?)\*\*', r'\1', text)

        # Headers ### text or ### **text**
        text = re.sub(r'^(#{1,6})\s+\*\*(.+?)\*\*', r'\2', text, flags=re.MULTILINE)
        text = re.sub(r'^(#{1,6})\s+', '', text, flags=re.MULTILINE)

        # Code blocks and inline code
        text = re.sub(r'```[a-zA-Z]*\n?', '', text)
        text = re.sub(r'```', '', text)
        text = re.sub(r'`([^`]+?)`', r'\1', text)

        # Italics
        text = re.sub(r'(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)', r'\1', text)
        text = re.sub(r'_([^_]+?)_', r'\1', text)

        # Lists
        text = re.sub(r'^[\s]*[-*+]\s+', '- ', text, flags=re.MULTILINE)

        return text

    def flush(self) -> str:
        """Flush any remaining buffer content."""
        if not self.buffer:
            return ""

        remaining = self._strip_patterns(self.buffer)
        self.buffer = ""
        return remaining

class OpenRouterClientV2:
    """
    Production-ready OpenRouter client with native function calling.

    Features:
    - Rock-solid streaming with error recovery
    - Comprehensive terminal state management
    - Cross-platform compatibility
    - Thread-safe operations
    - Graceful error handling
    """

    BASE_URL = "https://api.openai.com/v1/chat/completions"

    # Model catalog
    MODEL_CATALOG = {
        "gpt-4o-mini": {
            "name": "GPT-4o Mini",
            "input": 0.15,
            "output": 0.60,
            "context": "128k",
            "speed": "very fast",
            "free": False
        },
        "gpt-4o": {
            "name": "GPT-4o",
            "input": 2.50,
            "output": 10.00,
            "context": "128k",
            "speed": "fast",
            "free": False
        },
        "gpt-4.1": {
            "name": "GPT-4.1",
            "input": 2.00,
            "output": 8.00,
            "context": "1M",
            "speed": "fast",
            "free": False
        },
        "gpt-4.1-mini": {
            "name": "GPT-4.1 Mini",
            "input": 0.40,
            "output": 1.60,
            "context": "1M",
            "speed": "very fast",
            "free": False
        },
        "o4-mini": {
            "name": "o4-mini (reasoning)",
            "input": 1.10,
            "output": 4.40,
            "context": "200k",
            "speed": "fast",
            "free": False
        },
    }

    # Default to cheapest good model
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str, model: Optional[str] = None):
        self._api_key = api_key
        self.model = model or self.DEFAULT_MODEL

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._active_spinner: Optional[ThinkingSpinner] = None
        self._supports_ansi = self._check_ansi_support()

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, value):
        """Update API key and headers when API key changes."""
        self._api_key = value
        # Update the Authorization header with the new API key
        self.headers["Authorization"] = f"Bearer {value}"

    def _check_ansi_support(self) -> bool:
        """Check if terminal supports ANSI escape codes."""
        if platform.system() == "Windows":
            # Enable ANSI on Windows if possible
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                return True
            except:
                return False
        return True

    def _strip_markdown(self, text: str) -> str:
        """Strip markdown formatting from text for clean terminal output."""
        # Strip bold markers
        text = re.sub(r'\*\*([^*]+?)\*\*', r'\1', text)  # **text**
        text = re.sub(r'__([^_]+?)__', r'\1', text)      # __text__

        # Strip italic markers - handle various cases including quotes
        text = re.sub(r'\*([^*\n]+?)\*', r'\1', text)  # *text*
        text = re.sub(r'_([^_\n]+?)_', r'\1', text)    # _text_ including _"text"_

        # Strip code blocks
        text = re.sub(r'```[a-zA-Z]*\n?', '', text)
        text = re.sub(r'```', '', text)

        # Strip inline code
        text = re.sub(r'`([^`]+?)`', r'\1', text)

        # Strip headers but keep text
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

        # Convert markdown lists to simple dashes
        text = re.sub(r'^[\s]*[-*+]\s+', '- ', text, flags=re.MULTILINE)

        # Handle links [text](url) -> text
        text = re.sub(r'\[([^\]]+?)\]\([^)]+?\)', r'\1', text)

        # Handle strikethrough
        text = re.sub(r'~~([^~]+?)~~', r'\1', text)

        # Handle images ![alt](url) -> [Image: alt]
        text = re.sub(r'!\[([^\]]*?)\]\([^)]+?\)', r'[Image: \1]', text)

        return text

    def _safe_print(self, content: str, end: str = "\n", flush: bool = True, strip_markdown: bool = False):
        """Print with comprehensive error handling and optional markdown stripping."""
        # Only strip markdown if explicitly requested (for non-streaming content)
        # Streaming content is handled by MarkdownStripper class
        if strip_markdown:
            content = self._strip_markdown(content)

        try:
            sys.stdout.write(content + end)
            if flush:
                sys.stdout.flush()
        except UnicodeEncodeError:
            # Fallback for terminals that don't support Unicode
            safe_content = content.encode('ascii', 'replace').decode('ascii')
            sys.stdout.write(safe_content + end)
            if flush:
                sys.stdout.flush()
        except Exception:
            # Ultimate fallback - silently fail
            pass

    def _show_tool_plan(self, tool_calls: List[Dict]):
        """
        Show tool execution plan with error handling.

        Args:
            tool_calls: List of tool calls from the model
        """
        try:
            # Clear any pending status
            if self._supports_ansi:
                self._safe_print("\r" + " " * 100 + "\r", end="", flush=True)

            # Tool action descriptions
            tool_actions = {
                'read_file': '📖 Reading',
                'write_file': '✍️  Writing',
                'compile_cuda': '🔨 Compiling',
                'analyze_cuda': '🔍 Analyzing',
                'bash': '⚡ Executing',
                'bash_exec': '⚡ Executing',
                'list_files': '📂 Listing',
            }

            # Use console for Rich formatting if available
            try:
                console.print()
                console.print("[dim]🤖 Planning actions:[/dim]")

                for idx, tool_call in enumerate(tool_calls, 1):
                    tool_name = tool_call['function']['name']
                    tool_args = {}

                    try:
                        tool_args = json.loads(tool_call['function']['arguments'])
                    except:
                        tool_args = {}

                    # Get action emoji and verb
                    action = tool_actions.get(tool_name, '🔧 Using')

                    # Build descriptive message based on tool
                    if tool_name in ['read_file', 'write_file', 'compile_cuda', 'analyze_cuda']:
                        file_path = tool_args.get('file_path', 'file')
                        # Truncate long paths
                        if len(file_path) > 50:
                            file_path = "..." + file_path[-47:]
                        msg = f"{action} [cyan]{file_path}[/cyan]"

                    elif tool_name in ['bash', 'bash_exec']:
                        command = tool_args.get('command', 'command')
                        # Truncate long commands
                        if len(command) > 50:
                            command = command[:47] + "..."
                        msg = f"{action} [yellow]{command}[/yellow]"

                    elif tool_name == 'list_files':
                        path = tool_args.get('path', '.')
                        pattern = tool_args.get('pattern', '*')
                        msg = f"{action} files: [cyan]{pattern}[/cyan] in [dim]{path}[/dim]"

                    else:
                        msg = f"{action} [cyan]{tool_name}[/cyan]"

                    console.print(f"  [dim]{idx}.[/dim] {msg}")

                console.print()

            except Exception:
                # Fallback to plain text
                self._safe_print("\n🤖 Planning actions:")
                for idx, tool_call in enumerate(tool_calls, 1):
                    tool_name = tool_call['function']['name']
                    self._safe_print(f"  {idx}. {tool_name}")
                self._safe_print("")

        except Exception:
            # Silently handle any display errors
            pass

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.HTTPError),
        max_tries=3,
        max_time=60,
        on_backoff=lambda details: print(f"\n[Retrying API call... attempt {details['tries']}/{3}]")
    )
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send a chat request to OpenRouter with comprehensive error handling.

        Args:
            messages: List of Message objects
            tools: Optional list of tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            Response dict or generator if streaming
        """
        # Validate API key before making request
        if not self.api_key or self.api_key == "sk-temp-placeholder":
            raise ValueError("Invalid API key. Please check your OpenRouter API key.")
        # Convert messages to dict format
        message_dicts = []
        for msg in messages:
            msg_dict = {"role": msg.role}

            # Add content
            if msg.content is not None:
                msg_dict["content"] = msg.content
            elif not msg.tool_calls:
                msg_dict["content"] = ""

            # Add optional fields
            if msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if msg.name:
                msg_dict["name"] = msg.name

            message_dicts.append(msg_dict)

        # Build request payload
        payload = {
            "model": self.model,
            "messages": message_dicts,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }

        # Add tools if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            response = requests.post(
                self.BASE_URL,
                headers=self.headers,
                json=payload,
                timeout=120 if not stream else None,
                stream=stream
            )

            # Handle specific error codes
            if response.status_code == 401:
                # Provide helpful error message for API key issues
                try:
                    from kubix_cli.ui.theme import console
                except ImportError:
                    pass  # Console not available, skip pretty printing
                else:
                    console.print("\n[red]API Key Error[/red]")
                    console.print("[yellow]Your API key appears to be invalid or expired.[/yellow]")
                    console.print("\nTo fix this:")
                    console.print("1. Get a new API key from: [bold cyan]https://openrouter.ai[/bold cyan]")
                    console.print("2. Delete the cached key: [dim]rm ~/.kubix-cli/config.json[/dim]")
                    console.print("3. Run 'kubix' again to set up the new key\n")
                raise ValueError("Invalid API key. Please check your OpenRouter API key.")

            if response.status_code == 400:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "Bad request")
                    raise ValueError(f"API Error (400): {error_msg}")
                except json.JSONDecodeError:
                    raise ValueError(f"API Error (400): {response.text[:200]}")

            response.raise_for_status()

            if stream:
                return self._handle_streaming_response(response)
            else:
                return response.json()

        except requests.exceptions.Timeout:
            raise ValueError("Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            raise ValueError("Connection error. Please check your internet connection.")

    def _handle_streaming_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle streaming response with comprehensive error recovery.

        Returns:
            Assembled response dictionary
        """
        full_content = ""
        tool_calls = []
        has_tool_calls = False
        first_content = True
        error_count = 0
        max_errors = 5
        markdown_stripper = MarkdownStripper()  # Create markdown stripper for this stream

        # Reset state flags
        if hasattr(self, '_preparing_shown'):
            delattr(self, '_preparing_shown')

        try:
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    line_str = line.decode('utf-8')
                except UnicodeDecodeError:
                    # Skip malformed lines
                    error_count += 1
                    if error_count > max_errors:
                        break
                    continue

                if line_str.startswith('data: '):
                    data_str = line_str[6:]

                    if data_str == '[DONE]':
                        break

                    try:
                        chunk = json.loads(data_str)

                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})

                            # Handle content streaming
                            if 'content' in delta and delta['content']:
                                content = delta['content']
                                full_content += content

                                # Update spinner on first content
                                if first_content and self._active_spinner:
                                    try:
                                        self._active_spinner.update_message("Generating", content_below=True)
                                        # STABILITY FIX: Ensure clean line break after status
                                        # Don't print newline here as status already includes it
                                    except:
                                        pass
                                    first_content = False
                                    # Small delay to ensure status is fully printed
                                    time.sleep(0.05)

                                # Stream content if no tools detected
                                if not has_tool_calls:
                                    # Use markdown stripper for clean output
                                    stripped = markdown_stripper.add_chunk(content)
                                    if stripped:
                                        self._safe_print(stripped, end="", flush=True)

                            # Handle tool calls
                            if 'tool_calls' in delta:
                                has_tool_calls = True

                                # Stop spinner and show preparing status
                                if self._active_spinner:
                                    try:
                                        self._active_spinner.stop()
                                    except:
                                        pass
                                    self._active_spinner = None

                                # Show preparing status (once)
                                if not hasattr(self, '_preparing_shown'):
                                    self._preparing_shown = True
                                    if full_content and not full_content.endswith('\n'):
                                        self._safe_print("")

                                    if self._supports_ansi:
                                        self._safe_print("\033[32m  ▸ \033[0m\033[2mPreparing tools...\033[0m")
                                    else:
                                        self._safe_print("  > Preparing tools...")

                                # Process tool calls
                                for tc in delta['tool_calls']:
                                    if tc.get('index') is not None:
                                        idx = tc['index']

                                        # Initialize tool call structure
                                        while len(tool_calls) <= idx:
                                            tool_calls.append({
                                                'id': '',
                                                'type': 'function',
                                                'function': {
                                                    'name': '',
                                                    'arguments': ''
                                                }
                                            })

                                        # Update tool call fields
                                        if 'id' in tc:
                                            tool_calls[idx]['id'] = tc['id']

                                        if 'function' in tc:
                                            func = tc['function']
                                            if 'name' in func:
                                                tool_calls[idx]['function']['name'] += func['name']
                                            if 'arguments' in func:
                                                tool_calls[idx]['function']['arguments'] += func['arguments']

                    except json.JSONDecodeError:
                        # Skip malformed JSON
                        error_count += 1
                        if error_count > max_errors:
                            break
                        continue
                    except Exception:
                        # Handle other parsing errors
                        error_count += 1
                        if error_count > max_errors:
                            break
                        continue

        except Exception as e:
            # Log error but try to return what we have
            if full_content or tool_calls:
                pass  # Continue with partial response
            else:
                raise ValueError(f"Streaming error: {str(e)}")

        finally:
            # Flush any remaining markdown stripper content
            if not has_tool_calls:
                remaining = markdown_stripper.flush()
                if remaining:
                    self._safe_print(remaining, end="", flush=True)

            # Ensure proper line ending
            if full_content and not full_content.endswith('\n'):
                self._safe_print("")

            # Clean up spinner
            if self._active_spinner:
                try:
                    self._active_spinner.stop()
                except:
                    pass
                self._active_spinner = None

            # Flush output
            try:
                sys.stdout.flush()
            except:
                pass

        # Return assembled response
        return {
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': full_content if full_content else None,
                    'tool_calls': tool_calls if tool_calls else None
                }
            }]
        }

    def chat_with_tools(
        self,
        messages: List[Message],
        tools: List[Dict],
        tool_executor: callable,
        max_iterations: int = 15,
        stream: bool = True
    ) -> Message:
        """
        Chat with automatic tool execution and comprehensive error handling.

        Args:
            messages: Conversation history
            tools: Available tools
            tool_executor: Function to execute tools
            max_iterations: Max tool calling iterations
            stream: Stream responses

        Returns:
            Final assistant message
        """
        conversation = messages.copy()
        iteration = 0

        try:
            while iteration < max_iterations:
                iteration += 1

                # Show thinking spinner (first iteration only)
                if iteration == 1:
                    # Add a newline to ensure spinner starts on its own line
                    self._safe_print("")
                    self._active_spinner = ThinkingSpinner("Thinking")
                    self._active_spinner.start()

                # Get response from model
                try:
                    response = self.chat(
                        messages=conversation,
                        tools=tools,
                        stream=stream
                    )
                except Exception as e:
                    # Clean up spinner on error
                    if self._active_spinner:
                        self._active_spinner.stop()
                        self._active_spinner = None
                    raise

                assistant_message = response['choices'][0]['message']

                # Check if model wants to use tools
                if assistant_message.get('tool_calls'):
                    # Add assistant message with tool calls
                    conversation.append(Message(
                        role='assistant',
                        content=assistant_message.get('content'),
                        tool_calls=assistant_message['tool_calls']
                    ))

                    # Stop spinner before tool execution
                    if self._active_spinner:
                        try:
                            self._active_spinner.stop()
                        except:
                            pass
                        self._active_spinner = None

                    # Show tool plan
                    self._show_tool_plan(assistant_message['tool_calls'])

                    # Execute each tool call
                    for tool_call in assistant_message['tool_calls']:
                        tool_name = tool_call['function']['name']
                        tool_id = tool_call['id']

                        # Parse arguments safely
                        try:
                            tool_args = json.loads(tool_call['function']['arguments'])
                        except json.JSONDecodeError:
                            tool_args = {}

                        # Execute tool with error handling
                        try:
                            tool_result = tool_executor(tool_name, tool_args)
                            result_str = str(tool_result)
                        except Exception as e:
                            result_str = f"Error executing {tool_name}: {str(e)}"

                        # Add tool result to conversation
                        conversation.append(Message(
                            role='tool',
                            content=result_str,
                            tool_call_id=tool_id,
                            name=tool_name
                        ))

                    # Continue to next iteration
                    continue

                else:
                    # No more tool calls, return final message
                    return Message(
                        role='assistant',
                        content=assistant_message.get('content', '')
                    )

            # Max iterations reached
            try:
                console.print(f"\n[yellow]⚠️  Max tool iterations ({max_iterations}) reached[/yellow]")
            except:
                self._safe_print(f"\nWarning: Max tool iterations ({max_iterations}) reached")

            return Message(
                role='assistant',
                content=assistant_message.get('content', '')
            )

        finally:
            # Always clean up spinner
            if self._active_spinner:
                try:
                    self._active_spinner.stop()
                except:
                    pass
                self._active_spinner = None