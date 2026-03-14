"""
CUDA code syntax highlighting with NVIDIA theme
"""

from rich.syntax import Syntax
from .theme import console, NVIDIA_GREEN


def highlight_cuda_code(code: str, line_numbers: bool = True, start_line: int = 1) -> Syntax:
    """
    Highlight CUDA code with NVIDIA-themed syntax highlighting.

    Args:
        code: CUDA code to highlight
        line_numbers: Show line numbers
        start_line: Starting line number

    Returns:
        Rich Syntax object

    Example:
        cuda_code = '''
        __global__ void kernel() {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
        }
        '''
        syntax = highlight_cuda_code(cuda_code)
        console.print(syntax)
    """
    return Syntax(
        code,
        "cuda",  # Use CUDA lexer (C++ variant)
        theme="monokai",  # Dark theme that complements NVIDIA green
        line_numbers=line_numbers,
        start_line=start_line,
        word_wrap=False,
        background_color="#1E1E1E"  # NVIDIA dark background
    )


def print_cuda_code(code: str, title: str = None, line_numbers: bool = True):
    """
    Print highlighted CUDA code to console.

    Args:
        code: CUDA code to print
        title: Optional title
        line_numbers: Show line numbers
    """
    from .theme import themed_panel

    syntax = highlight_cuda_code(code, line_numbers=line_numbers)

    if title:
        panel = themed_panel(
            syntax,
            title=title,
            border_style="nvidia"
        )
        console.print(panel)
    else:
        console.print(syntax)


def highlight_diff(old_code: str, new_code: str, title: str = "Changes"):
    """
    Show a side-by-side diff of CUDA code.

    Args:
        old_code: Original code
        new_code: Modified code
        title: Diff title
    """
    from rich.columns import Columns
    from .theme import themed_panel

    old_syntax = highlight_cuda_code(old_code, line_numbers=True)
    new_syntax = highlight_cuda_code(new_code, line_numbers=True)

    old_panel = themed_panel(old_syntax, title="Before", border_style="error")
    new_panel = themed_panel(new_syntax, title="After", border_style="success")

    console.print(f"\n[nvidia]### {title}[/nvidia]\n")
    console.print(Columns([old_panel, new_panel], equal=True))


def highlight_error_line(code: str, error_line: int, error_message: str):
    """
    Highlight code with emphasis on error line.

    Args:
        code: CUDA code
        error_line: Line number with error
        error_message: Error message to display
    """
    from .theme import themed_panel, show_error

    syntax = highlight_cuda_code(code, line_numbers=True)

    panel = themed_panel(
        syntax,
        title=f"Error at line {error_line}",
        subtitle=f"[error]{error_message}[/error]",
        border_style="error"
    )

    console.print(panel)


def highlight_optimization(code: str, optimized_code: str, improvements: list[str]):
    """
    Show optimization results with before/after and improvement list.

    Args:
        code: Original code
        optimized_code: Optimized code
        improvements: List of improvements made
    """
    from rich.columns import Columns
    from .theme import themed_panel, themed_table

    # Code comparison
    old_syntax = highlight_cuda_code(code, line_numbers=True)
    new_syntax = highlight_cuda_code(optimized_code, line_numbers=True)

    old_panel = themed_panel(old_syntax, title="Original", border_style="warning")
    new_panel = themed_panel(new_syntax, title="Optimized", border_style="success")

    console.print(f"\n[nvidia]🚀 Optimization Results[/nvidia]\n")
    console.print(Columns([old_panel, new_panel], equal=True))

    # Improvements table
    console.print()
    table = themed_table(title="Improvements Applied")
    table.add_column("Optimization", style="nvidia")

    for improvement in improvements:
        table.add_row(f"✓ {improvement}")

    console.print(table)
