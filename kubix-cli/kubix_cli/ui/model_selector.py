"""
Model selector UI for choosing OpenRouter models.
"""

from rich.table import Table
from rich.panel import Panel
from rich import box
from .theme import console


def show_models(current_model: str, model_catalog: dict) -> None:
    """
    Display available models with pricing.

    Args:
        current_model: Currently selected model ID
        model_catalog: Dictionary of available models with metadata
    """
    console.print()

    # Create table
    table = Table(
        title="🤖 Available Models",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold green",
        border_style="dim"
    )

    table.add_column("#", style="dim", width=3)
    table.add_column("Model", style="cyan", no_wrap=False)
    table.add_column("Input", justify="right", style="green")
    table.add_column("Output", justify="right", style="yellow")
    table.add_column("Context", justify="center", style="blue")
    table.add_column("Speed", justify="center", style="magenta")

    # Sort models by price (input + output)
    sorted_models = sorted(
        model_catalog.items(),
        key=lambda x: x[1]["input"] + x[1]["output"]
    )

    # Add rows
    for idx, (model_id, info) in enumerate(sorted_models, 1):
        # Highlight current model
        if model_id == current_model:
            marker = "▶"
            name_style = "bold green"
        else:
            marker = " "
            name_style = ""

        # Format pricing
        if info["input"] == 0.0 and info["output"] == 0.0:
            input_price = "[bold green]FREE[/bold green]"
            output_price = "[bold green]FREE[/bold green]"
        else:
            input_price = f"${info['input']:.3f}"
            output_price = f"${info['output']:.3f}"

        # Format speed with emoji
        speed_emoji = {
            "very fast": "⚡",
            "fast": "🚀",
            "medium": "⏱️",
            "slow": "🐢"
        }
        speed_display = f"{speed_emoji.get(info['speed'], '')} {info['speed']}"

        # Format name with style only if style exists
        if name_style:
            name_display = f"[{name_style}]{info['name']}[/{name_style}]"
        else:
            name_display = info['name']

        table.add_row(
            f"{marker}{idx}",
            name_display,
            input_price,
            output_price,
            info["context"],
            speed_display
        )

    console.print(table)

    # Legend
    console.print()
    console.print("[dim]💡 Pricing is per 1M tokens  •  [bold green]▶[/bold green] = current model[/dim]")
    console.print()
    console.print("[green]Usage:[/green] [white]/models <number>[/white] to switch  •  [white]/models reset[/white] for cheapest")
    console.print()


def select_model(choice: str, model_catalog: dict, default_model: str) -> str:
    """
    Select a model by number or reset to default.

    Args:
        choice: User's choice (number or "reset")
        model_catalog: Dictionary of available models
        default_model: Default (cheapest) model ID

    Returns:
        Selected model ID or None if invalid
    """
    if choice.lower() == "reset":
        return default_model

    try:
        # Parse number
        idx = int(choice)

        # Sort models same way as display
        sorted_models = sorted(
            model_catalog.items(),
            key=lambda x: x[1]["input"] + x[1]["output"]
        )

        # Get model by index
        if 1 <= idx <= len(sorted_models):
            model_id, info = sorted_models[idx - 1]
            return model_id
        else:
            return None
    except ValueError:
        return None
