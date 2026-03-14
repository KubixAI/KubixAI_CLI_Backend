"""
Kubix CLI - Interactive Mode

Minimal, smooth UX with arrow key navigation.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich import box

# Arrow key navigation
import inquirer
from inquirer.themes import GreenPassion

console = Console()

# Colors
CYAN = "cyan"
GREEN = "green"
YELLOW = "yellow"
RED = "red"


class InteractiveSession:
    """Interactive session for Kubix CLI."""

    def __init__(self):
        self.current_dir = Path.cwd()

        from .cache import CacheManager
        self.cache_manager = CacheManager()

    def start(self):
        """Start interactive session."""
        # ASCII banner
        banner = """
    ██╗  ██╗██╗   ██╗██████╗ ██╗██╗  ██╗
    ██║ ██╔╝██║   ██║██╔══██╗██║╚██╗██╔╝
    █████╔╝ ██║   ██║██████╔╝██║ ╚███╔╝
    ██╔═██╗ ██║   ██║██╔══██╗██║ ██╔██╗
    ██║  ██╗╚██████╔╝██████╔╝██║██╔╝ ██╗
    ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═╝
        """
        console.print(f"[bold cyan]{banner}[/bold cyan]")
        console.print(f"[dim]CUDA Kernel Optimizer[/dim]")
        console.print(f"[dim]📂 {self.current_dir}[/dim]")
        console.print(f"[dim]Use ↑↓ arrows, Space to select, Enter to confirm[/dim]\n")

        while True:
            try:
                choice = self._show_main_menu()
                if not choice:
                    continue

                if choice == "optimize":
                    self._optimize_workflow()
                elif choice == "config":
                    self._show_config()
                elif choice == "exit":
                    console.print("\n[cyan]👋 Goodbye![/cyan]\n")
                    break

            except KeyboardInterrupt:
                console.print("\n")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]\n")
                import traceback
                traceback.print_exc()

    def _show_main_menu(self) -> Optional[str]:
        """Main menu with arrow keys."""

        choices = [
            ('Optimize kernels - Find and optimize CUDA kernels with AI', 'optimize'),
            ('Configuration - API key and settings', 'config'),
            ('Exit', 'exit'),
        ]

        questions = [
            inquirer.List('action',
                         message="What do you want to do?",
                         choices=choices,
                         carousel=True)
        ]

        try:
            answers = inquirer.prompt(questions, theme=GreenPassion())
            if answers:
                return answers.get('action')
            return None
        except Exception as e:
            console.print(f"[red]Menu error: {e}[/red]")
            return None

    def _optimize_workflow(self):
        """Optimization workflow - SIMPLE and CLEAR."""

        # Check API key
        if not self.cache_manager.has_api_key():
            console.print("\n[yellow]⚠️  API key required[/yellow]")
            self._setup_api_key()
            if not self.cache_manager.has_api_key():
                return

        # Find CUDA files
        console.print("\n[cyan]Looking for CUDA files...[/cyan]")
        cuda_files = self._find_cuda_files()

        if not cuda_files:
            questions = [inquirer.Confirm('recursive', message="No files found. Search subdirectories?", default=True)]
            answers = inquirer.prompt(questions)
            if answers and answers['recursive']:
                cuda_files = self._find_cuda_files(recursive=True)

        if not cuda_files:
            console.print("[red]No CUDA files found[/red]\n")
            return

        # Select file
        console.print(f"\n[dim]Found {len(cuda_files)} CUDA file(s)[/dim]")
        selected_file = self._select_file(cuda_files)
        if not selected_file:
            return

        # Parse kernels
        console.print(f"\n[cyan]Parsing {selected_file.name}...[/cyan]")
        kernels = self._parse_kernels(selected_file)

        if not kernels:
            console.print("[red]No kernels found[/red]\n")
            return

        # Select kernels
        console.print(f"[dim]Found {len(kernels)} kernel(s)[/dim]")
        selected_kernels = self._select_kernels(kernels)
        if not selected_kernels:
            return

        # Get options
        options = self._get_options(len(selected_kernels))
        if not options:
            return

        # Optimize
        console.print()
        self._run_optimization(selected_file, selected_kernels, options)

    def _find_cuda_files(self, recursive: bool = False) -> List[Path]:
        """Find CUDA files."""
        pattern = "**/*.cu" if recursive else "*.cu"
        cu_files = list(self.current_dir.glob(pattern))

        pattern = "**/*.cuh" if recursive else "*.cuh"
        cuh_files = list(self.current_dir.glob(pattern))

        return sorted(cu_files + cuh_files, key=lambda x: x.stat().st_mtime, reverse=True)

    def _select_file(self, files: List[Path]) -> Optional[Path]:
        """Select file with arrow keys."""

        if len(files) == 1:
            console.print(f"[green]Using {files[0].name}[/green]")
            return files[0]

        import datetime
        choices = []
        for f in files[:20]:
            kernels = self._parse_kernels(f)
            size = f.stat().st_size / 1024
            size_str = f"{size:.1f}KB" if size < 1024 else f"{size/1024:.1f}MB"
            date = datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d")

            label = f"{f.name:30s} {size_str:8s} {len(kernels):2d} kernels {date}"
            choices.append((label, f))

        questions = [
            inquirer.List('file',
                         message="Select file to optimize",
                         choices=choices,
                         carousel=True)
        ]

        try:
            answers = inquirer.prompt(questions, theme=GreenPassion())
            if answers:
                return answers.get('file')
            return None
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return None

    def _parse_kernels(self, file: Path) -> List[Dict[str, Any]]:
        """Parse kernels from file."""
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            pattern = r'__global__\s+(?:void|[\w\<\>\*\&]+)\s+(\w+)\s*\((.*?)\)'
            matches = re.finditer(pattern, content, re.MULTILINE)

            kernels = []
            for match in matches:
                name = match.group(1)
                params = match.group(2).strip()
                line = content[:match.start()].count('\n') + 1

                # Extract full kernel
                start = match.end()
                brace_count = 0
                end = start
                in_kernel = False

                for i, char in enumerate(content[start:], start):
                    if char == '{':
                        brace_count += 1
                        in_kernel = True
                    elif char == '}':
                        brace_count -= 1
                        if in_kernel and brace_count == 0:
                            end = i + 1
                            break

                code = content[match.start():end]

                kernels.append({
                    'name': name,
                    'params': params,
                    'line': line,
                    'code': code
                })

            return kernels
        except Exception as e:
            return []

    def _select_kernels(self, kernels: List[Dict]) -> List[Dict]:
        """Select kernels with checkboxes."""

        if len(kernels) == 1:
            console.print(f"[green]Optimizing {kernels[0]['name']}[/green]")
            return kernels

        choices = []
        for k in kernels:
            params = k['params'][:40] + "..." if len(k['params']) > 40 else k['params']
            label = f"{k['name']:30s} Line {k['line']:4d} {params}"
            choices.append((label, k))

        questions = [
            inquirer.Checkbox('kernels',
                            message="Select kernels (Space to select, Enter to confirm)",
                            choices=choices,
                            carousel=True)
        ]

        try:
            answers = inquirer.prompt(questions, theme=GreenPassion())
            if answers:
                selected = answers.get('kernels', [])
                if selected:
                    names = ", ".join([k['name'] for k in selected])
                    console.print(f"[green]Selected: {names}[/green]")
                return selected
            return []
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return []

    def _get_options(self, num_kernels: int) -> Optional[Dict[str, Any]]:
        """Get optimization options."""

        # Variants
        variant_choices = [(f"{i} variants", i) for i in [1, 2, 3, 5, 10]]
        questions = [
            inquirer.List('variants',
                         message="How many optimization attempts per kernel?",
                         choices=variant_choices,
                         default=3,
                         carousel=True)
        ]

        try:
            answers = inquirer.prompt(questions)
            if not answers:
                return None
            variants = answers['variants']
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return None

        # GPU
        gpu_choices = [
            ('sm_70 - Volta (V100)', 'sm_70'),
            ('sm_75 - Turing (RTX 20xx)', 'sm_75'),
            ('sm_80 - Ampere (A100)', 'sm_80'),
            ('sm_86 - Ampere (RTX 3060/3070)', 'sm_86'),
            ('sm_89 - Ada (RTX 40xx)', 'sm_89'),
            ('sm_90 - Hopper (H100)', 'sm_90'),
        ]

        questions = [
            inquirer.List('gpu',
                         message="Target GPU",
                         choices=gpu_choices,
                         default='sm_86',
                         carousel=True)
        ]

        try:
            answers = inquirer.prompt(questions)
            if not answers:
                return None
            gpu = answers['gpu']
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return None

        return {
            'variants': variants,
            'target_gpu': gpu,
            'max_registers': 255,
            'shared_mem_kb': 48,
            'iterations': 100
        }

    def _run_optimization(self, file: Path, kernels: List[Dict], options: Dict):
        """Run optimization - SIMPLIFIED."""

        from .kernel_analyzer import KernelAnalyzer
        from .openrouter import OpenRouterClient
        from .compiler import CUDACompiler
        from .bench import Benchmarker

        api_key = self.cache_manager.get_api_key()
        client = OpenRouterClient(api_key)
        analyzer = KernelAnalyzer()
        compiler = CUDACompiler()
        benchmarker = Benchmarker(iterations=options['iterations'])

        for idx, kernel in enumerate(kernels, 1):
            console.print(f"\n[bold cyan]Kernel {idx}/{len(kernels)}: {kernel['name']}[/bold cyan]")

            # Analyze
            console.print("[dim]Analyzing...[/dim]")
            try:
                analysis = analyzer.analyze_kernel(kernel['code'])
                console.print(f"[green]✓ Found {len(analysis.get('optimization_opportunities', []))} optimization opportunities[/green]")
            except Exception as e:
                console.print(f"[red]✗ Analysis failed: {e}[/red]")
                continue

            # Generate variants
            console.print(f"[dim]Generating {options['variants']} variants with AI...[/dim]")

            constraints = {
                'max_registers': options['max_registers'],
                'shared_memory_kb': options['shared_mem_kb'],
                'target_gpu': options['target_gpu']
            }

            variants = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[dim]{task.description}[/dim]"),
                BarColumn(bar_width=30),
                TextColumn("{task.percentage:>3.0f}%"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("", total=options['variants'])

                for i in range(options['variants']):
                    try:
                        generated = client.generate_kernel_optimizations(
                            original_code=kernel['code'],
                            analysis=analysis,
                            constraints=constraints,
                            num_variants=1
                        )
                        if generated:
                            variants.extend(generated)
                    except Exception as e:
                        console.print(f"[yellow]Variant {i+1} failed: {str(e)[:50]}[/yellow]")
                    progress.update(task, advance=1)

            if not variants:
                console.print("[red]✗ No variants generated[/red]")
                continue

            console.print(f"[green]✓ Generated {len(variants)} variants[/green]")

            # Compile
            console.print("[dim]Compiling...[/dim]")
            compiled = []

            for i, variant in enumerate(variants):
                try:
                    variant_dict = {
                        'code': variant.code,
                        'operation': variant.operation,
                        'constraints': variant.constraints,
                        'metadata': variant.metadata
                    }
                    comp = compiler.compile_kernel(variant_dict)
                    compiled.append(comp)
                except Exception as e:
                    # Silently skip failed compilations
                    pass

            if not compiled:
                console.print("[red]✗ All variants failed to compile[/red]")
                continue

            console.print(f"[green]✓ {len(compiled)}/{len(variants)} compiled successfully[/green]")

            # Benchmark
            console.print("[dim]Benchmarking...[/dim]")
            results = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[dim]{task.description}[/dim]"),
                BarColumn(bar_width=30),
                TextColumn("{task.percentage:>3.0f}%"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("", total=len(compiled))

                for variant in compiled:
                    try:
                        bench = benchmarker.benchmark_kernel_standalone(variant, analysis)
                        results.append({'variant': variant, 'time': bench['avg_time_ms']})
                    except:
                        pass
                    progress.update(task, advance=1)

            if not results:
                console.print("[red]✗ Benchmarking failed[/red]")
                continue

            # Show results
            results.sort(key=lambda x: x['time'])

            console.print("\n[bold cyan]Results:[/bold cyan]")
            table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
            table.add_column("#", style="cyan", width=3)
            table.add_column("Time (ms)", justify="right", style="yellow")
            table.add_column("Speedup", justify="right", style="green")

            baseline = results[-1]['time']
            for i, r in enumerate(results[:5], 1):  # Top 5
                speedup = baseline / r['time']
                table.add_row(
                    str(i),
                    f"{r['time']:.3f}",
                    f"{speedup:.2f}x"
                )

            console.print(table)

            # Save best
            best = results[0]
            output_name = f"{file.stem}_{kernel['name']}_optimized{file.suffix}"
            output_path = self.current_dir / output_name

            with open(output_path, 'w') as f:
                f.write(best['variant']['code'])

            console.print(f"\n[green]✓ Saved: {output_name}[/green]")
            console.print(f"[dim]Best time: {best['time']:.3f}ms ({baseline/best['time']:.2f}x faster)[/dim]")

        console.print("\n[bold green]✓ Done![/bold green]\n")

    def _show_config(self):
        """Show config."""
        config = self.cache_manager.get_config()

        console.print(f"\n[cyan]Configuration:[/cyan]")
        console.print(f"  API key: {'✓ Set' if config.get('openrouter_api_key') else '✗ Not set'}")
        console.print(f"  Cache: {self.cache_manager.count_cached_kernels()} kernels\n")

        questions = [
            inquirer.Confirm('update', message="Update API key?", default=False)
        ]
        answers = inquirer.prompt(questions)

        if answers and answers['update']:
            self._setup_api_key()

    def _setup_api_key(self):
        """Setup API key."""
        console.print("\n[yellow]Get your API key from: https://openrouter.ai[/yellow]")

        questions = [
            inquirer.Password('key', message="Enter API key")
        ]

        try:
            answers = inquirer.prompt(questions)
            if answers and answers['key']:
                self.cache_manager.save_api_key(answers['key'])
                console.print("[green]✓ API key saved[/green]\n")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")
