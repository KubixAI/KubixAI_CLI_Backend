#!/usr/bin/env python
"""
Kubix CLI - CUDA Kernel Optimizer
Part of the Kubix AI Code Editor ecosystem

Multi-Agent System with Session Management

Clean UX - suppress warnings before anything else.
"""

# Suppress warnings BEFORE any imports for clean output
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

if __name__ == "__main__":
    from kubix_cli.cli_minimal import main
    main()
