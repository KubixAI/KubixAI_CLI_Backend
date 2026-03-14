"""
Kubix CLI - Warning Suppression

Suppress annoying warnings for better UX.
"""

import warnings
import os
import sys


def suppress_all_warnings():
    """Suppress common annoying warnings."""

    # Suppress Python warnings
    warnings.filterwarnings('ignore')

    # Suppress PyTorch CUDA warnings about pynvml
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', message='.*pynvml.*')

    # Set environment variables to suppress warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'

    # Suppress CUDA warnings
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    # Redirect stderr temporarily for torch import
    original_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        import torch.cuda
        sys.stderr = original_stderr
    except:
        sys.stderr = original_stderr


def clean_stderr():
    """Clean up stderr after imports."""
    sys.stderr.flush()
    sys.stdout.flush()
