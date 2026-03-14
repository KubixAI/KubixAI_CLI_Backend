# Installation Guide

Kubix CLI can be installed using various methods depending on your platform and preferences.

## Quick Install

### Linux / macOS

```bash
# Using curl (recommended)
curl -sSL [url] | bash

# Or using pip
pip install kubix-cli
```

### Windows

```powershell
# Using PowerShell (run as Administrator)
irm [url] | iex

# Or using pip
pip install kubix-cli
```

## Prerequisites

### Required
- **Python 3.9+** - [Download Python](#)
- **OpenRouter API Key** - Get Free API Key (30 seconds, no credit card)

### Optional (for GPU acceleration)
- **NVIDIA GPU** - Any CUDA-capable GPU
- **CUDA Toolkit 11.0+** - [Download CUDA](#)
- **cuDNN** - [Download cuDNN](#)

## Installation Methods

### Method 1: Install via pip (Recommended)

The simplest way to install Kubix CLI:

```bash
# Install from PyPI
pip install kubix-cli

# Or install with user flag (no admin required)
pip install --user kubix-cli

# Upgrade to latest version
pip install --upgrade kubix-cli
```

### Method 2: Install from Source

For the latest development version:

```bash
# Clone the repository
git clone [url]
cd kubix-cli

# Install the package
pip install -e .
```

### Method 3: Install in Virtual Environment

For an isolated installation:

```bash
# Create virtual environment
python -m venv kubix-env

# Activate environment
# On Linux/macOS:
source kubix-env/bin/activate
# On Windows:
kubix-env\Scripts\activate

# Install Kubix CLI
pip install -e .
```

### Method 4: Using Docker (Coming Soon)

```bash
# Pull the Docker image
docker pull kubixai/kubix-cli

# Run the container
docker run -it kubixai/kubix-cli
```

## Platform-Specific Instructions

### Ubuntu/Debian

```bash
# Install Python and pip
sudo apt update
sudo apt install python3 python3-pip

# Install CUDA (optional)
wget [url]
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda

# Install Kubix CLI
pip install kubix-cli
```

### Fedora/RHEL/CentOS

```bash
# Install Python and pip
sudo dnf install python3 python3-pip

# Install Kubix CLI
pip install kubix-cli
```

### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL [url])"

# Install Python
brew install python@3.11

# Install Kubix CLI
pip3 install kubix-cli
```

### Windows

#### Option 1: Using Windows Package Manager

```powershell
# Install Python
winget install Python.Python.3.11

# Install Kubix CLI
pip install kubix-cli
```

#### Option 2: Manual Installation

1. Download Python from [python.org](#)
2. During installation, check "Add Python to PATH"
3. Open Command Prompt or PowerShell
4. Run: `pip install kubix-cli`

## Post-Installation Setup

### 1. Verify Installation

```bash
# Check if Kubix is installed
kubix --version

# Get help
kubix --help
```

### 2. Configure API Key

When you first run Kubix, you'll be prompted to enter your OpenRouter API key:

```bash
# Start Kubix CLI
kubix

# You'll see:
# ============================================================
# Welcome to Kubix CLI!
# ============================================================
#
# API Key Setup Required
#
# Quick Setup:
# 1. Open OpenRouter
# 2. Click 'Sign Up' (use Google/GitHub for instant access)
# 3. Copy your API key from the dashboard
```

### 3. Test GPU Support (Optional)

```bash
# In Kubix CLI, use the /gpu command
/gpu

# You should see your GPU status if CUDA is properly installed
```

## Environment Variables

You can configure Kubix using environment variables:

```bash
# Set API key (alternative to interactive setup)
export OPENROUTER_API_KEY="your-api-key-here"

# Set default model
export KUBIX_MODEL="google/gemini-2.0-flash-thinking-exp:free"

# Set working directory
export KUBIX_WORKDIR="/path/to/cuda/projects"
```

## Troubleshooting

### Command Not Found

If `kubix` command is not found after installation:

**Linux/macOS:**
```bash
# Add to PATH in ~/.bashrc or ~/.zshrc
export PATH="$HOME/.local/bin:$PATH"

# Reload shell configuration
source ~/.bashrc  # or source ~/.zshrc
```

**Windows:**
```powershell
# Add Python Scripts to PATH
$env:PATH += ";$env:APPDATA\Python\Python311\Scripts"

# Or reinstall with admin privileges
pip uninstall kubix-cli
pip install kubix-cli --user
```

### Permission Denied

If you get permission errors:

```bash
# Use --user flag
pip install --user kubix-cli

# Or use sudo (not recommended)
sudo pip install kubix-cli
```

### SSL Certificate Error

If you encounter SSL errors:

```bash
# Upgrade certificates
pip install --upgrade certifi

# Or use trusted host
pip install --trusted-host pypi.org kubix-cli
```

### CUDA Not Detected

If GPU/CUDA is not detected:

1. Verify NVIDIA drivers: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Ensure CUDA is in PATH:
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

## Uninstallation

To remove Kubix CLI:

```bash
# Uninstall via pip
pip uninstall kubix-cli

# Remove configuration files (optional)
rm -rf ~/.kubix-cli
```

## Support

- **Email**: hello@kubix.ai

## Next Steps

1. Get your free API key from OpenRouter
2. Run `kubix` to start the CLI
3. Try creating a simple CUDA kernel: "Create a vector addition kernel"
4. Explore commands with `/help`

Happy CUDA coding!
