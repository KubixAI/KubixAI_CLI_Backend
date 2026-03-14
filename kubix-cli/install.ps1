# Kubix CLI Installation Script for Windows
# Run this script in PowerShell as Administrator:
# irm https://raw.githubusercontent.com/Kubix-AI/kubix-cli/main/install.ps1 | iex

$ErrorActionPreference = "Stop"

# Configuration
$REPO = "Kubix-AI/kubix-cli"
$MIN_PYTHON_VERSION = "3.9"

# Colors
function Write-Success { Write-Host "✓ " -ForegroundColor Green -NoNewline; Write-Host $args }
function Write-Error { Write-Host "✗ " -ForegroundColor Red -NoNewline; Write-Host $args }
function Write-Info { Write-Host "ℹ " -ForegroundColor Cyan -NoNewline; Write-Host $args }
function Write-Warning { Write-Host "⚠ " -ForegroundColor Yellow -NoNewline; Write-Host $args }

# Banner
function Show-Banner {
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║        Kubix CLI Installer                  ║" -ForegroundColor Cyan
    Write-Host "║        GPU-Native AI Code Editor               ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

# Check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Check Python installation
function Test-Python {
    Write-Info "Checking Python installation..."

    $pythonCmd = $null

    # Try different Python commands
    foreach ($cmd in @("python", "python3", "py")) {
        try {
            $version = & $cmd --version 2>&1
            if ($version -match "Python (\d+)\.(\d+)") {
                $major = [int]$matches[1]
                $minor = [int]$matches[2]

                if ($major -eq 3 -and $minor -ge 9) {
                    $pythonCmd = $cmd
                    Write-Success "Python $major.$minor found"
                    return $cmd
                } else {
                    Write-Warning "Python $major.$minor found (need 3.9+)"
                }
            }
        } catch {
            # Command not found, continue
        }
    }

    if (-not $pythonCmd) {
        Write-Error "Python 3.9+ is required but not found"
        Write-Host ""
        Write-Host "Please install Python from: " -NoNewline
        Write-Host "https://python.org" -ForegroundColor Cyan
        Write-Host "Make sure to check 'Add Python to PATH' during installation!"
        exit 1
    }

    return $pythonCmd
}

# Check CUDA installation
function Test-Cuda {
    Write-Info "Checking CUDA installation..."

    try {
        $nvccVersion = & nvcc --version 2>&1
        if ($nvccVersion -match "release (\d+\.\d+)") {
            Write-Success "CUDA $($matches[1]) found"
            return $true
        }
    } catch {
        # CUDA not found
    }

    Write-Warning "CUDA not found (optional for GPU acceleration)"
    Write-Host "    Install CUDA from: https://developer.nvidia.com/cuda-downloads"
    return $false
}

# Check NVIDIA GPU
function Test-NvidiaGpu {
    Write-Info "Checking for NVIDIA GPU..."

    try {
        $smiOutput = & nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
        if ($LASTEXITCODE -eq 0) {
            $gpuName = $smiOutput.Trim()
            Write-Success "NVIDIA GPU found: $gpuName"
            return $true
        }
    } catch {
        # nvidia-smi not found
    }

    Write-Warning "No NVIDIA GPU detected"
    return $false
}

# Install via pip
function Install-ViaPip {
    param([string]$PythonCmd)

    Write-Info "Installing Kubix CLI via pip..."

    # Upgrade pip
    Write-Info "Upgrading pip..."
    & $PythonCmd -m pip install --upgrade pip | Out-Null

    # Install kubix-cli
    Write-Info "Installing kubix-cli package..."
    try {
        & $PythonCmd -m pip install --user kubix-cli
        if ($LASTEXITCODE -ne 0) {
            throw "Installation failed"
        }
        Write-Success "Kubix CLI installed successfully"
    } catch {
        Write-Error "Installation failed: $_"
        exit 1
    }

    # Check if Scripts directory is in PATH
    $userScriptsPath = "$env:APPDATA\Python\Python3*\Scripts"
    $scriptsDirs = Get-ChildItem -Path $env:APPDATA -Filter "Python" -Directory -ErrorAction SilentlyContinue |
                   Get-ChildItem -Filter "Python3*" -Directory |
                   ForEach-Object { Join-Path $_.FullName "Scripts" }

    if ($scriptsDirs) {
        $scriptsPath = $scriptsDirs[0]
        if ($env:PATH -notlike "*$scriptsPath*") {
            Write-Info "Adding Python Scripts to PATH..."

            # Add to user PATH permanently
            $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
            if ($userPath -notlike "*$scriptsPath*") {
                [Environment]::SetEnvironmentVariable("PATH", "$userPath;$scriptsPath", "User")
                Write-Success "Added $scriptsPath to user PATH"
                Write-Warning "Please restart your terminal for PATH changes to take effect"
            }
        }
    }
}

# Install from source
function Install-FromSource {
    param([string]$PythonCmd)

    Write-Info "Installing Kubix CLI from source..."

    # Check for git
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Error "Git is required for source installation"
        Write-Host "Install Git from: https://git-scm.com/download/win"
        exit 1
    }

    # Create temp directory
    $tempDir = Join-Path $env:TEMP "kubix-cli-install-$(Get-Random)"
    New-Item -ItemType Directory -Path $tempDir | Out-Null

    try {
        Push-Location $tempDir

        # Clone repository
        Write-Info "Cloning repository..."
        git clone "https://github.com/$REPO.git" | Out-Null

        Set-Location "kubix-cli"

        # Install package
        Write-Info "Installing package..."
        & $PythonCmd -m pip install --user -e . | Out-Null

        Write-Success "Kubix CLI installed from source"
    } catch {
        Write-Error "Installation failed: $_"
        exit 1
    } finally {
        Pop-Location
        Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Install in virtual environment
function Install-InVenv {
    param([string]$PythonCmd)

    $venvPath = Join-Path $env:USERPROFILE ".kubix-cli"

    Write-Info "Installing Kubix CLI in virtual environment..."

    # Check if venv exists
    if (Test-Path $venvPath) {
        Write-Warning "Virtual environment already exists at $venvPath"
        $confirm = Read-Host "Remove and recreate? [y/N]"
        if ($confirm -eq 'y' -or $confirm -eq 'Y') {
            Remove-Item -Path $venvPath -Recurse -Force
        } else {
            Write-Info "Using existing virtual environment"
        }
    }

    # Create venv if needed
    if (-not (Test-Path $venvPath)) {
        Write-Info "Creating virtual environment..."
        & $PythonCmd -m venv $venvPath
    }

    # Activate and install
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    & $activateScript

    Write-Info "Installing in virtual environment..."
    & python -m pip install --upgrade pip | Out-Null
    & python -m pip install kubix-cli | Out-Null

    # Create wrapper batch file
    $wrapperPath = Join-Path $env:USERPROFILE "AppData\Local\Microsoft\WindowsApps\kubix.bat"
    $wrapperContent = @"
@echo off
call "$venvPath\Scripts\activate.bat"
python -m kubix_cli %*
"@

    New-Item -ItemType Directory -Path (Split-Path $wrapperPath) -Force -ErrorAction SilentlyContinue | Out-Null
    Set-Content -Path $wrapperPath -Value $wrapperContent

    Write-Success "Kubix CLI installed in virtual environment"
    Write-Info "Wrapper created at $wrapperPath"

    deactivate
}

# Post-installation
function Show-PostInstall {
    Write-Host ""
    Write-Success "Installation complete!"
    Write-Host ""
    Write-Host "Getting Started:" -ForegroundColor Cyan
    Write-Host "  1. Get your FREE API key from: " -NoNewline
    Write-Host "https://openrouter.ai" -ForegroundColor Blue
    Write-Host "  2. Run: " -NoNewline
    Write-Host "kubix" -ForegroundColor Yellow
    Write-Host "  3. Paste your API key when prompted"
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Cyan
    Write-Host "  kubix          - Start the CLI"
    Write-Host "  kubix --help   - Show help"
    Write-Host ""
    Write-Host "Documentation:" -ForegroundColor Cyan
    Write-Host "  https://github.com/$REPO" -ForegroundColor Blue
    Write-Host ""

    # Test if kubix command works
    try {
        $testCmd = & kubix --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Kubix CLI is ready to use!"
        }
    } catch {
        Write-Warning "You may need to restart your terminal for the 'kubix' command to work"
    }
}

# Main installation
function Main {
    Show-Banner

    # Check admin rights (optional)
    if (-not (Test-Administrator)) {
        Write-Warning "Not running as administrator"
        Write-Host "Some features may require administrator privileges"
        Write-Host ""
    }

    # Check Python
    $pythonCmd = Test-Python

    # Check GPU/CUDA
    Test-NvidiaGpu | Out-Null
    Test-Cuda | Out-Null

    # Installation method
    Write-Host ""
    Write-Host "Choose installation method:"
    Write-Host "  1) Install via pip (recommended)"
    Write-Host "  2) Install from source (latest development)"
    Write-Host "  3) Install in virtual environment (isolated)"
    Write-Host ""
    $choice = Read-Host "Select option [1-3]"

    switch ($choice) {
        "1" { Install-ViaPip -PythonCmd $pythonCmd }
        "2" { Install-FromSource -PythonCmd $pythonCmd }
        "3" { Install-InVenv -PythonCmd $pythonCmd }
        default {
            Write-Warning "Invalid option, using pip installation"
            Install-ViaPip -PythonCmd $pythonCmd
        }
    }

    # Show post-install info
    Show-PostInstall
}

# Run main installation
Main