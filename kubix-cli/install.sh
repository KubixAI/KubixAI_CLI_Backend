#!/bin/bash
# Kubix CLI Installation Script
# This script installs Kubix CLI on Linux and macOS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REPO="Kubix-AI/kubix-cli"
INSTALL_DIR="$HOME/.local/bin"
VENV_DIR="$HOME/.kubix-cli"

# Functions
print_banner() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║        ${GREEN}Kubix CLI Installer${CYAN}   ║${NC}"
    echo -e "${CYAN}║        GPU-Native AI Code Editor               ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

check_python() {
    print_info "Checking Python installation..."

    # Check for Python 3.9+
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python 3.9+ is required but not found"
        echo "Please install Python from https://python.org"
        exit 1
    fi

    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]); then
        print_error "Python 3.9+ is required (found $PYTHON_VERSION)"
        exit 1
    fi

    print_success "Python $PYTHON_VERSION found"
}

check_cuda() {
    print_info "Checking CUDA installation..."

    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d, -f1)
        print_success "CUDA $CUDA_VERSION found"
    else
        print_warning "CUDA not found (optional for GPU acceleration)"
        echo "    Install CUDA from: https://developer.nvidia.com/cuda-downloads"
    fi
}

install_method_selection() {
    echo ""
    echo "Choose installation method:"
    echo "  1) Install via pip (recommended)"
    echo "  2) Install from source (latest development)"
    echo "  3) Install in virtual environment (isolated)"
    echo ""
    read -p "Select option [1-3]: " choice

    case $choice in
        1) install_via_pip ;;
        2) install_from_source ;;
        3) install_in_venv ;;
        *)
            print_warning "Invalid option, using pip installation"
            install_via_pip
            ;;
    esac
}

install_via_pip() {
    print_info "Installing Kubix CLI via pip..."

    # Ensure pip is up to date
    $PYTHON_CMD -m pip install --upgrade pip

    # Install kubix-cli
    if $PYTHON_CMD -m pip install --user kubix-cli; then
        print_success "Kubix CLI installed successfully"

        # Add to PATH if needed
        if ! echo $PATH | grep -q "$HOME/.local/bin"; then
            print_info "Adding $HOME/.local/bin to PATH..."

            # Detect shell and update appropriate config
            if [ -n "$BASH_VERSION" ]; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
                print_info "Added to ~/.bashrc"
            elif [ -n "$ZSH_VERSION" ]; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
                print_info "Added to ~/.zshrc"
            fi

            print_warning "Please restart your terminal or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
        fi
    else
        print_error "Installation failed"
        exit 1
    fi
}

install_from_source() {
    print_info "Installing Kubix CLI from source..."

    # Check for git
    if ! command -v git &> /dev/null; then
        print_error "Git is required for source installation"
        exit 1
    fi

    # Clone repository
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    print_info "Cloning repository..."
    if git clone https://github.com/$REPO.git; then
        cd kubix-cli

        # Install package
        print_info "Installing package..."
        $PYTHON_CMD -m pip install --user -e .

        print_success "Kubix CLI installed from source"
    else
        print_error "Failed to clone repository"
        exit 1
    fi

    # Cleanup
    rm -rf "$TEMP_DIR"
}

install_in_venv() {
    print_info "Installing Kubix CLI in virtual environment..."

    # Create virtual environment
    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment already exists at $VENV_DIR"
        read -p "Remove and recreate? [y/N]: " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            rm -rf "$VENV_DIR"
        else
            print_info "Using existing virtual environment"
        fi
    fi

    if [ ! -d "$VENV_DIR" ]; then
        print_info "Creating virtual environment..."
        $PYTHON_CMD -m venv "$VENV_DIR"
    fi

    # Activate and install
    source "$VENV_DIR/bin/activate"

    print_info "Installing Kubix CLI..."
    pip install --upgrade pip
    pip install kubix-cli

    # Create wrapper script
    mkdir -p "$INSTALL_DIR"
    cat > "$INSTALL_DIR/kubix" << EOF
#!/bin/bash
source "$VENV_DIR/bin/activate"
exec kubix "\$@"
EOF

    chmod +x "$INSTALL_DIR/kubix"

    print_success "Kubix CLI installed in virtual environment"
    print_info "Wrapper script created at $INSTALL_DIR/kubix"

    deactivate
}

post_install() {
    echo ""
    print_success "Installation complete!"
    echo ""
    echo -e "${CYAN}Getting Started:${NC}"
    echo "  1. Get your FREE API key from: https://openrouter.ai"
    echo "  2. Run: kubix"
    echo "  3. Paste your API key when prompted"
    echo ""
    echo -e "${CYAN}Commands:${NC}"
    echo "  kubix          - Start the CLI"
    echo "  kubix --help   - Show help"
    echo ""
    echo -e "${CYAN}Documentation:${NC}"
    echo "  https://github.com/$REPO"
    echo ""
}

# Main installation flow
main() {
    print_banner

    # Check requirements
    check_python
    check_cuda

    # Install
    install_method_selection

    # Post-install info
    post_install
}

# Run main function
main