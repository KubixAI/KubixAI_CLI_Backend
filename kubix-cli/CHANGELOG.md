# Changelog

All notable changes to Kubix CLI will be documented in this file.

## [1.0.0] - 2024-01-XX

### Initial Release

**Kubix CLI** - GPU-Native AI Code Editor for CUDA development

### Features

#### Smart AI Agents
- **General Assistant**: Versatile CUDA development helper
- **Optimizer Agent**: Performance optimization specialist
- **Debugger Agent**: Bug detection and fixing expert
- **Analyzer Agent**: Code analysis and explanation

#### Powerful Tools
- File operations (read, write, edit)
- CUDA code analysis
- GPU status monitoring
- Interactive code editing with post-action menu
- Bash command execution

#### Beautiful UI
- NVIDIA-themed interface (#76B900)
- Syntax highlighting for CUDA code
- Interactive command completion
- Real-time tool execution display
- Progress indicators and spinners

#### Configuration System
- Hierarchical config (global → project → local)
- Compiler path setup wizard (`/setup`)
- Model selection and switching
- Permission management
- Safe mode and auto-permission options

#### Developer Experience
- Cross-platform (Windows, Linux, macOS)
- Advanced compiler detection (MSVC, GCC, nvcc)
- Automatic 64-bit compiler selection for CUDA 12+
- Session management and export
- Plugin system for extensibility

### Installation
```bash
pip install kubix-cli
```

### Requirements
- Python 3.9+
- CUDA Toolkit 11.0+ (optional, for GPU features)
- NVIDIA GPU (optional)

### License
Proprietary Non-Commercial License - Free for personal and educational use.
See LICENSE for details.
