"""
Compile CUDA Tool - Production-Ready, Cross-Platform

Features:
- Advanced cross-platform toolchain detection
- Automatic compiler detection
- Platform-specific error messages
- Actionable setup instructions
- Fallback strategies
- Comprehensive error handling
"""

import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any
from .base import Tool, ToolContext, ToolResult
from ..utils.detection import detect_nvcc, detect_cpp_compiler


class CompileCudaTool(Tool):
    """
    Production-ready CUDA compilation tool with advanced detection.

    Features:
    - Cross-platform support (Windows/Linux/macOS)
    - Advanced toolchain detection
    - Detailed error messages
    - Setup instructions
    - Fallback suggestions
    """

    @property
    def name(self) -> str:
        return "compile_cuda"

    @property
    def description(self) -> str:
        return """Compile a CUDA kernel file using nvcc.
Automatically detects your platform and provides setup instructions if needed.
Supports Windows (with Visual Studio), Linux (with g++), and provides alternatives for macOS."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the CUDA file to compile (.cu). REQUIRED. Also accepts 'file_path' for compatibility."
                },
                "file_path": {
                    "type": "string",
                    "description": "Alternative parameter name for filepath (for compatibility)"
                },
                "gpu_arch": {
                    "type": "string",
                    "description": "GPU architecture (e.g., sm_70, sm_75, sm_80, sm_86). Optional, defaults to sm_75.",
                    "default": "sm_75"
                },
                "output": {
                    "type": "string",
                    "description": "Output binary name. Optional, defaults to <filename> (no extension on Linux/macOS, .exe on Windows)",
                    "default": None
                },
                "optimization": {
                    "type": "string",
                    "description": "Optimization level: O0, O1, O2, O3. Optional, defaults to O2.",
                    "default": "O2"
                }
            },
            "required": []  # Either 'filepath' or 'file_path' is required, checked in execute()
        }

    async def execute(self, params: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Compile CUDA file with comprehensive error handling."""

        # Get parameters - support both 'filepath' and 'file_path' for compatibility
        filepath = params.get("filepath") or params.get("file_path")

        if not filepath:
            return ToolResult(
                title="Missing parameter",
                output="",
                error="Required parameter 'filepath' is missing. Please provide the path to the CUDA file to compile.",
                metadata={"success": False}
            )

        gpu_arch = params.get("gpu_arch", "sm_75")
        output = params.get("output")
        optimization = params.get("optimization", "O2")

        # Resolve path
        if not Path(filepath).is_absolute():
            filepath = Path(ctx.working_dir) / filepath
        filepath = Path(filepath)

        # Check if file exists
        if not filepath.exists():
            return ToolResult(
                title=f"File not found: {filepath.name}",
                output="",
                error=f"CUDA file does not exist: {filepath}\n\nPlease verify the file path."
            )

        # Detect platform
        platform_name = platform.system()

        # Check for nvcc using advanced detection
        nvcc_info = detect_nvcc()
        if not nvcc_info.available:
            return ToolResult(
                title="nvcc not found",
                output="",
                error=f"nvcc (CUDA compiler) not found.\n\n{nvcc_info.error}",
                metadata={
                    "platform": platform_name,
                    "missing": "nvcc",
                    "success": False
                }
            )

        # Skip compiler checks if environment variable is set
        skip_checks = os.getenv("KUBIX_SKIP_COMPILER_CHECKS", "").lower() in ("1", "true", "yes")

        # Platform-specific compiler checks using advanced detection (can be skipped)
        if not skip_checks:
            compiler_info = detect_cpp_compiler()
            if not compiler_info.available:
                # On Windows, provide specific instructions for VS setup
                if platform_name == "Windows":
                    error_msg = (
                        "Visual Studio C++ compiler not detected.\n\n"
                        "✓ Kubix CLI automatically detects Visual Studio 2017-2022\n"
                        "✓ Supports Community, Professional, Enterprise, and BuildTools editions\n"
                        "✗ No valid installation found on your system\n\n"
                        "**Quick Fix:**\n"
                        "1. Install Visual Studio from: https://visualstudio.microsoft.com/downloads/\n"
                        "2. During installation, select 'Desktop development with C++' workload\n"
                        "3. Restart Kubix CLI - it will auto-detect the new installation\n\n"
                        "**Alternative:** Skip compiler checks (not recommended):\n"
                        "  Set environment variable: KUBIX_SKIP_COMPILER_CHECKS=1"
                    )
                else:
                    error_msg = f"C++ compiler not found.\n\n{compiler_info.error}\n\nOr set: KUBIX_SKIP_COMPILER_CHECKS=1"

                return ToolResult(
                    title="C++ compiler not found",
                    output="",
                    error=error_msg,
                    metadata={
                        "platform": platform_name,
                        "missing": "cpp_compiler",
                        "nvcc_version": nvcc_info.version,
                        "success": False,
                        "auto_detect_enabled": True
                    }
                )

        # Determine output file
        if not output:
            if platform_name == "Windows":
                output = filepath.stem + ".exe"
            else:
                output = filepath.stem
        else:
            output = Path(output)
            if not output.is_absolute():
                output = Path(ctx.working_dir) / output

        # Build nvcc command using detected nvcc path (if available)
        nvcc_cmd = nvcc_info.path if nvcc_info.path else "nvcc"

        cmd = [
            nvcc_cmd,
            str(filepath),
            f"-arch={gpu_arch}",
            f"-{optimization}",
            "-o", str(output),
        ]

        # CRITICAL FIX: On Windows, explicitly set compiler path with -ccbin to avoid x86/x64 mismatch
        # This prevents cudafe++ ACCESS_VIOLATION errors caused by nvcc finding wrong cl.exe
        ccbin_path = None
        if platform_name == "Windows" and not skip_checks:
            # Detect compiler and get its directory for -ccbin flag
            from ..utils.detection import ToolchainDetector
            temp_detector = ToolchainDetector()
            temp_compiler_info = temp_detector.detect_cpp_compiler()

            if temp_compiler_info.available and temp_compiler_info.path:
                # Get the directory containing cl.exe (not the cl.exe itself)
                cl_dir = str(Path(temp_compiler_info.path).parent)

                # Ensure it's x64 (CUDA 12+ requirement)
                if 'x86' in cl_dir.lower() and 'x64' not in cl_dir.lower():
                    # Auto-correct to x64
                    x64_dir = cl_dir.replace('x86', 'x64').replace('X86', 'X64').replace('Hostx86', 'Hostx64')
                    if Path(x64_dir, 'cl.exe').exists():
                        cl_dir = x64_dir

                # Add -ccbin flag to explicitly specify compiler
                # This is the MOST ROBUST solution - tells nvcc exactly which compiler to use
                cmd.insert(2, f"-ccbin={cl_dir}")  # Insert after nvcc and filepath
                ccbin_path = cl_dir

        # Platform-specific flags
        if platform_name == "Windows":
            # Suppress deprecated GPU target warnings on Windows
            cmd.append("-Wno-deprecated-gpu-targets")

        # Add line info for debugging
        cmd.append("-lineinfo")

        # Prepare environment for compilation
        # On Windows, use MSVC environment if available
        compilation_env = os.environ.copy()

        if platform_name == "Windows" and not skip_checks:
            # Get MSVC environment variables for compilation
            # IMPORTANT: Create fresh detector instance to avoid cache issues
            from ..utils.detection import ToolchainDetector
            detector = ToolchainDetector()  # Fresh instance, no cache

            compiler_info = detector.detect_cpp_compiler()
            if compiler_info.available and compiler_info.env_vars:
                # IMPORTANT: Verify we're using x64 compiler (CUDA 12+ requirement)
                # Using x86 cl.exe causes cudafe++ ACCESS_VIOLATION error
                cl_path_str = str(compiler_info.path).lower()
                if 'x86' in cl_path_str and 'x64' not in cl_path_str:
                    # Try to find x64 version
                    x64_path = str(compiler_info.path).replace('x86', 'x64').replace('X86', 'X64')
                    x64_path = x64_path.replace('Hostx86', 'Hostx64')
                    if Path(x64_path).exists():
                        compiler_info.path = x64_path
                        # Update environment with x64 path
                        cl_dir = str(Path(x64_path).parent)
                        compilation_env['PATH'] = f"{cl_dir};{compilation_env.get('PATH', '')}"

                # Merge MSVC environment variables
                compilation_env.update(compiler_info.env_vars)

                # Ensure cl.exe directory is in PATH
                cl_dir = str(Path(compiler_info.path).parent)
                if cl_dir not in compilation_env.get('PATH', ''):
                    compilation_env['PATH'] = f"{cl_dir};{compilation_env.get('PATH', '')}"
            else:
                # Fallback: try to get environment from detector
                msvc_env = detector.get_msvc_environment()
                if msvc_env:
                    compilation_env.update(msvc_env)

        try:
            # Run nvcc with proper environment
            result = subprocess.run(
                cmd,
                cwd=str(filepath.parent),
                capture_output=True,
                text=True,
                timeout=120,  # 2 minutes for large files
                env=compilation_env  # Use configured environment
            )

            # Check result
            if result.returncode == 0:
                # Success
                warnings = []
                if result.stderr:
                    # Filter out known harmless warnings
                    for line in result.stderr.split('\n'):
                        line = line.strip()
                        if line and 'deprecated-gpu-targets' not in line:
                            warnings.append(line)

                output_text = f"✓ Compiled successfully\nOutput: {output}\nArchitecture: {gpu_arch}\nOptimization: {optimization}"

                # Show compiler used (helpful for debugging)
                if ccbin_path:
                    output_text += f"\nCompiler: {ccbin_path}"

                if warnings:
                    output_text += f"\n\nWarnings:\n" + "\n".join(warnings)

                return ToolResult(
                    title=f"Compiled {filepath.name}",
                    output=output_text,
                    metadata={
                        "filepath": str(filepath),
                        "output": str(output),
                        "gpu_arch": gpu_arch,
                        "optimization": optimization,
                        "platform": platform_name,
                        "nvcc_version": nvcc_info.version,
                        "compiler_path": ccbin_path,
                        "success": True,
                        "warnings": warnings if warnings else None
                    }
                )

            else:
                # Compilation failed
                error_output = result.stderr if result.stderr else result.stdout

                # Try to extract specific error messages
                error_lines = []
                for line in error_output.split('\n'):
                    if 'error' in line.lower() or 'fatal' in line.lower():
                        error_lines.append(line.strip())

                if error_lines:
                    summary = "\n".join(error_lines[:5])  # First 5 errors
                    if len(error_lines) > 5:
                        summary += f"\n... and {len(error_lines) - 5} more errors"
                else:
                    summary = error_output[:500]  # First 500 chars

                return ToolResult(
                    title=f"Compilation failed: {filepath.name}",
                    output="",
                    error=f"Compilation errors:\n\n{summary}\n\nFull output:\n{error_output}",
                    metadata={
                        "filepath": str(filepath),
                        "gpu_arch": gpu_arch,
                        "platform": platform_name,
                        "success": False,
                        "exit_code": result.returncode,
                        "error_count": len(error_lines)
                    }
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                title=f"Compilation timeout: {filepath.name}",
                output="",
                error="Compilation took longer than 2 minutes. The file may be too large or complex.",
                metadata={
                    "filepath": str(filepath),
                    "success": False,
                    "timeout": True
                }
            )

        except Exception as e:
            return ToolResult(
                title=f"Error compiling {filepath.name}",
                output="",
                error=f"Unexpected error: {str(e)}\n\nPlatform: {platform_name}\nIf this persists, please report this issue.",
                metadata={
                    "filepath": str(filepath),
                    "platform": platform_name,
                    "success": False,
                    "exception": str(e)
                }
            )
