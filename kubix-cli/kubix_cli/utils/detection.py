"""
Advanced Cross-Platform Toolchain Detection System

Detects CUDA toolchain components with maximum reliability:
- NVCC (CUDA compiler)
- C/C++ compilers (cl.exe, gcc, clang)
- NCU (Nsight Compute profiler)
- NVPROF (legacy profiler)
- NSYS (Nsight Systems)

Features:
- Cross-platform (Windows, Linux, macOS)
- PATH search + common install locations
- Environment variable detection
- Version extraction
- Result caching for performance
- Comprehensive error messages
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import shutil


@dataclass
class ToolInfo:
    """Information about a detected tool."""
    available: bool
    path: Optional[str] = None
    version: Optional[str] = None
    error: Optional[str] = None
    env_vars: Optional[Dict[str, str]] = None  # Environment variables for compilation


class ToolchainDetector:
    """
    Advanced cross-platform toolchain detection.

    Caches results for performance.
    Checks multiple locations for maximum reliability.
    """

    def __init__(self, config_paths: Optional[Dict[str, str]] = None):
        """
        Initialize detector with optional configured compiler paths.

        Args:
            config_paths: Optional dict of configured paths (e.g., {'nvcc': '/path/to/nvcc', 'cl': '/path/to/cl.exe'})
        """
        self._cache: Dict[str, ToolInfo] = {}
        self.platform = platform.system()
        self.config_paths = config_paths or {}

    # ============================================================================
    # NVCC Detection - CUDA Compiler
    # ============================================================================

    def detect_nvcc(self) -> ToolInfo:
        """
        Detect NVCC (CUDA compiler) with advanced cross-platform support.

        Checks:
        1. Configured path (from config)
        2. PATH environment variable
        3. CUDA_PATH environment variable
        4. Common installation locations
        5. Extracts version information

        Returns:
            ToolInfo with availability, path, version
        """
        if 'nvcc' in self._cache:
            return self._cache['nvcc']

        # Check configured path first (highest priority)
        if 'nvcc' in self.config_paths and self.config_paths['nvcc']:
            nvcc_configured = Path(self.config_paths['nvcc'])
            if nvcc_configured.exists():
                version = self._get_nvcc_version(str(nvcc_configured))
                info = ToolInfo(
                    available=True,
                    path=str(nvcc_configured),
                    version=version
                )
                self._cache['nvcc'] = info
                return info

        # Check PATH (fast auto-detection)
        nvcc_path = shutil.which('nvcc')

        if nvcc_path:
            version = self._get_nvcc_version(nvcc_path)
            info = ToolInfo(
                available=True,
                path=nvcc_path,
                version=version
            )
            self._cache['nvcc'] = info
            return info

        # Check CUDA_PATH environment variable
        cuda_path = os.getenv('CUDA_PATH')
        if cuda_path:
            nvcc_bin = Path(cuda_path) / 'bin' / ('nvcc.exe' if self.platform == 'Windows' else 'nvcc')
            if nvcc_bin.exists():
                version = self._get_nvcc_version(str(nvcc_bin))
                info = ToolInfo(
                    available=True,
                    path=str(nvcc_bin),
                    version=version
                )
                self._cache['nvcc'] = info
                return info

        # Check common installation locations
        common_paths = self._get_cuda_common_paths()

        for base_path in common_paths:
            nvcc_bin = base_path / 'bin' / ('nvcc.exe' if self.platform == 'Windows' else 'nvcc')
            if nvcc_bin.exists():
                version = self._get_nvcc_version(str(nvcc_bin))
                info = ToolInfo(
                    available=True,
                    path=str(nvcc_bin),
                    version=version
                )
                self._cache['nvcc'] = info
                return info

        # Not found
        error = self._get_nvcc_install_hint()
        info = ToolInfo(
            available=False,
            error=error
        )
        self._cache['nvcc'] = info
        return info

    def _get_nvcc_version(self, nvcc_path: str) -> Optional[str]:
        """Extract NVCC version."""
        try:
            result = subprocess.run(
                [nvcc_path, '--version'],
                capture_output=True,
                text=True,
                timeout=3
            )

            if result.returncode == 0:
                # Parse version from output
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        parts = line.split('release')
                        if len(parts) > 1:
                            version = parts[1].split(',')[0].strip()
                            return version

            return "Unknown"
        except:
            return None

    def _get_cuda_common_paths(self) -> List[Path]:
        """Get common CUDA installation paths by platform."""
        if self.platform == 'Windows':
            # Windows: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X
            base = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA')
            if base.exists():
                # Find all version directories
                return sorted([p for p in base.iterdir() if p.is_dir()], reverse=True)
            return []

        elif self.platform == 'Linux':
            # Linux: /usr/local/cuda, /usr/local/cuda-X.X, /opt/cuda
            paths = [
                Path('/usr/local/cuda'),
                Path('/opt/cuda'),
            ]

            # Add versioned paths
            local_base = Path('/usr/local')
            if local_base.exists():
                cuda_dirs = sorted(
                    [p for p in local_base.iterdir() if p.is_dir() and p.name.startswith('cuda-')],
                    reverse=True
                )
                paths.extend(cuda_dirs)

            return [p for p in paths if p.exists()]

        elif self.platform == 'Darwin':  # macOS
            # macOS: /Developer/NVIDIA/CUDA-X.X, /usr/local/cuda
            paths = [
                Path('/usr/local/cuda'),
            ]

            dev_base = Path('/Developer/NVIDIA')
            if dev_base.exists():
                cuda_dirs = sorted(
                    [p for p in dev_base.iterdir() if p.is_dir() and p.name.startswith('CUDA-')],
                    reverse=True
                )
                paths.extend(cuda_dirs)

            return [p for p in paths if p.exists()]

        return []

    def _get_nvcc_install_hint(self) -> str:
        """Get platform-specific installation hint for NVCC."""
        if self.platform == 'Windows':
            return "Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
        elif self.platform == 'Linux':
            return "Install: sudo apt install nvidia-cuda-toolkit  (or download from nvidia.com/cuda-downloads)"
        elif self.platform == 'Darwin':
            return "Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
        return "Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"

    # ============================================================================
    # C/C++ Compiler Detection
    # ============================================================================

    def detect_cpp_compiler(self) -> ToolInfo:
        """
        Detect C/C++ compiler (cl.exe, gcc, clang).

        Windows: cl.exe (MSVC)
        Linux: gcc/g++ or clang
        macOS: clang (Xcode Command Line Tools)

        Returns:
            ToolInfo with availability, path, version
        """
        if 'cpp_compiler' in self._cache:
            return self._cache['cpp_compiler']

        if self.platform == 'Windows':
            return self._detect_msvc()
        else:
            return self._detect_gcc_clang()

    def _detect_msvc(self) -> ToolInfo:
        """
        Detect MSVC (cl.exe) on Windows with automatic environment setup.

        This method is rock-solid and works globally:
        - Checks configured path first
        - Automatically finds all Visual Studio installations (2017-2022)
        - Sets up environment variables programmatically
        - No manual user intervention needed
        - Caches results for performance
        """
        # Check configured path first (highest priority)
        if 'cl' in self.config_paths and self.config_paths['cl']:
            cl_configured = Path(self.config_paths['cl'])
            if cl_configured.exists():
                version = self._get_msvc_version(str(cl_configured))
                info = ToolInfo(
                    available=True,
                    path=str(cl_configured),
                    version=version
                )
                self._cache['cpp_compiler'] = info
                return info

        # Check PATH (fast - already configured environment)
        cl_path = shutil.which('cl')

        if cl_path:
            version = self._get_msvc_version(cl_path)
            info = ToolInfo(
                available=True,
                path=cl_path,
                version=version
            )
            # IMPORTANT: If cl.exe is in PATH, capture current environment
            # This ensures LIB, INCLUDE, and other MSVC variables are available
            info.env_vars = os.environ.copy()
            self._cache['cpp_compiler'] = info
            return info

        # AUTOMATIC DETECTION - Find Visual Studio installations
        vs_installation = self._find_visual_studio_installation()

        if vs_installation:
            # Found VS - automatically set up environment
            cl_path, env_vars = self._setup_msvc_environment(vs_installation)

            if cl_path:
                version = self._get_msvc_version(cl_path)
                info = ToolInfo(
                    available=True,
                    path=cl_path,
                    version=version
                )
                # Store environment variables for compilation
                info.env_vars = env_vars
                self._cache['cpp_compiler'] = info
                return info

        # Not found - provide installation guidance
        error = (
            "Visual Studio with C++ support not found.\n"
            "Install from: https://visualstudio.microsoft.com/downloads/\n"
            "Required: 'Desktop development with C++' workload"
        )
        info = ToolInfo(
            available=False,
            error=error
        )
        self._cache['cpp_compiler'] = info
        return info

    def _find_visual_studio_installation(self) -> Optional[Dict[str, str]]:
        """
        Find Visual Studio installation automatically.

        Searches:
        - VS 2022 (all editions: Community, Professional, Enterprise)
        - VS 2019 (all editions)
        - VS 2017 (all editions)
        - Both Program Files and Program Files (x86)

        Returns:
            Dict with 'root', 'version', 'edition', 'vcvars_path' or None
        """
        # Search paths for Visual Studio (2017-2022)
        search_bases = [
            Path('C:/Program Files/Microsoft Visual Studio'),
            Path('C:/Program Files (x86)/Microsoft Visual Studio'),
        ]

        # Versions to search (newest first)
        versions = ['2022', '2019', '2017']

        # Editions (in priority order)
        editions = ['Enterprise', 'Professional', 'Community', 'BuildTools']

        for base in search_bases:
            if not base.exists():
                continue

            for version in versions:
                for edition in editions:
                    vs_root = base / version / edition

                    if not vs_root.exists():
                        continue

                    # Check for vcvarsall.bat (required for setup)
                    vcvars = vs_root / 'VC' / 'Auxiliary' / 'Build' / 'vcvars64.bat'

                    if vcvars.exists():
                        # Found a valid installation
                        return {
                            'root': str(vs_root),
                            'version': version,
                            'edition': edition,
                            'vcvars_path': str(vcvars)
                        }

        return None

    def _setup_msvc_environment(self, vs_install: Dict[str, str]) -> Tuple[Optional[str], Dict[str, str]]:
        """
        Automatically set up MSVC environment variables.

        This is the magic that makes it work without Developer Command Prompt!

        Args:
            vs_install: Visual Studio installation info

        Returns:
            Tuple of (cl.exe path, environment variables dict)
        """
        try:
            # Run vcvars64.bat and capture environment
            vcvars_path = vs_install['vcvars_path']

            # Create a batch script that runs vcvars and outputs environment
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.bat', delete=False) as f:
                f.write(f'@echo off\n')
                f.write(f'call "{vcvars_path}" >nul 2>&1\n')
                f.write('set\n')  # Output all environment variables
                temp_script = f.name

            try:
                # Run the script and capture environment
                result = subprocess.run(
                    [temp_script],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    shell=True
                )

                if result.returncode == 0:
                    # Parse environment variables
                    env_vars = {}
                    for line in result.stdout.split('\n'):
                        line = line.strip()
                        if '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key] = value

                    # Find cl.exe in the new PATH
                    if 'PATH' in env_vars:
                        for path_entry in env_vars['PATH'].split(';'):
                            cl_candidate = Path(path_entry) / 'cl.exe'
                            if cl_candidate.exists():
                                return str(cl_candidate), env_vars

                    # Fallback: construct cl.exe path from VS root
                    # Try multiple architectures in priority order
                    vs_root = Path(vs_install['root'])
                    msvc_path = vs_root / 'VC' / 'Tools' / 'MSVC'
                    if msvc_path.exists():
                        # Find latest MSVC version
                        msvc_versions = sorted([p for p in msvc_path.iterdir() if p.is_dir()], reverse=True)
                        if msvc_versions:
                            # Check multiple architecture combinations
                            # IMPORTANT: CUDA 12.x requires 64-bit compiler
                            # Using 32-bit cl.exe causes cudafe++ ACCESS_VIOLATION (0xC0000005)
                            arch_combinations = [
                                ('Hostx64', 'x64'),  # 64-bit native (REQUIRED for modern CUDA)
                                ('HostX64', 'x64'),  # Alternative capitalization
                            ]

                            for host_arch, target_arch in arch_combinations:
                                cl_path = msvc_versions[0] / 'bin' / host_arch / target_arch / 'cl.exe'
                                if cl_path.exists():
                                    # If env_vars is empty (vcvars failed), manually construct minimum environment
                                    if not env_vars or 'LIB' not in env_vars:
                                        env_vars = self._construct_msvc_environment(vs_root, msvc_versions[0], cl_path.parent)
                                    return str(cl_path), env_vars

            finally:
                # Clean up temp script
                try:
                    Path(temp_script).unlink()
                except:
                    pass

        except Exception as e:
            # Setup failed - will return None
            pass

        return None, {}

    def _construct_msvc_environment(self, vs_root: Path, msvc_version_path: Path, cl_bin_path: Path) -> Dict[str, str]:
        """
        Manually construct MSVC environment variables when vcvars fails.

        Args:
            vs_root: Visual Studio installation root
            msvc_version_path: Path to specific MSVC version (e.g., 14.44.35207)
            cl_bin_path: Path to cl.exe directory

        Returns:
            Dict of environment variables
        """
        import os
        env = os.environ.copy()

        # IMPORTANT: Ensure we're using x64 architecture (CUDA 12+ requirement)
        # Using x86 cl.exe with 64-bit nvcc causes cudafe++ ACCESS_VIOLATION
        cl_dir = str(cl_bin_path)
        if 'x86' in cl_dir.lower() and 'x64' not in cl_dir.lower():
            # Try to switch to x64 version
            x64_dir = cl_dir.replace('x86', 'x64').replace('X86', 'X64').replace('Hostx86', 'Hostx64')
            if Path(x64_dir, 'cl.exe').exists():
                cl_dir = x64_dir

        # Add cl.exe directory to PATH
        if 'PATH' in env:
            env['PATH'] = f"{cl_dir};{env['PATH']}"
        else:
            env['PATH'] = cl_dir

        # Add MSVC lib paths (x64 ONLY)
        lib_paths = [
            msvc_version_path / 'lib' / 'x64',
            msvc_version_path / 'atlmfc' / 'lib' / 'x64',
        ]

        # Add Windows SDK lib paths (try common locations)
        win_sdk_base = Path('C:/Program Files (x86)/Windows Kits/10')
        if win_sdk_base.exists():
            lib_dir = win_sdk_base / 'Lib'
            if lib_dir.exists():
                # Find latest SDK version
                sdk_versions = sorted([p for p in lib_dir.iterdir() if p.is_dir()], reverse=True)
                if sdk_versions:
                    lib_paths.extend([
                        sdk_versions[0] / 'um' / 'x64',
                        sdk_versions[0] / 'ucrt' / 'x64',
                    ])

        env['LIB'] = ';'.join(str(p) for p in lib_paths if p.exists())

        # Add MSVC include paths
        include_paths = [
            msvc_version_path / 'include',
            msvc_version_path / 'atlmfc' / 'include',
        ]

        # Add Windows SDK include paths
        if win_sdk_base.exists():
            inc_dir = win_sdk_base / 'Include'
            if inc_dir.exists():
                sdk_versions = sorted([p for p in inc_dir.iterdir() if p.is_dir()], reverse=True)
                if sdk_versions:
                    include_paths.extend([
                        sdk_versions[0] / 'um',
                        sdk_versions[0] / 'ucrt',
                        sdk_versions[0] / 'shared',
                    ])

        env['INCLUDE'] = ';'.join(str(p) for p in include_paths if p.exists())

        return env

    def get_msvc_environment(self) -> Dict[str, str]:
        """
        Get MSVC environment variables for compilation.

        Returns:
            Dict of environment variables to use for compilation
        """
        compiler_info = self.detect_cpp_compiler()

        if hasattr(compiler_info, 'env_vars') and compiler_info.env_vars:
            return compiler_info.env_vars

        return {}

    def _detect_gcc_clang(self) -> ToolInfo:
        """Detect gcc/g++ or clang on Linux/macOS."""
        # Try g++ first (common for CUDA)
        compilers = ['g++', 'gcc', 'clang++', 'clang']

        for compiler in compilers:
            path = shutil.which(compiler)
            if path:
                version = self._get_gcc_version(path)
                info = ToolInfo(
                    available=True,
                    path=path,
                    version=version
                )
                self._cache['cpp_compiler'] = info
                return info

        # Not found
        if self.platform == 'Linux':
            error = "Install: sudo apt install build-essential  (or equivalent for your distro)"
        elif self.platform == 'Darwin':
            error = "Install: xcode-select --install"
        else:
            error = "Install a C++ compiler (gcc, clang)"

        info = ToolInfo(
            available=False,
            error=error
        )
        self._cache['cpp_compiler'] = info
        return info

    def _get_msvc_version(self, cl_path: str) -> Optional[str]:
        """Extract MSVC version."""
        try:
            result = subprocess.run(
                [cl_path],
                capture_output=True,
                text=True,
                timeout=3
            )

            # cl.exe prints version to stderr
            output = result.stderr + result.stdout

            for line in output.split('\n'):
                if 'Version' in line:
                    parts = line.split('Version')
                    if len(parts) > 1:
                        version = parts[1].strip().split()[0]
                        return version

            return "Unknown"
        except:
            return None

    def _get_gcc_version(self, gcc_path: str) -> Optional[str]:
        """Extract GCC/Clang version."""
        try:
            result = subprocess.run(
                [gcc_path, '--version'],
                capture_output=True,
                text=True,
                timeout=3
            )

            if result.returncode == 0:
                # First line usually has version
                first_line = result.stdout.split('\n')[0]
                # Extract version number (e.g., "11.4.0")
                import re
                match = re.search(r'(\d+\.\d+\.\d+)', first_line)
                if match:
                    return match.group(1)

                # Fallback: just return first line
                return first_line.strip()

            return "Unknown"
        except:
            return None

    # ============================================================================
    # NCU Detection - NVIDIA Nsight Compute
    # ============================================================================

    def detect_ncu(self) -> ToolInfo:
        """
        Detect NCU (NVIDIA Nsight Compute profiler).

        Checks:
        1. Configured path (from config)
        2. PATH
        3. CUDA installation directories
        4. Common Nsight Compute installation paths

        Returns:
            ToolInfo with availability, path, version
        """
        if 'ncu' in self._cache:
            return self._cache['ncu']

        # Check configured path first (highest priority)
        if 'ncu' in self.config_paths and self.config_paths['ncu']:
            ncu_configured = Path(self.config_paths['ncu'])
            if ncu_configured.exists():
                version = self._get_tool_version(str(ncu_configured), ['--version'])
                info = ToolInfo(
                    available=True,
                    path=str(ncu_configured),
                    version=version
                )
                self._cache['ncu'] = info
                return info

        # Check PATH
        ncu_path = shutil.which('ncu')

        if ncu_path:
            version = self._get_tool_version(ncu_path, ['--version'])
            info = ToolInfo(
                available=True,
                path=ncu_path,
                version=version
            )
            self._cache['ncu'] = info
            return info

        # Check CUDA directories
        cuda_paths = self._get_cuda_common_paths()

        for cuda_path in cuda_paths:
            # NCU is in extras/
            ncu_paths = [
                cuda_path / 'nsight-compute' / ('ncu.exe' if self.platform == 'Windows' else 'ncu'),
                cuda_path / 'extras' / 'nsight-compute' / ('ncu.exe' if self.platform == 'Windows' else 'ncu'),
            ]

            for ncu_bin in ncu_paths:
                if ncu_bin.exists():
                    version = self._get_tool_version(str(ncu_bin), ['--version'])
                    info = ToolInfo(
                        available=True,
                        path=str(ncu_bin),
                        version=version
                    )
                    self._cache['ncu'] = info
                    return info

        # Check standalone Nsight Compute installation
        if self.platform == 'Windows':
            nsight_paths = [
                Path('C:/Program Files/NVIDIA Corporation/Nsight Compute'),
            ]
        elif self.platform == 'Linux':
            nsight_paths = [
                Path('/opt/nvidia/nsight-compute'),
                Path(f'{os.path.expanduser("~")}/NVIDIA-Nsight-Compute'),
            ]
        else:  # macOS
            nsight_paths = [
                Path('/Applications/NVIDIA Nsight Compute.app/Contents/MacOS'),
            ]

        for nsight_path in nsight_paths:
            if nsight_path.exists():
                # Find latest version
                versions = sorted([p for p in nsight_path.iterdir() if p.is_dir()], reverse=True)
                if versions:
                    ncu_bin = versions[0] / ('ncu.exe' if self.platform == 'Windows' else 'ncu')
                    if ncu_bin.exists():
                        version = self._get_tool_version(str(ncu_bin), ['--version'])
                        info = ToolInfo(
                            available=True,
                            path=str(ncu_bin),
                            version=version
                        )
                        self._cache['ncu'] = info
                        return info

        # Not found
        error = "Install: NVIDIA Nsight Compute from https://developer.nvidia.com/nsight-compute"
        info = ToolInfo(
            available=False,
            error=error
        )
        self._cache['ncu'] = info
        return info

    # ============================================================================
    # NSYS Detection - NVIDIA Nsight Systems
    # ============================================================================

    def detect_nsys(self) -> ToolInfo:
        """
        Detect NSYS (NVIDIA Nsight Systems profiler).

        Checks:
        1. Configured path (from config)
        2. PATH
        3. CUDA installation directories

        Returns:
            ToolInfo with availability, path, version
        """
        if 'nsys' in self._cache:
            return self._cache['nsys']

        # Check configured path first (highest priority)
        if 'nsys' in self.config_paths and self.config_paths['nsys']:
            nsys_configured = Path(self.config_paths['nsys'])
            if nsys_configured.exists():
                version = self._get_tool_version(str(nsys_configured), ['--version'])
                info = ToolInfo(
                    available=True,
                    path=str(nsys_configured),
                    version=version
                )
                self._cache['nsys'] = info
                return info

        # Check PATH
        nsys_path = shutil.which('nsys')

        if nsys_path:
            version = self._get_tool_version(nsys_path, ['--version'])
            info = ToolInfo(
                available=True,
                path=nsys_path,
                version=version
            )
            self._cache['nsys'] = info
            return info

        # Check CUDA directories
        cuda_paths = self._get_cuda_common_paths()

        for cuda_path in cuda_paths:
            nsys_paths = [
                cuda_path / 'nsight-systems' / 'bin' / ('nsys.exe' if self.platform == 'Windows' else 'nsys'),
                cuda_path / 'bin' / ('nsys.exe' if self.platform == 'Windows' else 'nsys'),
            ]

            for nsys_bin in nsys_paths:
                if nsys_bin.exists():
                    version = self._get_tool_version(str(nsys_bin), ['--version'])
                    info = ToolInfo(
                        available=True,
                        path=str(nsys_bin),
                        version=version
                    )
                    self._cache['nsys'] = info
                    return info

        # Not found
        error = "Install: NVIDIA Nsight Systems from https://developer.nvidia.com/nsight-systems"
        info = ToolInfo(
            available=False,
            error=error
        )
        self._cache['nsys'] = info
        return info

    # ============================================================================
    # NVPROF Detection - Legacy Profiler
    # ============================================================================

    def detect_nvprof(self) -> ToolInfo:
        """
        Detect NVPROF (legacy CUDA profiler).

        Returns:
            ToolInfo with availability, path, version
        """
        if 'nvprof' in self._cache:
            return self._cache['nvprof']

        # Check PATH
        nvprof_path = shutil.which('nvprof')

        if nvprof_path:
            version = self._get_tool_version(nvprof_path, ['--version'])
            info = ToolInfo(
                available=True,
                path=nvprof_path,
                version=version
            )
            self._cache['nvprof'] = info
            return info

        # Check CUDA directories
        cuda_paths = self._get_cuda_common_paths()

        for cuda_path in cuda_paths:
            nvprof_bin = cuda_path / 'bin' / ('nvprof.exe' if self.platform == 'Windows' else 'nvprof')

            if nvprof_bin.exists():
                version = self._get_tool_version(str(nvprof_bin), ['--version'])
                info = ToolInfo(
                    available=True,
                    path=str(nvprof_bin),
                    version=version
                )
                self._cache['nvprof'] = info
                return info

        # Not found (but this is OK - it's deprecated)
        error = "nvprof is deprecated. Use nsys (Nsight Systems) instead."
        info = ToolInfo(
            available=False,
            error=error
        )
        self._cache['nvprof'] = info
        return info

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _get_tool_version(self, tool_path: str, version_args: List[str]) -> Optional[str]:
        """
        Generic version extraction.

        Args:
            tool_path: Path to tool
            version_args: Arguments to get version (e.g., ['--version'])

        Returns:
            Version string or None
        """
        try:
            result = subprocess.run(
                [tool_path] + version_args,
                capture_output=True,
                text=True,
                timeout=3
            )

            if result.returncode == 0:
                # Extract first line (usually has version)
                output = result.stdout + result.stderr
                first_line = output.strip().split('\n')[0]
                return first_line[:100]  # Limit length

            return "Unknown"
        except:
            return None

    def clear_cache(self):
        """Clear detection cache (force re-detection)."""
        self._cache.clear()

    def get_all_tools(self) -> Dict[str, ToolInfo]:
        """
        Detect all tools and return status.

        Returns:
            Dict mapping tool name to ToolInfo
        """
        return {
            'nvcc': self.detect_nvcc(),
            'cpp_compiler': self.detect_cpp_compiler(),
            'ncu': self.detect_ncu(),
            'nsys': self.detect_nsys(),
            'nvprof': self.detect_nvprof(),
        }

    def print_status(self):
        """Print detection status (for debugging)."""
        tools = self.get_all_tools()

        print(f"\n{'='*70}")
        print(f"  CUDA Toolchain Detection - {self.platform}")
        print(f"{'='*70}\n")

        for name, info in tools.items():
            # Use ASCII-safe characters for Windows compatibility
            status = '[OK]' if info.available else '[NO]'
            color = '\033[32m' if info.available else '\033[31m'
            reset = '\033[0m'

            print(f"{color}{status}{reset} {name.upper():<15}", end='')

            if info.available:
                print(f" {info.version or 'Unknown version'}")
                print(f"{'':20} {info.path}")
            else:
                print(f" Not found")
                if info.error:
                    print(f"{'':20} {info.error}")

            print()


# ============================================================================
# Global Singleton Instance
# ============================================================================

_detector: Optional[ToolchainDetector] = None
_detector_config_paths: Optional[Dict[str, str]] = None


def get_detector(config_paths: Optional[Dict[str, str]] = None) -> ToolchainDetector:
    """
    Get global detector instance (singleton).

    If config_paths are provided and different from cached paths,
    creates a new detector with updated configuration.

    Args:
        config_paths: Optional dict of configured compiler paths

    Returns:
        ToolchainDetector instance
    """
    global _detector, _detector_config_paths

    # Check if we need to update the detector
    if config_paths != _detector_config_paths or _detector is None:
        _detector = ToolchainDetector(config_paths)
        _detector_config_paths = config_paths

    return _detector


# ============================================================================
# Convenience Functions
# ============================================================================

def detect_nvcc() -> ToolInfo:
    """Convenience: Detect NVCC."""
    return get_detector().detect_nvcc()


def detect_cpp_compiler() -> ToolInfo:
    """Convenience: Detect C++ compiler."""
    return get_detector().detect_cpp_compiler()


def detect_ncu() -> ToolInfo:
    """Convenience: Detect NCU."""
    return get_detector().detect_ncu()


def detect_nsys() -> ToolInfo:
    """Convenience: Detect NSYS."""
    return get_detector().detect_nsys()


def detect_nvprof() -> ToolInfo:
    """Convenience: Detect NVPROF."""
    return get_detector().detect_nvprof()


def print_detection_status():
    """Print detection status for all tools."""
    get_detector().print_status()


# ============================================================================
# CLI Entry Point (for testing)
# ============================================================================

if __name__ == '__main__':
    # Test detection
    print_detection_status()
