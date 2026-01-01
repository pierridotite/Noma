"""
NOMA Cell Magic Implementation
==============================

Provides %%noma magic for executing NOMA code in Jupyter notebooks.
Handles execution context, artifact management, and compilation caching.
"""

import os
import sys
import subprocess
import tempfile
import hashlib
import json
from pathlib import Path
from datetime import datetime
from IPython.core.magic import register_cell_magic
from IPython.core.error import UsageError
from IPython.display import display, HTML, Javascript
from typing import Optional, Dict, Tuple


class NomaExecutor:
    """Manages execution of NOMA code in Jupyter notebooks."""
    
    def __init__(self):
        """Initialize the NOMA executor."""
        # Create stable working directory in notebook directory
        self.notebook_dir = Path.home() / ".noma_jupyter"
        self.notebook_dir.mkdir(exist_ok=True)
        
        # Working directory for temporary files
        self.work_dir = self.notebook_dir / "workspace"
        self.work_dir.mkdir(exist_ok=True)
        
        # Cache directory for compiled outputs
        self.cache_dir = self.notebook_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Artifacts directory (outputs, logs)
        self.artifacts_dir = self.notebook_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Execution log
        self.log_file = self.notebook_dir / "execution.log"
    
    def find_noma_binary(self) -> str:
        """Find the NOMA binary in PATH or build directory."""
        # Try to find in common locations
        candidates = [
            "noma",  # in PATH
            "/workspaces/NOMA/target/release/noma",
            "/workspaces/NOMA/target/debug/noma",
            str(Path.cwd() / "target" / "release" / "noma"),
            str(Path.cwd() / "target" / "debug" / "noma"),
        ]
        
        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, "--version"],
                    capture_output=True,
                    timeout=1
                )
                if result.returncode == 0:
                    return candidate
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        # If not found, raise error with helpful message
        raise FileNotFoundError(
            "NOMA binary not found. Please ensure NOMA is built and in PATH.\n"
            f"Checked: {', '.join(candidates)}"
        )
    
    def hash_content(self, content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get_cache_path(self, content_hash: str, mode: str = "interpreter") -> Path:
        """Get cache path for a compiled output."""
        return self.cache_dir / f"{content_hash}_{mode}"
    
    def check_cache(self, content: str, mode: str = "interpreter") -> Optional[Path]:
        """Check if compiled output exists in cache."""
        content_hash = self.hash_content(content)
        cache_path = self.get_cache_path(content_hash, mode)
        
        if cache_path.exists():
            return cache_path
        return None
    
    def save_execution_log(self, cell_num: int, code: str, output: str, 
                          error: Optional[str], mode: str, success: bool):
        """Log execution details for debugging."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "cell_number": cell_num,
            "mode": mode,
            "success": success,
            "code_hash": self.hash_content(code),
            "output_preview": output[:200] if output else "",
            "error_preview": error[:200] if error else "",
        }
        
        try:
            logs = []
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            
            logs.append(log_entry)
            
            with open(self.log_file, 'w') as f:
                json.dump(logs[-100:], f, indent=2)  # Keep last 100 entries
        except Exception:
            pass  # Silently fail on logging errors
    
    def execute(self, code: str, cell_num: int = 0, mode: str = "interpreter",
                options: Optional[Dict] = None) -> Tuple[str, Optional[str], int]:
        """
        Execute NOMA code.
        
        Args:
            code: NOMA source code to execute
            cell_num: Cell number in notebook (for debugging)
            mode: "interpreter" or "build-exe"
            options: Optional dict with keys:
                - noma_path: Path to NOMA binary
                - work_dir: Custom working directory
                - use_cache: Enable caching (default: True)
                - extra_args: List of extra CLI arguments
        
        Returns:
            (stdout, stderr, return_code)
        """
        options = options or {}
        noma_binary = options.get("noma_path") or self.find_noma_binary()
        work_dir = Path(options.get("work_dir", self.work_dir))
        use_cache = options.get("use_cache", True)
        extra_args = options.get("extra_args", [])
        
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Check cache if enabled
        if use_cache and mode == "interpreter":
            cached = self.check_cache(code, mode)
            if cached:
                try:
                    with open(cached, 'r') as f:
                        return f.read(), None, 0
                except Exception:
                    pass  # Fall through to execution
        
        # Save code to temporary file
        code_hash = self.hash_content(code)
        code_file = work_dir / f"cell_{cell_num}_{code_hash}.noma"
        
        try:
            with open(code_file, 'w') as f:
                f.write(code)
        except Exception as e:
            return "", f"Error writing NOMA code file: {e}", 1
        
        # Build command
        if mode == "build-exe":
            output_file = work_dir / f"noma_cell_{cell_num}"
            cmd = [noma_binary, "build-exe", str(code_file), "-o", str(output_file)] + extra_args
            execute_cmd = [str(output_file)]
        else:  # interpreter mode
            cmd = [noma_binary, "run", str(code_file)] + extra_args
            execute_cmd = cmd
        
        # Execute
        try:
            result = subprocess.run(
                execute_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(work_dir)
            )
            
            output = result.stdout
            error = result.stderr if result.returncode != 0 else None
            
            # Map error lines from file to notebook cell
            if error and "error" in error.lower():
                error = self._map_error_lines(error, cell_num)
            
            # Cache successful interpreter output
            if use_cache and result.returncode == 0 and mode == "interpreter":
                try:
                    cache_path = self.get_cache_path(code_hash, mode)
                    with open(cache_path, 'w') as f:
                        f.write(output)
                except Exception:
                    pass  # Silently skip cache write errors
            
            # Log execution
            self.save_execution_log(cell_num, code, output, error, mode, 
                                   result.returncode == 0)
            
            return output, error, result.returncode
        
        except subprocess.TimeoutExpired:
            error = f"NOMA execution timed out (>30s) for cell {cell_num}"
            self.save_execution_log(cell_num, code, "", error, mode, False)
            return "", error, 124
        
        except Exception as e:
            error = f"Error executing NOMA: {e}"
            self.save_execution_log(cell_num, code, "", error, mode, False)
            return "", error, 1
    
    def _map_error_lines(self, error: str, cell_num: int) -> str:
        """Map file line numbers to cell lines for better error reporting."""
        lines = error.split('\n')
        mapped_lines = []
        
        for line in lines:
            # Try to find patterns like "line 5:" or "L5"
            if 'line' in line.lower():
                # Add cell reference
                mapped_lines.append(f"[Cell {cell_num}] {line}")
            else:
                mapped_lines.append(line)
        
        return '\n'.join(mapped_lines)
    
    def list_artifacts(self) -> Dict[str, list]:
        """List available artifacts from executions."""
        artifacts = {
            "workspace": list(self.work_dir.glob("*.noma")) if self.work_dir.exists() else [],
            "cache": list(self.cache_dir.glob("*")) if self.cache_dir.exists() else [],
            "outputs": list(self.artifacts_dir.glob("*")) if self.artifacts_dir.exists() else [],
        }
        return artifacts
    
    def clear_cache(self):
        """Clear compilation cache."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir()
        return "Cache cleared"


# Global executor instance
_executor = NomaExecutor()


def noma_magic(line: str, cell: str) -> Tuple[str, Optional[str], int]:
    """
    %%noma cell magic implementation.
    
    Usage:
        %%noma
        // NOMA code
        
        %%noma -m build-exe
        // Build executable
        
        %%noma --no-cache
        // Disable caching
    """
    # Allow empty cells or cells with only comments
    cell_stripped = cell.strip()
    if not cell_stripped or cell_stripped.startswith("//"):
        # Empty cell or comment-only cell - just return success
        return "", None, 0
    
    # Parse magic options
    options = {}
    mode = "interpreter"
    use_cache = True
    extra_args = []
    
    if line.strip():
        parts = line.strip().split()
        for part in parts:
            if part in ("-m", "--mode"):
                continue
            elif part in ("build-exe", "interpreter"):
                mode = part
            elif part in ("--no-cache",):
                use_cache = False
            elif part.startswith("-"):
                extra_args.append(part)
    
    options["use_cache"] = use_cache
    options["extra_args"] = extra_args
    
    # Get cell number from IPython
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        cell_num = len(ipython.user_ns.get('_noma_cells', []))
        ipython.user_ns.setdefault('_noma_cells', []).append({
            'num': cell_num,
            'code': cell,
        })
    except Exception:
        cell_num = 0
    
    # Execute
    output, error, code = _executor.execute(cell, cell_num=cell_num, mode=mode, 
                                           options=options)
    
    # Display results
    if output:
        display(HTML(f"<pre style='background:#f5f5f5; padding:10px; border-radius:5px; color:#000;'>{output}</pre>"))
    
    if error:
        display(HTML(f"<pre style='background:#ffe6e6; padding:10px; border-radius:5px; color:#c00;'><b>Error:</b>\n{error}</pre>"))
    
    if code != 0:
        raise RuntimeError(f"NOMA execution failed with code {code}")
    
    return output


def load_ipython_extension(ipython):
    """Load the NOMA extension in IPython."""
    ipython.register_magics(NomaMagics)
    ipython.user_ns['_noma_executor'] = _executor
    
    print("NOMA magic loaded. Use %%noma in a cell to execute NOMA code.")
    print(f"Working directory: {_executor.work_dir}")
    print(f"Artifacts directory: {_executor.artifacts_dir}")


from IPython.core.magic import Magics, magics_class, cell_magic


@magics_class
class NomaMagics(Magics):
    """NOMA magic commands."""
    
    @cell_magic
    def noma(self, line: str, cell: str):
        """Execute NOMA code in a cell."""
        return noma_magic(line, cell)
    
    @cell_magic
    def noma_info(self, line: str, cell: str):
        """Display information about NOMA workspace."""
        artifacts = _executor.list_artifacts()
        
        info = f"""
        <div style='background:#f9f9f9; padding:15px; border-radius:5px; font-family:monospace; font-size:12px;'>
        <h4>NOMA Jupyter Environment</h4>
        <p><b>Binary:</b> {_executor.find_noma_binary()}</p>
        <p><b>Working directory:</b> {_executor.work_dir}</p>
        <p><b>Cache directory:</b> {_executor.cache_dir}</p>
        <p><b>Artifacts:</b> {len(artifacts['outputs'])} files</p>
        <p><b>Cached compilations:</b> {len(artifacts['cache'])}</p>
        </div>
        """
        display(HTML(info))
    
    @cell_magic
    def noma_clear_cache(self, line: str, cell: str):
        """Clear the NOMA compilation cache."""
        msg = _executor.clear_cache()
        display(HTML(f"<p style='color:#060;'>{msg}</p>"))
