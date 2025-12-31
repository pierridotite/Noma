"""
NOMA Jupyter Extension
======================

A lightweight IPython extension providing a %%noma cell magic to execute NOMA code
directly from Jupyter notebooks with robust context management, artifact handling,
and compilation caching.

Usage:
    Load the magic with:
        %load_ext noma_magic
    
    Then use %%noma in a cell:
        %%noma
        // NOMA code here
        fn sigmoid(x) { 1.0 / (1.0 + exp(-x)) }
        print(sigmoid(0.0))
"""

__version__ = "0.1.0"

from .magic import load_ipython_extension

__all__ = ["load_ipython_extension"]
