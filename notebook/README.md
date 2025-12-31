# NOMA Jupyter Extension

NOMA binary have to be build 

Installation of rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
```

NOMA build

```bash
source $HOME/.cargo/env && cd /workspaces/NOMA && cargo build --release
```

IPython extension to execute NOMA code in Jupyter notebooks with the `%%noma` cell magic.

## Installation

In a notebook cell:

```python
import sys
sys.path.insert(0, '/workspaces/NOMA/notebook')
%load_ext noma_magic
```

## Usage

Write `%%noma` at the beginning of a Python cell to execute NOMA code:

```python
%%noma
fn sigmoid(x) { 1.0 / (1.0 + exp(-x)) }
print(sigmoid(0.0))
```

## Commands

- `%%noma` : Execute NOMA code
- `%%noma --no-cache` : Skip cache
- `%%noma_info` : Show environment info
- `%%noma_clear_cache` : Clear cache

## Files and cache

Files are stored in `~/.noma_jupyter/`:
- `workspace/` : temporary .noma files
- `cache/` : compiled results
- `artifacts/` : outputs from your programs
- `execution.log` : JSON history

Cache uses SHA256 hash of the code to avoid unnecessary recompilation.

## Access from Python

```python
executor = get_ipython().user_ns['_noma_executor']
print(f"Workspace: {executor.work_dir}")
```

## Examples

See `examples/`:
- `01_getting_started.ipynb` : Getting started
- `02_neural_networks.ipynb` : Neural networks  
- `03_advanced.ipynb` : Advanced patterns