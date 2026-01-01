# Changelog

All notable changes to the NOMA VS Code extension will be documented in this file.

---

## [Unreleased]

### Added
- Enhanced extension logic with automatic NOMA binary detection
- Fallback to `cargo run` when binary not found in PATH
- `NOMA: Show Version Info` command to display compiler version
- Comprehensive documentation with INSTALL.md guide

### Changed
- Improved terminal integration for command execution
- Updated package description to emphasize ML-first and autodiff features
- Enhanced README with categorized snippets and complete feature overview

---

## [0.1.0] - 2025-12-27

### Added

#### Syntax Highlighting
- Complete TextMate grammar for `.noma` files
- **Keywords**: 
  - Control flow: `if`, `else`, `while`, `for`, `return`, `until`, `optimize`, `minimize`, `batch`, `in`
  - Declarations: `fn`, `let`, `learn`, `struct`
  - Memory management: `alloc`, `free`, `realloc`
  - Optimizer control: `reset_optimizer`
  - File I/O: `load_csv`, `save_csv`, `load_safetensors`, `save_safetensors`
  - Type casting: `as`
- **Built-in functions**:
  - Activation: `sigmoid`, `relu`, `tanh`
  - Math: `exp`, `log`, `sqrt`, `abs`, `sin`, `cos`, `tan`, `pow`, `min`, `max`, `floor`, `ceil`
  - Tensor ops: `sum`, `mean`, `dot`, `matmul`
  - Random: `rand`, `rand_uniform`, `rand_normal`, `rand_tensor`, `rand_normal_tensor`
  - Initialization: `xavier_init`, `he_init`
  - I/O: `print`
- **Operators**: Arithmetic (`+`, `-`, `*`, `/`, `%`, `^`, `**`), comparison, logical, assignment, casting
- **Types**: `tensor`, `f64`, `f32`, `i32`, `i64`
- **Literals**: Numbers (int/float), strings with escape sequences
- **Comments**: Line (`//`) and block (`/* */`)

#### Code Snippets (30+)
- **Core language**: `fn main`, `fn`, `let`, `learn`, `optimize`, `if`, `ifelse`, `while`
- **Tensors**: `tensor1d`, `tensor2d`, `matmul`, `dot`, `sum`, `mean`
- **Activations**: `sigmoid`, `relu`, `tanh`
- **Math**: `pow`, `min`, `max`
- **Optimizers**: `hyper`, `adam`, `rmsprop`, `resetopt`
- **Memory**: `alloc`, `free`, `realloc`
- **File I/O**: `loadcsv`, `savecsv`, `loadsafe`, `savesafe`
- **Batch processing**: `batch`, `batchi`
- **Initialization**: `randinit`, `xavierinit`
- **Templates**: `linreg`, `nnlayer`, `mse`
- **I/O**: `print`, `prints`, `as`

#### Commands & Keybindings
- `NOMA: Run Current File` - Execute NOMA program (Ctrl+Shift+R / Cmd+Shift+R)
- `NOMA: Build Current File` - Build executable (Ctrl+Shift+B / Cmd+Shift+B)
- `NOMA: Show Version Info` - Display NOMA version
- Play button in editor title bar for `.noma` files

#### Editor Features
- Auto-closing pairs for brackets, braces, parentheses, quotes
- Smart indentation for blocks
- Code folding with region markers
- Word pattern matching for identifiers
- Tab size defaults (4 spaces)

#### Documentation
- Comprehensive README with feature overview
- Installation guide (INSTALL.md)
- Snippet reference with categories
- Command usage examples
- Troubleshooting section

#### Extension Logic
- Automatic NOMA binary detection (PATH or workspace)
- Fallback to `cargo run` if binary not found
- Terminal integration for command execution
- File save before execution
- Language ID validation

### Technical Details
- TypeScript implementation with VS Code API
- Supports VS Code 1.74.0+
- Node.js 18+ for development
- VSIX packaging support

---

[0.1.0]: https://github.com/pierridotite/NOMA/releases/tag/v0.1.0
