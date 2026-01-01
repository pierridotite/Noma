# NOMA Language Extension for VS Code

Syntax highlighting, snippets, and language support for the NOMA programming language - an ML-first language with automatic differentiation.

## Features

- **Syntax Highlighting** - Full TextMate grammar for `.noma` files
  - Keywords (control flow, declarations, memory management)
  - Built-in functions (math, activation, tensor operations)
  - Operators (arithmetic, comparison, logical, casting)
  - String literals and numeric constants
  - Comments (line and block)
- **Rich Snippets** - 30+ quick templates for common patterns
- **Auto-closing** - Brackets, braces, parentheses, and strings
- **Indentation** - Smart indent/outdent for blocks
- **Commands** - Run and build NOMA files directly from VS Code
  - `Ctrl+Shift+R` (Mac: `Cmd+Shift+R`) - Run current file
  - `Ctrl+Shift+B` (Mac: `Cmd+Shift+B`) - Build current file

## Installation

### From Source (Development)

1. Clone the repository:
   ```bash
   cd noma-vscode
   npm install
   ```

2. Open VS Code and press `F5` to launch the Extension Development Host

### Package as VSIX

```bash
### Core Language
| Prefix | Description |
|--------|-------------|
| `fn main` | Main function template |
| `fn` | Function template |
| `let` | Immutable variable |
| `learn` | Learnable parameter |
| `optimize` | Optimization loop |
| `if` | If statement |
| `ifelse` | If-else statement |
| `while` | While loop |

### Tensors & Operations
| Prefix | Description |
|--------|-------------|
| `tensor1d` | 1D tensor |
| `tensor2d` | 2D tensor (matrix) |
| `matmul` | Matrix multiplication |
| `dot` | Dot product |
| `sum` | Sum reduction |
| `mean` | Mean reduction |

### Activation Functions
| Prefix | Description |
|--------|-------------|
| `sigmoid` | Sigmoid activation |
| `relu` | ReLU activation |
| `tanh` | Tanh activation |

### Math Functions
| Prefix | Description comprehensive highlighting for:

### Keywords
- **Control flow**: `if`, `else`, `while`, `for`, `return`, `until`, `optimize`, `minimize`, `batch`, `in`
- **Declarations**: `fn`, `let`, `learn`, `struct`
- **Memory management**: `alloc`, `free`, `realloc`
- **Optimizer control**: `reset_optimizer`
- **File I/O**: `load_csv`, `save_csv`, `load_safetensors`, `save_safetensors`
- **Type casting**: `as`

### Types
- `tensor` - Multi-dimensional arrays
- `f64`, `f32`, `i32`, `i64` - Type annotations (all cast to f64 internally)

### // Print status messages
    print("Training neural network...");
    
    // Load data from CSV
    load_csv X = "data.csv";
    
    // Initialize weights with He initialization
    learn W = he_init(64.0, 64.0, 10.0);
    learn b = rand_tensor(10.0);
    
    // Configure Adam optimizer
    let optimizer = 2.0;
    let learning_rate = 0.001;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 0.00000001;
    let max_iterations = 10000;
    
    // Training loop
    optimize(W) until loss < 0.01 {
        let z = matmul(X, W) + b;
        let pred = relu(z);
        let loss = mean(pred * pred);
        minimize loss;
    }
    
    // Type casting
    let final_loss = loss as f64;
    
    // Save trained model
    save_safetensors {
        weights: W,
        bias: b
    }, "model.safetensors";
    
    print("Training complete!");
    print(final_loss);
    
    return W;
}
```

## Language Features

NOMA is designed for machine learning with:
- **Automatic differentiation** - Gradients computed automatically
- **Built-in optimizers** - SGD, Adam, RMSprop
- **Tensor operations** - Efficient linear algebra
- **Dynamic memory** - Heap allocation for flexible shapes
- **File I/O** - CSV and Safetensors support
- **Batch processing** - Efficient data iteration

For complete language documentation, see [LANGUAGE_GUIDE.md](../LANGUAGE_GUIDE.md).dot` - Dot product
- `matmul` - Matrix multiplication

**Random number generation:**
- `rand`, `rand_uniform`, `rand_normal`
- `rand_tensor`, `rand_normal_tensor`
- `xavier_init`, `he_init` - Weight initialization

**I/O:**
- `print` - Output values and strings

### Operators
- **Arithmetic**: `+`, `-`, `*`, `/`, `%`, `^`, `**` (power)
- **Comparison**: `==`, `!=`, `<`, `>`, `<=`, `>=`
- **Logical**: `&&`, `||`, `!`
- **Assignment**: `=`
- **Type casting**: `as`

### Literals
- **Numbers**: Integers and floats (with scientific notation)
- **Strings**: Double-quoted with escape sequences (`\n`, `\t`, `\"`, `\\`)
- **Comments**: Line (`//`) and block (`/* */`)
| `hyper` | Basic hyperparameters |
| `adam` | Adam optimizer configuration |
| `rmsprop` | RMSprop optimizer configuration |
| `resetopt` | Reset optimizer state |

### Memory Management
| Prefix | Description |
|--------|-------------|
| `alloc` | Allocate heap tensor |
| `free` | Free heap tensor |
| `realloc` | Reallocate tensor with new shape |

### File I/O
| Prefix | Description |
|--------|-------------|
| `loadcsv` | Load data from CSV |
| `savecsv` | Save data to CSV |
| `loadsafe` | Load from Safetensors |
| `savesafe` | Save to Safetensors |

### Batch Processing
| Prefix | Description |
|--------|-------------|
| `batch` | Batch loop |
| `batchi` | Batch loop with index |

### Initialization
| Prefix | Description |
|--------|-------------|
| `randinit` | He initialization (for ReLU) |
| `xavierinit` | Xavier initialization (for sigmoid/tanh) |

### Complete Templates
| Prefix | Description |
|--------|-------------|
| `linreg` | Full linear regression template |
| `nnlayer` | Neural network layer with initialization |
| `mse` | Mean squared error loss |

### I/O & Debugging
| Prefix | Description |
|--------|-------------|
| `print` | Print numeric value |
| `prints` | Print string message |
| `as` | Type cased error |
| `hyper` | Hyperparameters |
| `linreg` | Full linear regression template |
| `sigmoid` | Sigmoid activation |
| `relu` | ReLU activation |
| `print` | Print value |
| `sum` | Sum reduction |
| `mean` | Mean reduction |
| `dot` | Dot product |

## Syntax Highlighting

The extension provides highlighting for:

- **Keywords** - `fn`, `let`, `learn`, `return`, `optimize`, `until`, `minimize`, `if`, `else`, `while`, `for`
- **Types** - `tensor`
- **Built-ins** - `sigmoid`, `relu`, `sum`, `mean`, `dot`, `matmul`, `print`, `exp`, `log`, `sqrt`, `abs`, `sin`, `cos`, `tan`
- **Operators** - Arithmetic, comparison, logical, assignment
- **Numbers** - Integers and floats (including scientific notation)
- **Comments** - Line (`//`) and block (`/* */`)
- **Strings** - Double-quoted with escape sequences

## Example

```noma
fn main() {
    learn x = 5.0;
    
    let learning_rate = 0.01;
    let max_iterations = 1000;
    
    optimize(x) until loss < 0.0001 {
        let loss = x * x;
        minimize loss;
    }
    
    print(x);
    return x;
}
```

## License

MIT
