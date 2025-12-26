# NOMA Compiler

**Status:** Milestone 4 In Progress** (CPU autodiff complete; LLVM/PTX backends present but minimal)

## Quick Start

### Build the Compiler

```bash
cargo build --release
```

### Run Examples

Tokenize a NOMA file:

```bash
cargo run -- build examples/simple.noma --tokens
```

Check syntax:

```bash
cargo run -- check examples/xor.noma
```

### Run the Test Suite

```bash
cargo test
```

## Project Structure

```
noma/
├── src/
│   ├── main.rs         # CLI entry point
│   ├── lib.rs          # Library exports
│   ├── lexer.rs        # Lexical analyzer
│   ├── token.rs        # Token definitions
│   └── error.rs        # Error types
├── examples/           # Sample .noma files
│   ├── simple.noma
│   ├── xor.noma
│   ├── linear_regression.noma
│   └── test_tokens.noma
├── Cargo.toml
└── README.md
```

## What Works Now

- Lexer: keywords, comments, numbers, identifiers, operators including `%`, `^/**`, `&&`, `||`.
- Parser: functions, structs, blocks, assignments, `minimize`, `optimize … until` with proper precedence for power/mod.
- Computational graph: topological forward/backward, gradients for arithmetic and sigmoid/relu calls; safe handling of non-differentiable ops (mod/and/or).
- Backends: LLVM IR emission for arithmetic, power, comparisons, logical ops; PTX emission for the same with pow approximation.
- Optimize loops: executed with SGD until the condition becomes true.
- Tests: coverage across lexer, parser, graph, LLVM/PTX codegen.

## Next Steps

- Tensor types and GPU kernels.
- Richer type checking and semantics.
- More differentiation-aware intrinsics and math library.

## Testing

Try tokenizing one of the examples:

```bash
cargo run -- build examples/simple.noma --tokens
```

You should see the token stream output showing how NOMA breaks down your code.
