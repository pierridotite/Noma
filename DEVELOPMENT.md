# NOMA Compiler

**Status:** Milestone 1 Complete - The Skeleton ✓

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

### Test the Lexer

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

- ✓ Complete lexer with keyword recognition (`learn`, `optimize`, `tensor`, etc.)
- ✓ Operator tokenization (+, -, *, /, ==, !=, <, >, etc.)
- ✓ Number literals (integers and floats)
- ✓ Identifiers and variable names
- ✓ Comment support (`//`)
- ✓ CLI with `build` and `check` commands
- ✓ Unit tests for tokenization

## Next Steps (Milestone 2)

- [ ] Parser implementation (recursive descent)
- [ ] AST node definitions
- [ ] Type system foundation
- [ ] Computational graph representation

## Testing

Try tokenizing one of the examples:

```bash
cargo run -- build examples/simple.noma --tokens
```

You should see the token stream output showing how NOMA breaks down your code.
