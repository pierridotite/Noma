# NOMA VS Code Extension - Installation

Installation and usage guide for the NOMA Language extension for Visual Studio Code.

## Prerequisites

You need Visual Studio Code 1.74.0 or higher and the NOMA compiler installed on your system.

## Installation

### From VSIX Package

Build the extension package:

```bash
cd noma-vscode
npm install
npm run package
```

Install in VS Code by opening the command palette with Ctrl+Shift+P (Cmd+Shift+P on Mac), typing "Extensions: Install from VSIX", and selecting the generated `noma-lang-0.1.0.vsix` file.

### Development Mode

For testing or contributing:

```bash
cd noma-vscode
npm install
npm run compile
```

Open the noma-vscode folder in VS Code and press F5 to launch the Extension Development Host.

## Configuration

### NOMA Binary

The extension searches for the NOMA binary in your system PATH, or in `target/release/noma` and `target/debug/noma` within your workspace.

To add NOMA to your PATH on Linux or Mac:

```bash
export PATH="$PATH:/path/to/NOMA/target/release"
```

On Windows, add the path through System Properties > Environment Variables.

### Editor Settings

You can customize settings in your VS Code settings.json:

```json
{
  "[noma]": {
    "editor.tabSize": 2,
    "editor.formatOnSave": true
  }
}
```

## Usage

The extension provides syntax highlighting, over 30 code snippets, and commands for running and building NOMA programs.

Press Ctrl+Shift+R (Cmd+Shift+R on Mac) to run the current file, or Ctrl+Shift+B (Cmd+Shift+B on Mac) to build an executable.

Code folding is supported with region markers:

```noma
// #region Section Name
// your code here
// #endregion
```

## Troubleshooting

If syntax highlighting is not working, verify your file has the .noma extension. You can manually set the language by clicking the language indicator in the bottom-right corner of VS Code.

If the NOMA binary is not found, the extension falls back to using `cargo run`. For faster execution, ensure the NOMA binary is in your PATH.

To check if the extension is active, open a .noma file and look for "NOMA Language extension is now active!" in the Output panel under Extension Host.

Reload VS Code with Ctrl+Shift+P > "Developer: Reload Window" if needed.

## Development

For continuous compilation during development, run `npm run watch`.

To add features, edit the relevant files and test by pressing F5. See README.md for a complete list of snippets and features.

## License

MIT License

## Prerequisites

Before installing the extension, ensure you have:

1. **Visual Studio Code** (version 1.74.0 or higher)
2. **NOMA compiler** - One of the following:
   - NOMA binary in your PATH
   - NOMA source repository with Cargo build setup

## Installation Methods

### Method 1: From VSIX (Recommended)

1. **Build the extension package:**
   ```bash
   cd noma-vscode
   npm install
   npm run package
   ```

2. **Install in VS Code:**
   - Open VS Code
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Extensions: Install from VSIX"
   - Select the generated `noma-lang-0.1.0.vsix` file

### Method 2: Development Mode

For testing or contributing to the extension:

1. **Setup:**
   ```bash
   cd noma-vscode
   npm install
   npm run compile
   ```

2. **Launch Extension Development Host:**
   - Open the `noma-vscode` folder in VS Code
   - Press `F5` to launch a new VS Code window with the extension loaded
   - Open any `.noma` file to see the extension in action

### Method 3: From Marketplace (Future)

Once published, you'll be able to install directly from the VS Code Marketplace:
- Open Extensions view (`Ctrl+Shift+X`)
- Search for "NOMA Language"
- Click Install

## Configuration

### NOMA Binary Setup

The extension will automatically search for the NOMA binary in:
1. System PATH
2. Workspace `target/release/noma`
3. Workspace `target/debug/noma`

**To add NOMA to PATH:**

**Linux/Mac:**
```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$PATH:/path/to/NOMA/target/release"
```

**Windows:**
```powershell
# Add to system PATH via Environment Variables
setx PATH "%PATH%;C:\\path\\to\\NOMA\\target\\release"
```

### VS Code Settings

The extension provides default settings for NOMA files:

```json
{
  "editor.tabSize": 4,
  "editor.insertSpaces": true,
  "editor.detectIndentation": false
}
```

To customize, add to your `settings.json`:

```json
{
  "[noma]": {
    "editor.tabSize": 2,
    "editor.formatOnSave": true,
    "editor.wordWrap": "on"
  }
}
```

## Features

### Syntax Highlighting

The extension provides rich syntax highlighting for:
- Keywords (control flow, declarations, memory management)
- Built-in functions (math, activation, tensor operations)
- Operators (arithmetic, comparison, logical, casting)
- String literals and numeric constants
- Comments (line and block)

### Code Snippets

Type a prefix and press `Tab` to insert a snippet. Examples:

**Basic snippets:**
- `fn` → function template
- `let` → variable declaration
- `learn` → learnable parameter
- `optimize` → optimization loop

**Tensor operations:**
- `tensor1d` → 1D tensor
- `matmul` → matrix multiplication
- `dot` → dot product

**Complete templates:**
- `linreg` → full linear regression
- `nnlayer` → neural network layer

See [README.md](README.md) for the complete list of 30+ snippets.

### Commands

Three commands are available:

| Command | Keyboard Shortcut | Description |
|---------|------------------|-------------|
| `NOMA: Run Current File` | `Ctrl+Shift+R` (Mac: `Cmd+Shift+R`) | Compile and run the current file |
| `NOMA: Build Current File` | `Ctrl+Shift+B` (Mac: `Cmd+Shift+B`) | Build an executable |
| `NOMA: Show Version Info` | (Command Palette) | Display NOMA version |

**To use commands:**
1. Open a `.noma` file
2. Press the keyboard shortcut, or
3. Press `Ctrl+Shift+P` and type the command name

### Auto-completion

The extension provides:
- Auto-closing brackets, braces, and parentheses
- Auto-closing quotes
- Smart indentation
- Word-based suggestions from your code

### Code Folding

Fold code blocks using:
- Click the fold icon next to line numbers
- `Ctrl+Shift+[` to fold current block
- `Ctrl+Shift+]` to unfold current block

Region markers:
```noma
// #region Training Loop
optimize(W) until loss < 0.01 {
    // ...
}
// #endregion
```

## Usage Examples

### Running a NOMA Program

1. Create or open a `.noma` file
2. Write your code:
   ```noma
   fn main() {
       print("Hello, NOMA!");
       let x = 42.0;
       print(x);
       return x;
   }
   ```
3. Press `Ctrl+Shift+R` to run
4. View output in the integrated terminal

### Using Snippets

1. Type `optimize` and press `Tab`
2. Fill in the placeholders:
   ```noma
   optimize(w) until loss < 0.001 {
       let loss = w * w;
       minimize loss;
   }
   ```
3. Press `Tab` to move between placeholders

### Building an Executable

1. Open your `.noma` file
2. Press `Ctrl+Shift+B`
3. Find the compiled executable in your workspace directory

## Troubleshooting

### Extension Not Working

1. **Check NOMA installation:**
   ```bash
   noma --version
   ```
   If this fails, ensure NOMA is built and in PATH.

2. **Reload VS Code:**
   - Press `Ctrl+Shift+P`
   - Type "Developer: Reload Window"

3. **Check extension activation:**
   - Open a `.noma` file
   - Look for "NOMA Language extension is now active!" in Output → Extension Host

### Syntax Highlighting Not Showing

1. **Verify file extension:**
   - File must have `.noma` extension
   - Check bottom-right corner of VS Code shows "NOMA"

2. **Manually set language:**
   - Click language indicator (bottom-right)
   - Select "NOMA" from the list

### Commands Not Available

1. **Check file type:**
   - Commands only work in `.noma` files
   - Open a NOMA file and try again

2. **Reinstall extension:**
   - Remove current installation
   - Rebuild and reinstall VSIX

### Terminal Output Issues

1. **Terminal not showing:**
   - Press `` Ctrl+` `` to toggle terminal
   - Look for "NOMA" terminal tab

2. **Binary not found:**
   - Extension falls back to `cargo run`
   - Add NOMA binary to PATH for faster execution

## Development & Contributing

### Building from Source

```bash
cd noma-vscode
npm install
npm run compile
```

### Testing Changes

```bash
npm run watch    # Auto-compile on changes
```

Then press `F5` in VS Code to launch Extension Development Host.

### Extension Structure

```
noma-vscode/
├── src/
│   └── extension.ts          # Main extension code
├── syntaxes/
│   └── noma.tmLanguage.json  # Syntax highlighting rules
├── snippets/
│   └── noma.json             # Code snippets
├── language-configuration.json  # Language features
├── package.json              # Extension manifest
└── tsconfig.json             # TypeScript config
```

### Adding New Features

**To add a keyword:**
1. Edit `syntaxes/noma.tmLanguage.json`
2. Add to appropriate pattern group
3. Reload extension

**To add a snippet:**
1. Edit `snippets/noma.json`
2. Follow existing pattern
3. Test in Extension Development Host

**To add a command:**
1. Edit `src/extension.ts`
2. Register command in `activate()`
3. Add to `package.json` contributes.commands
4. Compile and test

## Resources

- [NOMA Language Guide](../LANGUAGE_GUIDE.md) - Complete language reference
- [Example Programs](../examples/) - 30+ example NOMA programs
- [GitHub Repository](https://github.com/pierridotite/NOMA) - Source code and issues
- [VS Code Extension API](https://code.visualstudio.com/api) - For contributors

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or contributions:
- Open an issue on [GitHub](https://github.com/pierridotite/NOMA/issues)
- Check existing documentation
- Review example programs

---

**Happy coding with NOMA! 🚀**
