import * as vscode from 'vscode';
import * as child_process from 'child_process';
import * as path from 'path';

/**
 * Find the NOMA binary in common locations
 */
function findNomaBinary(): string | null {
    const candidates = [
        'noma',  // in PATH
        path.join(vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '', 'target', 'release', 'noma'),
        path.join(vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '', 'target', 'debug', 'noma'),
    ];

    for (const candidate of candidates) {
        try {
            child_process.execSync(`"${candidate}" --version`, { stdio: 'ignore' });
            return candidate;
        } catch {
            continue;
        }
    }

    return null;
}

/**
 * Execute a NOMA command in the terminal
 */
function executeNomaCommand(command: string, filePath: string) {
    const nomaBinary = findNomaBinary();
    
    let terminal = vscode.window.terminals.find(t => t.name === 'NOMA');
    if (!terminal) {
        terminal = vscode.window.createTerminal({
            name: 'NOMA',
            hideFromUser: false,
        });
    }
    
    terminal.show();

    if (nomaBinary) {
        // Use noma binary directly
        terminal.sendText(`"${nomaBinary}" ${command} "${filePath}"`);
    } else {
        // Fall back to cargo run
        vscode.window.showWarningMessage('NOMA binary not found. Using cargo run...');
        terminal.sendText(`cargo run -- ${command} "${filePath}"`);
    }
}

export function activate(context: vscode.ExtensionContext) {
    console.log('NOMA Language extension is now active!');

    // Register a command to run the current NOMA file
    const runCommand = vscode.commands.registerCommand('noma.run', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const document = editor.document;
        if (document.languageId !== 'noma') {
            vscode.window.showErrorMessage('Not a NOMA file');
            return;
        }

        // Save the document first
        await document.save();

        const filePath = document.fileName;
        executeNomaCommand('run', filePath);
    });

    // Register a command to build the current NOMA file
    const buildCommand = vscode.commands.registerCommand('noma.build', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const document = editor.document;
        if (document.languageId !== 'noma') {
            vscode.window.showErrorMessage('Not a NOMA file');
            return;
        }

        await document.save();

        const filePath = document.fileName;
        executeNomaCommand('build-exe', filePath);
    });

    // Register a command to show NOMA info
    const infoCommand = vscode.commands.registerCommand('noma.info', () => {
        const nomaBinary = findNomaBinary();
        
        if (nomaBinary) {
            try {
                const version = child_process.execSync(`"${nomaBinary}" --version`).toString().trim();
                vscode.window.showInformationMessage(`NOMA: ${version}`);
            } catch {
                vscode.window.showErrorMessage('Failed to get NOMA version');
            }
        } else {
            vscode.window.showWarningMessage('NOMA binary not found in PATH or workspace');
        }
    });

    context.subscriptions.push(runCommand, buildCommand, infoCommand);
}

export function deactivate() {}
