// extension.ts
// @ts-ignore
import * as vscode from 'vscode';

let sessionStart = Date.now();
let keystrokes: { key: string; time: number }[] = [];

function buildSessionState(keystrokes: {key: string; time: number}[], sessionStart: number) {
    
}

export function activate(context: vscode.ExtensionContext) {
  // Track active time
  const disposable = vscode.window.onDidChangeActiveTextEditor(() => {
    sessionStart = Date.now(); // or refine with focus events
  });

  // Capture keystrokes (only in code files)
  vscode.workspace.onDidChangeTextDocument(e => {
    const now = Date.now();
    e.contentChanges.forEach(change => {
      if (change.text === '') {
        // Deletion
        keystrokes.push({ key: 'Backspace', time: now });
      } else {
        // Insertion
        for (const char of change.text) {
          keystrokes.push({ key: char, time: now });
        }
      }
    });
  });

  // Send to backend every 60s
  setInterval(() => {
    const payload = buildSessionState(keystrokes, sessionStart);
    fetch('http://localhost:8000/ingest', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
    keystrokes = []; // reset
  }, 60000);
}