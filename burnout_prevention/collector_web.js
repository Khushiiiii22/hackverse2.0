let keystrokes = [];
let sessionStart = performance.now();

document.addEventListener('keydown', (e) => {
  keystrokes.push({ key: e.key, time: Date.now() });
});

// Monitor AI assistant queries
const chatInput = document.querySelector('#ai-chat-input');
chatInput?.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') {
    const query = chatInput.value;
    if (/explain|how to fix|why/i.test(query)) {
      sendHelpQuery(query);
    }
  }
});

function buildAndSendState() {
  const payload = {
    typing_speed_wpm: computeWPM(keystrokes),
    backspace_rate: keystrokes.filter(k => k.key === 'Backspace').length / keystrokes.length,
    spelling_errors: countMisspelledWords(editor.getValue()),
    session_duration_minutes: (Date.now() - sessionStart) / 60000,
    current_time: new Date().toISOString(),
    recent_queries: getRecentHelpQueries()
  };
  fetch('/api/ingest', { method: 'POST', body: JSON.stringify(payload) });
}