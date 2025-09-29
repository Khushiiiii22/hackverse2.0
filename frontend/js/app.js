let burnoutEnabled = false;
let intervalId = null;
let pollId = null;
let evtSource = null;

const toggleButton = document.getElementById('burnoutToggle');
const statusDiv = document.getElementById('burnoutStatus');
const checkBurnoutBtn = document.getElementById('checkBurnoutBtn');
const issuesDiv = document.getElementById('burnoutIssues');
const debugPre = document.getElementById('burnoutDebug');
const forceTestBtn = document.getElementById('forceTestBtn');

// Guard missing elements (in case HTML wasn't served correctly)
if (!toggleButton || !statusDiv || !checkBurnoutBtn) {
  console.error('Burnout UI elements not found.');
}
console.log('burnout app.js loaded, elements:', {toggleButton, statusDiv, checkBurnoutBtn, issuesDiv, debugPre});

toggleButton.onclick = () => {
  burnoutEnabled = !burnoutEnabled;
  console.log('toggle clicked, enabled=', burnoutEnabled);

  if (burnoutEnabled) {
    toggleButton.textContent = "Disable Burnout Prevention";
    statusDiv.textContent = "Burnout prevention enabled. Monitoring will start now. Results available after 60 seconds.";
    alert("Burnout prevention enabled! Start your coding. Results will be updated automatically every 60 seconds.");

    // Enable manual check button
    checkBurnoutBtn.disabled = false;

  // Start automatic sending every 60 seconds
  intervalId = setInterval(sendBurnoutData, 60000);
  sendBurnoutData(); // Also send initially immediately

  // Start polling server for latest result every 5 seconds so terminal monitors
  // that POST to the backend will update the UI quickly.
  pollId = setInterval(fetchLatestResult, 5000);
  fetchLatestResult();

  } else {
    toggleButton.textContent = "Enable Burnout Prevention";
    statusDiv.textContent = "Burnout prevention disabled.";
    checkBurnoutBtn.disabled = true;
  clearInterval(intervalId);
  if (pollId) { clearInterval(pollId); pollId = null; }
  }
};

// Manual check button triggers sending data and displaying result
checkBurnoutBtn.onclick = () => {
  if (!burnoutEnabled) {
    alert("Please enable burnout prevention to check the status.");
    return;
  }
  sendBurnoutData();
};

if (forceTestBtn) {
  forceTestBtn.onclick = async () => {
    try {
  const r = await fetch('/api/burnout/test', { cache: 'no-store' });
      const d = await r.json();
      console.log('force test got:', d);
      statusDiv.textContent = d.intervention_message || 'No issues detected.';
      if (issuesDiv) issuesDiv.textContent = (d.detected_issues || []).join('\n');
      if (debugPre) { debugPre.style.display='block'; debugPre.textContent = JSON.stringify(d,null,2); }
    } catch (e) { console.error('force test failed', e); }
  };
}


async function sendBurnoutData() {
  const payload = {
    typing_speed_wpm: getTypingSpeed(),
    spelling_errors: getSpellingErrors(),
    backspace_rate: getBackspaceRate(),
    session_duration_minutes: getSessionDuration(),
    current_time: new Date().toISOString(),
    recent_queries: getRecentHelpQueries()
  };

  try {
  const response = await fetch('/api/burnout', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
  }, { cache: 'no-store' });
    const data = await response.json();
  console.log('sendBurnoutData received:', data);
      statusDiv.textContent = data.intervention_message || "No issues detected.";
      if (issuesDiv) {
        const issues = data.detected_issues || [];
        issuesDiv.textContent = issues.length ? issues.join('\n') : '';
      }
      if (debugPre) {
        debugPre.style.display = 'block';
        debugPre.textContent = JSON.stringify(data, null, 2);
      }
  } catch (error) {
    console.error("Error sending burnout data:", error);
    statusDiv.textContent = "Error retrieving burnout status.";
  }
}

// Poll the server for the most recent burnout analysis
async function fetchLatestResult() {
  try {
  const resp = await fetch('/api/burnout/latest', { cache: 'no-store' });
    const data = await resp.json();
  console.log('fetchLatestResult got:', data);
    if (data) {
      statusDiv.textContent = data.intervention_message || "No issues detected.";
      if (issuesDiv) {
        const issues = data.detected_issues || [];
        issuesDiv.textContent = issues.length ? issues.join('\n') : '';
      }
      if (debugPre) {
        debugPre.style.display = 'block';
        debugPre.textContent = JSON.stringify(data, null, 2);
      }
    }
  } catch (e) {
    console.error('Error fetching latest burnout result:', e);
  }
}

// Dummy data collectors for demonstration, replace with real ones
function getTypingSpeed() { return 45.0; }
function getSpellingErrors() { return 3; }
function getBackspaceRate() { return 0.15; }
function getSessionDuration() { return 15; }
function getRecentHelpQueries() { return ["how to fix error", "why is this happening"]; }

// Fetch once on load so the page immediately reflects any recent result
window.addEventListener('load', () => {
  console.log('window.load: fetching latest burnout result once');
  fetchLatestResult().catch(e => console.error('initial fetchLatestResult failed', e));
  // Open SSE connection for real-time updates (preferred)
  if (window.EventSource) {
    try {
      evtSource = new EventSource('/api/burnout/stream');
      evtSource.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          console.log('SSE message:', data);
          statusDiv.textContent = data.intervention_message || 'No issues detected.';
          if (issuesDiv) issuesDiv.textContent = (data.detected_issues || []).join('\n');
          if (debugPre) { debugPre.style.display='block'; debugPre.textContent = JSON.stringify(data,null,2); }
        } catch (err) { console.error('Invalid SSE data', err, e.data); }
      };
      evtSource.onerror = (err) => { console.warn('SSE error', err); };
    } catch (e) {
      console.warn('EventSource failed to initialize', e);
    }
  }
});
