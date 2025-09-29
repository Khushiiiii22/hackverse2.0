# local_monitor.py
import time
import threading
from datetime import datetime
from pynput import keyboard
import requests

# Configuration
SEND_INTERVAL = 60  # seconds
# Point to Flask backend used by the frontend (port 5000)
BACKEND_URL = "http://127.0.0.1:5000/api/burnout"

# State
keystrokes = []
session_start = time.time()
monitoring = True

def on_press(key):
    if not monitoring:
        return
    try:
        # Only log printable characters and backspace
        if hasattr(key, 'char') and key.char:
            keystrokes.append(('char', time.time()))
        elif key == keyboard.Key.backspace:
            keystrokes.append(('backspace', time.time()))
    except:
        pass

def send_data():
    global keystrokes
    if not keystrokes:
        return

    now = time.time()
    total_time = now - session_start
    total_chars = len([k for k in keystrokes if k[0] == 'char'])
    backspace_count = len([k for k in keystrokes if k[0] == 'backspace'])

    wpm = (total_chars / 5) / (total_time / 60) if total_time > 0 else 0
    backspace_rate = backspace_count / len(keystrokes) if keystrokes else 0

    payload = {
        "typing_speed_wpm": round(wpm, 1),
        "spelling_errors": 0,  # hard to detect without editor access
        "backspace_rate": round(backspace_rate, 2),
        "session_duration_minutes": round(total_time / 60, 1),
        "current_time": datetime.now().isoformat(),
        "recent_queries": []  # would need AI chat integration
    }

    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=10)  # more generous
        if response.status_code == 200:
            data = response.json()
            # Print a console alert when the server suggests an intervention
            if data.get("intervention_message"):
                print("\n" + "="*60)
                print("🧠 BURNOUT ALERT:")
                print(data["intervention_message"])
                print("="*60 + "\n")
            else:
                print("Burnout check sent; no intervention suggested.")
    except Exception as e:
        print(f"⚠️  Failed to send data: {e}")

    # Reset for next window
    keystrokes = []

def periodic_sender():
    while monitoring:
        time.sleep(SEND_INTERVAL)
        if monitoring:
            send_data()

# Start listeners
listener = keyboard.Listener(on_press=on_press)
listener.start()

sender_thread = threading.Thread(target=periodic_sender, daemon=True)
sender_thread.start()

print("👁️  Burnout monitor active! Coding signals sent every 60s to your agent.")
print("💡 Keep this terminal open while you code in PyCharm/VS Code.")
print("🛑 Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    monitoring = False
    listener.stop()
    print("\n⏹️  Monitor stopped.")