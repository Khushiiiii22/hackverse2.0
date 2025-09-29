# test_sender.py
import requests
from datetime import datetime

payload = {
    "typing_speed_wpm": 28.0,
    "spelling_errors": 8,
    "backspace_rate": 0.35,
    "session_duration_minutes": 75.0,
    "current_time": datetime.now().isoformat(),
    "recent_queries": ["explain async", "how to fix null", "why error"]
}

try:
    response = requests.post("http://localhost:8000/ingest", json=payload)
    print("✅ Response:", response.json())
except Exception as e:
    print("❌ Error:", e)