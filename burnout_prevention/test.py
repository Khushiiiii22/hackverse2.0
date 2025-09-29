# test_now.py
import requests
from datetime import datetime

r = requests.post("http://localhost:8000/ingest", json={
    "typing_speed_wpm": 20,
    "session_duration_minutes": 70,
    "current_time": datetime.now().isoformat(),
    "recent_queries": ["explain", "how to fix", "why"]
}, timeout=10)

print(r.json()["intervention_message"])