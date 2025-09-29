import subprocess
import time
import sys
from datetime import datetime


def run_with_monitoring(script_path):
    start_time = time.time()
    process = subprocess.Popen([sys.executable, script_path],
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)

    stdout, stderr = process.communicate()

    # Infer frustration from errors
    error_count = len(stderr.split('\n')) - 1 if stderr else 0
    help_queries = ["explain" in line.lower() for line in stdout.split()]

    payload = {
        "typing_speed_wpm": 0,  # not available → estimate or skip
        "spelling_errors": 0,
        "backspace_rate": 0,
        "session_duration_minutes": (time.time() - start_time) / 60,
        "current_time": datetime.now().isoformat(),
        "recent_queries": ["user ran script with errors"] if error_count > 3 else []
    }
    send_to_backend(payload)