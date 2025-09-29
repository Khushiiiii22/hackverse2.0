import os
from datetime import datetime
from typing import TypedDict, List
import google.generativeai as genai
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("models/gemini-2.5-flash")
else:
    model = None

# === State Schema ===
class SessionState(TypedDict):
    typing_speed_wpm: float
    spelling_errors: int
    backspace_rate: float
    session_duration_minutes: float
    current_time: str
    recent_queries: List[str]
    detected_issues: List[str]
    intervention_message: str

# === Detection Functions ===
def detect_mental_fatigue(state: SessionState) -> bool:
    return state.get("session_duration_minutes", 0) > 60

def detect_frustration(state: SessionState) -> bool:
    return state.get("spelling_errors", 0) > 5 and state.get("backspace_rate", 0) > 0.25

def detect_helplessness(state: SessionState) -> bool:
    queries = state.get("recent_queries", [])
    explain_count = sum(1 for q in queries if "explain" in q.lower() or "how to fix" in q.lower() or "why" in q.lower())
    return explain_count >= 3

def detect_circadian_misalignment(state: SessionState) -> bool:
    try:
        dt = datetime.fromisoformat(state["current_time"].replace("Z", "+00:00"))
        hour = dt.hour
        return hour >= 22 or hour < 6
    except Exception:
        return False

def detect_cognitive_overload(state: SessionState) -> bool:
    return state.get("typing_speed_wpm", 100) < 35

# === Graph Nodes ===
def analyze_signals(state: SessionState) -> dict:
    issues = []
    if detect_mental_fatigue(state):
        issues.append("Mental Fatigue: Prolonged focus without breaks.")
    if detect_frustration(state):
        issues.append("Frustration & Self-Doubt: High errors and corrections.")
    if detect_helplessness(state):
        issues.append("Helplessness: Repeated requests for explanations.")
    if detect_circadian_misalignment(state):
        issues.append("Circadian Misalignment: Working during low-energy hours.")
    if detect_cognitive_overload(state):
        issues.append("Cognitive Overload: Slow, hesitant typing.")
    return {"detected_issues": issues}

def generate_intervention(state: SessionState) -> dict:
    issues = state["detected_issues"]
    if not issues:
        return {"intervention_message": ""}
    if model:
        try:
            prompt = f"""
You are a compassionate AI wellness coach for developers.
The user is showing signs of cognitive burnout based on these issues:
{chr(10).join(f'- {issue}' for issue in issues)}

Respond with a short, empathetic, and actionable message (1–2 sentences).
Offer a micro-break suggestion or mindset shift. Be warm, not robotic.
            """.strip()
            response = model.generate_content(prompt)
            message = response.text.strip()
        except Exception as e:
            print(f"⚠️ Gemini error: {e}")
            message = "🧠 Take a breath. You're doing great—maybe step away for 5 minutes?"
    else:
        message = "🧠 Take a breath. You're doing great—maybe step away for 5 minutes?"
    return {"intervention_message": message}

# === Build Graph ===
def create_burnout_prevention_graph():
    workflow = StateGraph(SessionState)
    workflow.add_node("analyze", analyze_signals)
    workflow.add_node("respond", generate_intervention)
    workflow.set_entry_point("analyze")

    def should_respond(state: SessionState) -> str:
        return "respond" if state["detected_issues"] else END

    workflow.add_conditional_edges("analyze", should_respond, {"respond": "respond", END: END})
    workflow.add_edge("respond", END)

    return workflow.compile()

burnout_graph = create_burnout_prevention_graph()

def analyze_burnout(session_data: dict) -> dict:
    state = {
        "typing_speed_wpm": float(session_data.get("typing_speed_wpm", 40)),
        "spelling_errors": int(session_data.get("spelling_errors", 0)),
        "backspace_rate": float(session_data.get("backspace_rate", 0.0)),
        "session_duration_minutes": float(session_data.get("session_duration_minutes", 0)),
        "current_time": str(session_data.get("current_time", datetime.now().isoformat())),
        "recent_queries": list(session_data.get("recent_queries", [])),
        "detected_issues": [],
        "intervention_message": ""
    }
    result = burnout_graph.invoke(state)
    return {
        "detected_issues": result.get("detected_issues", []),
        "intervention_message": result.get("intervention_message", "")
    }
