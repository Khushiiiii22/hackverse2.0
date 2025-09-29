# cognitive_burnout_agent.py
import os
from typing import TypedDict, List
from datetime import datetime
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini (fallback if missing)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    model = None

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
    return sum(1 for q in queries if any(kw in q.lower() for kw in ["explain", "how to fix", "why"])) >= 3

def detect_circadian_misalignment(state: SessionState) -> bool:
    try:
        dt = datetime.fromisoformat(state["current_time"].replace("Z", "+00:00"))
        return dt.hour >= 22 or dt.hour < 6
    except:
        return False

def detect_cognitive_overload(state: SessionState) -> bool:
    return state.get("typing_speed_wpm", 100) < 35

# === Nodes ===
def analyze_signals(state: SessionState) -> dict:
    issues = []
    if detect_mental_fatigue(state): issues.append("Mental Fatigue: Prolonged focus without breaks.")
    if detect_frustration(state): issues.append("Frustration & Self-Doubt: High errors and corrections.")
    if detect_helplessness(state): issues.append("Helplessness: Repeated requests for explanations.")
    if detect_circadian_misalignment(state): issues.append("Circadian Misalignment: Working during low-energy hours.")
    if detect_cognitive_overload(state): issues.append("Cognitive Overload: Slow, hesitant typing.")
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
            message = "🧠 Take a breath. You're doing great—maybe step away for 5 minutes?"
    else:
        message = "🧠 Take a breath. You're doing great—maybe step away for 5 minutes?"
    return {"intervention_message": message}

# === FIXED GRAPH (NO LOOP) ===
def create_burnout_prevention_graph():
    workflow = StateGraph(SessionState)
    workflow.add_node("analyze", analyze_signals)
    workflow.add_node("respond", generate_intervention)
    workflow.set_entry_point("analyze")

    def should_respond(state: SessionState) -> str:
        return "respond" if state["detected_issues"] else END

    workflow.add_conditional_edges("analyze", should_respond, {"respond": "respond", END: END})
    workflow.add_edge("respond", END)  # Critical: no loop!
    return workflow.compile()

# === Test ===
if __name__ == "__main__":
    app = create_burnout_prevention_graph()
    test_state = {
        "typing_speed_wpm": 28.0,
        "spelling_errors": 8,
        "backspace_rate": 0.35,
        "session_duration_minutes": 75.0,
        "current_time": "2024-06-01T02:30:00",
        "recent_queries": ["explain async", "how to fix null", "why error"],
        "detected_issues": [],
        "intervention_message": ""
    }
    result = app.invoke(test_state)
    print("\n🔍 Detected Issues:")
    for issue in result["detected_issues"]:
        print(f" - {issue}")
    if result["intervention_message"]:
        print(f"\n💬 Suggested Intervention:\n{result['intervention_message']}")
    else:
        print("\n✅ No burnout signs detected.")