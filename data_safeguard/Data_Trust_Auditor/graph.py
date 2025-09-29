from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
import pandas as pd


# -------------------------------
# Graph State Definition
# -------------------------------
class GraphState(TypedDict):
    dataframe: pd.DataFrame
    filename: Optional[str]
    penalty_1: Optional[int]
    penalty_2: Optional[int]
    penalty_3: Optional[int]
    penalty_4: Optional[int]
    penalty_5: Optional[int]
    penalty_6: Optional[int]
    penalty_7: Optional[int]
    penalty_8: Optional[int]
    penalty_9: Optional[int]
    penalty_10: Optional[int]
    penalty_11: Optional[int]
    penalty_12: Optional[int]
    insight_1: Optional[List[str]]
    insight_2: Optional[List[str]]
    insight_3: Optional[List[str]]
    insight_4: Optional[List[str]]
    insight_5: Optional[List[str]]
    insight_6: Optional[List[str]]
    insight_7: Optional[List[str]]
    insight_8: Optional[List[str]]
    insight_9: Optional[List[str]]
    insight_10: Optional[List[str]]
    insight_11: Optional[List[str]]
    insight_12: Optional[List[str]]
    trust_score: Optional[int]
    all_insights: Optional[List[str]]


# -------------------------------
# Aggregation Node
# -------------------------------
def aggregate_results(state: GraphState):
    total_penalty = sum(state.get(f"penalty_{i}", 0) or 0 for i in range(1, 13))
    trust_score = max(0, 100 - total_penalty)
    all_insights = []
    for i in range(1, 13):
        insights = state.get(f"insight_{i}")
        if insights:
            all_insights.extend(insights)
    return {"trust_score": trust_score, "all_insights": all_insights}


# -------------------------------
# Agent Functions
# -------------------------------
def representation_agent(state: GraphState):
    df = state["dataframe"]
    penalty = 0
    insights = []
    if "sex" in df.columns:
        female_ratio = (df["sex"] == "Female").mean()
        if female_ratio < 0.3:
            penalty = 15
            insights.append(
                f"⚠️ Representation Bias: Only {female_ratio:.0%} women → Oversample minority groups"
            )
    return {"penalty_1": penalty, "insight_1": insights}


def label_bias_agent(state: GraphState):
    df = state["dataframe"]
    penalty = 0
    insights = []
    if "occupation" in df.columns and "sex" in df.columns:
        execs = df[df["occupation"].str.contains("Exec", case=False, na=False)]
        if len(execs) > 0:
            male_pct = (execs["sex"] == "Male").mean()
            if male_pct > 0.9:
                penalty = 10
                insights.append("⚠️ Label Bias: Executive roles labeled as male-dominated")
    return {"penalty_2": penalty, "insight_2": insights}


def temporal_drift_agent(state: GraphState):
    penalty = 10 if "201" in (state.get("filename") or "") else 0
    insights = ["📉 Temporal Drift: Data from 2010 → Add collection year"] if penalty else []
    return {"penalty_3": penalty, "insight_3": insights}


def pii_agent(state: GraphState):
    df = state["dataframe"]
    penalty = 0
    insights = []
    for col in df.columns:
        sample = str(df[col].iloc[0]) if len(df) > 0 else ""
        if "SSN" in sample or (sample.isdigit() and len(sample) == 9):
            penalty = 20
            insights.append(f"🔒 PII Leak: '{col}' column → Anonymize or remove")
            break
    return {"penalty_4": penalty, "insight_4": insights}


def semantic_agent(state: GraphState):
    df = state["dataframe"]
    penalty = 0
    insights = []
    if "sex" in df.columns:
        unique_vals = set(str(v).lower() for v in df["sex"].unique())
        if len(unique_vals) > 2 and not {"male", "female"}.issuperset(unique_vals):
            penalty = 8
            insights.append("🔤 Semantic Inconsistency: Standardize 'sex' column values")
    return {"penalty_5": penalty, "insight_5": insights}


def completeness_agent(state: GraphState):
    df = state["dataframe"]
    required = ["age", "sex", "race"]
    missing = [c for c in required if c not in df.columns]
    penalty = len(missing) * 10
    insights = [f"❓ Missing Feature: No '{m}' → Add for fairness audit" for m in missing]
    return {"penalty_6": penalty, "insight_6": insights}


def synthetic_agent(state: GraphState):
    df = state["dataframe"]
    penalty = 5 if len(df) > 0 and df.select_dtypes(include="number").std().min() == 0 else 0
    insights = ["🤖 Synthetic Artifacts: Verify data source authenticity"] if penalty else []
    return {"penalty_7": penalty, "insight_7": insights}


def anomaly_agent(state: GraphState):
    df = state["dataframe"]
    penalty = 0
    insights = []
    if "age" in df.columns:
        if (df["age"] < 0).any() or (df["age"] > 120).any():
            penalty = 10
            insights.append("📈 Statistical Anomalies: Invalid age values → Clamp to [0,120]")
    return {"penalty_8": penalty, "insight_8": insights}


def provenance_agent(state: GraphState):
    penalty = 10
    insights = ["📜 Provenance Ambiguity: No source or license → Add metadata"]
    return {"penalty_9": penalty, "insight_9": insights}


def concept_drift_agent(state: GraphState):
    penalty = 8 if "201" in (state.get("filename") or "") else 0
    insights = ["🔄 Concept Drift: Feature meanings may be outdated → Version definitions"] if penalty else []
    return {"penalty_10": penalty, "insight_10": insights}


def sampling_agent(state: GraphState):
    df = state["dataframe"]
    penalty = 12 if "native-country" in df.columns and df["native-country"].nunique() == 1 else 0
    insights = ["📍 Sampling Bias: Single country → Expand to diverse regions"] if penalty else []
    return {"penalty_11": penalty, "insight_11": insights}


def duplicate_agent(state: GraphState):
    df = state["dataframe"]
    dupes = df.duplicated().sum()
    penalty = min(10, dupes // 100)
    insights = [f"🔄 Duplicates: {dupes} rows → Deduplicate"] if dupes > 0 else []
    return {"penalty_12": penalty, "insight_12": insights}


# -------------------------------
# Build Workflow Graph
# -------------------------------
workflow = StateGraph(GraphState)

agents = [
    ("agent1", representation_agent),
    ("agent2", label_bias_agent),
    ("agent3", temporal_drift_agent),
    ("agent4", pii_agent),
    ("agent5", semantic_agent),
    ("agent6", completeness_agent),
    ("agent7", synthetic_agent),
    ("agent8", anomaly_agent),
    ("agent9", provenance_agent),
    ("agent10", concept_drift_agent),
    ("agent11", sampling_agent),
    ("agent12", duplicate_agent),
]

for name, func in agents:
    workflow.add_node(name, func)

workflow.set_entry_point("agent1")
for i in range(len(agents) - 1):
    workflow.add_edge(agents[i][0], agents[i + 1][0])

workflow.add_edge("agent12", "aggregate")
workflow.add_node("aggregate", aggregate_results)
workflow.add_edge("aggregate", END)

app = workflow.compile()
