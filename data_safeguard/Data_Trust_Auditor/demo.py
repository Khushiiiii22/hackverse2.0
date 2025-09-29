from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
import pandas as pd
import numpy as np
import re

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
    # Weighted penalties (PII & missing features are critical)
    weights = [1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1]
    total_penalty = sum((state.get(f"penalty_{i}", 0) or 0) * weights[i-1] for i in range(1, 13))
    trust_score = max(0, 100 - total_penalty)
    all_insights = []
    for i in range(1, 13):
        insights = state.get(f"insight_{i}")
        if insights:
            all_insights.extend(insights)
    return {"trust_score": trust_score, "all_insights": all_insights}

# -------------------------------
# Helper: Normalize Columns
# -------------------------------
def normalize_columns(df: pd.DataFrame):
    df = df.rename(columns={
        "gender": "sex",
        "race_ethnicity": "race"
    })
    return df

# -------------------------------
# Agent Functions
# -------------------------------

def representation_agent(state: GraphState):
    df = normalize_columns(state["dataframe"])
    penalty = 0
    insights = []
    if "sex" in df.columns:
        female_ratio = (df["sex"].str.lower() == "female").mean()
        if female_ratio < 0.3:
            penalty = 15
            insights.append(f"⚠️ Representation Bias: Only {female_ratio:.0%} women → Oversample minority groups")
    return {"penalty_1": penalty, "insight_1": insights}

def label_bias_agent(state: GraphState):
    df = normalize_columns(state["dataframe"])
    penalty = 0
    insights = []
    if "occupation" in df.columns and "sex" in df.columns:
        execs = df[df["occupation"].str.contains("Exec", case=False, na=False)]
        if len(execs) > 0:
            male_pct = (execs["sex"].str.lower() == "male").mean()
            if male_pct > 0.9:
                penalty = 10
                insights.append("⚠️ Label Bias: Executive roles labeled as male-dominated")
    return {"penalty_2": penalty, "insight_2": insights}

def temporal_drift_agent(state: GraphState):
    penalty = 0
    insights = []
    filename = state.get("filename") or ""
    if re.search(r"\b201\d\b", filename):
        penalty = 10
        insights.append("📉 Temporal Drift: Data from old year → Consider updating collection")
    return {"penalty_3": penalty, "insight_3": insights}

def pii_agent(state: GraphState):
    df = state["dataframe"]
    penalty = 0
    insights = []
    email_pattern = r"[^@]+@[^@]+\.[^@]+"
    phone_pattern = r"\b\d{10}\b"
    ssn_pattern = r"\b\d{3}-?\d{2}-?\d{4}\b"

    for col in df.columns:
        for val in df[col].astype(str).head(100):  # sample first 100 rows for speed
            if re.search(email_pattern, val) or re.search(phone_pattern, val) or re.search(ssn_pattern, val):
                penalty = 25
                insights.append(f"🔒 PII Leak: '{col}' column → Anonymize or remove")
                break
        if penalty > 0:
            break
    return {"penalty_4": penalty, "insight_4": insights}

def semantic_agent(state: GraphState):
    df = normalize_columns(state["dataframe"])
    penalty = 0
    insights = []
    categorical_cols = df.select_dtypes(include="object").columns
    for col in categorical_cols:
        vals = set(str(v).lower() for v in df[col].unique() if pd.notna(v))
        if len(vals) > 10 and df[col].nunique() > 1:
            penalty += 5
            insights.append(f"🔤 Semantic Inconsistency: Column '{col}' may have inconsistent values")
    return {"penalty_5": penalty, "insight_5": insights}

def completeness_agent(state: GraphState):
    df = normalize_columns(state["dataframe"])
    required = ["age", "sex", "race"]
    missing = [c for c in required if c not in df.columns]
    penalty = len(missing) * 15
    insights = [f"❓ Missing Feature: No '{m}' → Add for fairness audit" for m in missing]
    return {"penalty_6": penalty, "insight_6": insights}

def synthetic_agent(state: GraphState):
    df = state["dataframe"]
    penalty = 0
    insights = []
    # Check if any numeric column has zero variance
    numeric_cols = df.select_dtypes(include=np.number).columns
    if any(df[col].std() == 0 for col in numeric_cols):
        penalty = 5
        insights.append("🤖 Synthetic Artifacts: Verify numeric columns for repeated values")
    return {"penalty_7": penalty, "insight_7": insights}

def anomaly_agent(state: GraphState):
    df = state["dataframe"]
    penalty = 0
    insights = []
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]
        if len(outliers) > 0:
            penalty += 5
            insights.append(f"📈 Statistical Anomalies: Outliers detected in '{col}'")
    return {"penalty_8": penalty, "insight_8": insights}

def provenance_agent(state: GraphState):
    penalty = 10
    insights = ["📜 Provenance Ambiguity: No source or license → Add metadata"]
    return {"penalty_9": penalty, "insight_9": insights}

def concept_drift_agent(state: GraphState):
    filename = state.get("filename") or ""
    penalty = 0
    insights = []
    if re.search(r"\b201\d\b", filename):
        penalty = 8
        insights.append("🔄 Concept Drift: Feature meanings may be outdated → Version definitions")
    return {"penalty_10": penalty, "insight_10": insights}

def sampling_agent(state: GraphState):
    df = state["dataframe"]
    penalty = 0
    insights = []
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique() == 1:
            penalty += 5
            insights.append(f"📍 Sampling Bias: Single value in '{col}' → Consider diversity")
    return {"penalty_11": penalty, "insight_11": insights}

def duplicate_agent(state: GraphState):
    df = state["dataframe"]
    dupes = df.duplicated().sum()
    penalty = min(10, dupes // 10)  # more aggressive penalty
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

# -------------------------------
# Helper to Run Tests
# -------------------------------
def run_test(df, filename):
    result = app.invoke({
        "dataframe": df,
        "filename": filename
    })
    print(f"\n📊 Trust Score: {result['trust_score']}/100")
    print("🔍 Issues Found:")
    for insight in result["all_insights"]:
        print(f"● {insight}")
    print("="*50)

# -------------------------------
# Example Test Cases
# -------------------------------
if __name__ == "__main__":
    # Representation Bias
    df1 = pd.DataFrame({
        "gender": ["male"]*95 + ["female"]*5,
        "age": list(range(100)),
        "income": [">50K"]*50 + ["<=50K"]*50
    })
    run_test(df1, "representation_bias.csv")

    # Label Bias
    df2 = pd.DataFrame({
        "gender": ["male", "female"]*50,
        "occupation": ["Exec"]*95 + ["Staff"]*5
    })
    run_test(df2, "label_bias.csv")

    # Temporal Drift
    df3 = pd.DataFrame({
        "gender": ["male", "female"]*50,
        "income": [">50K", "<=50K"]*50
    })
    run_test(df3, "adult_2010_bias.csv")

    # PII
    df4 = pd.DataFrame({
        "email": ["test1@gmail.com", "test2@yahoo.com"],
        "income": [">50K", "<=50K"]
    })
    run_test(df4, "pii_test.csv")
