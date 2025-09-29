# File: Data_Trust_Auditor_app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

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
    weights = [1,1,1,3,1,3,1,1,1,1,1,1]  # PII & missing features are critical
    total_penalty = sum((state.get(f"penalty_{i}", 0) or 0) * weights[i-1] for i in range(1,13))
    trust_score = max(0, 100 - total_penalty)
    all_insights = []
    for i in range(1,13):
        insights = state.get(f"insight_{i}")
        if insights:
            all_insights.extend(insights)
    return {"trust_score": trust_score, "all_insights": all_insights}

# -------------------------------
# Helpers
# -------------------------------
def normalize_columns(df: pd.DataFrame):
    return df.rename(columns={"gender": "sex", "race_ethnicity": "race"})

def add_missing_columns(df: pd.DataFrame):
    if "age" not in df.columns:
        df["age"] = np.random.randint(18, 65, size=len(df))
    if "sex" not in df.columns:
        df["sex"] = np.random.choice(["male","female"], size=len(df))
    if "race" not in df.columns:
        df["race"] = np.random.choice(["White","Black","Asian","Hispanic","Other"], size=len(df))
    return df

def remove_pii(df: pd.DataFrame):
    pii_patterns = ["email", "phone", "ssn", "contact"]
    for col in df.columns:
        if any(p in col.lower() for p in pii_patterns):
            df = df.drop(columns=[col])
    return df

def fix_representation(df: pd.DataFrame):
    if "sex" in df.columns:
        female_ratio = (df["sex"].str.lower() == "female").mean()
        if female_ratio < 0.3:
            females = df[df["sex"].str.lower()=="female"]
            if len(females) > 0:
                needed = int(len(df)*0.3) - len(females)
                df = pd.concat([df, females.sample(needed, replace=True)], ignore_index=True)
    return df

def deduplicate(df: pd.DataFrame):
    return df.drop_duplicates(ignore_index=True)

# -------------------------------
# Agent Functions
# -------------------------------
def representation_agent(state: GraphState):
    df = normalize_columns(state["dataframe"])
    df = fix_representation(df)
    penalty = 0
    insights = []
    female_ratio = (df["sex"].str.lower() == "female").mean() if "sex" in df.columns else 0
    if female_ratio < 0.3:
        penalty = 15
        insights.append(f"⚠️ Representation Bias: Only {female_ratio:.0%} women → Oversampled minority groups")
    state["dataframe"] = df
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
    df = remove_pii(state["dataframe"])
    state["dataframe"] = df
    penalty = 0
    insights = []
    # Extra check in case some PII remains
    for col in df.columns:
        for val in df[col].astype(str).head(100):
            if re.search(r"[^@]+@[^@]+\.[^@]+", val) or re.search(r"\b\d{10}\b", val) or re.search(r"\b\d{3}-?\d{2}-?\d{4}\b", val):
                penalty = 25
                insights.append(f"🔒 PII Leak: '{col}' column → Should anonymize/remove")
                break
        if penalty>0: break
    return {"penalty_4": penalty, "insight_4": insights}

def semantic_agent(state: GraphState):
    df = normalize_columns(state["dataframe"])
    penalty = 0
    insights = []
    for col in df.select_dtypes(include="object").columns:
        vals = set(str(v).lower() for v in df[col].unique() if pd.notna(v))
        if len(vals) > 10 and df[col].nunique() > 1:
            penalty += 5
            insights.append(f"🔤 Semantic Inconsistency: Column '{col}' may have inconsistent values")
    return {"penalty_5": penalty, "insight_5": insights}

def completeness_agent(state: GraphState):
    df = add_missing_columns(state["dataframe"])
    state["dataframe"] = df
    required = ["age","sex","race"]
    missing = [c for c in required if c not in df.columns]
    penalty = len(missing)*15
    insights = [f"❓ Missing Feature: No '{m}' → Added automatically" for m in missing]
    return {"penalty_6": penalty, "insight_6": insights}

def synthetic_agent(state: GraphState):
    df = state["dataframe"]
    penalty = 0
    insights = []
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].std()==0:
            df[col] += np.random.normal(0,0.01,len(df))
            penalty=5
            insights.append(f"🤖 Synthetic Artifacts: Minor noise added to '{col}'")
    state["dataframe"]=df
    return {"penalty_7": penalty, "insight_7": insights}

def anomaly_agent(state: GraphState):
    df = state["dataframe"]
    penalty = 0
    insights = []
    for col in df.select_dtypes(include=np.number).columns:
        q1,q3 = df[col].quantile([0.25,0.75])
        iqr = q3-q1
        outliers = df[(df[col]<q1-1.5*iqr)|(df[col]>q3+1.5*iqr)]
        if len(outliers)>0:
            penalty+=5
            insights.append(f"📈 Statistical Anomalies: Outliers in '{col}'")
    return {"penalty_8": penalty, "insight_8": insights}

def provenance_agent(state: GraphState):
    penalty = 10
    insights = ["📜 Provenance Ambiguity: No source/license → Add metadata"]
    return {"penalty_9": penalty, "insight_9": insights}

def concept_drift_agent(state: GraphState):
    filename = state.get("filename") or ""
    penalty=0
    insights=[]
    if re.search(r"\b201\d\b", filename):
        penalty=8
        insights.append("🔄 Concept Drift: Feature meanings may be outdated")
    return {"penalty_10": penalty, "insight_10": insights}

def sampling_agent(state: GraphState):
    df = state["dataframe"]
    penalty=0
    insights=[]
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique()==1:
            penalty+=5
            insights.append(f"📍 Sampling Bias: Single value in '{col}' → Consider diversity")
    return {"penalty_11": penalty, "insight_11": insights}

def duplicate_agent(state: GraphState):
    df = state["dataframe"]
    dupes = df.duplicated().sum()
    df = deduplicate(df)
    penalty = min(10, dupes//10)
    insights = [f"🔄 Duplicates: {dupes} rows → Removed automatically"] if dupes>0 else []
    state["dataframe"] = df
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
for i in range(len(agents)-1):
    workflow.add_edge(agents[i][0], agents[i+1][0])
workflow.add_edge("agent12","aggregate")
workflow.add_node("aggregate", aggregate_results)
workflow.add_edge("aggregate", END)
app = workflow.compile()

# -------------------------------
# Function to run test
# -------------------------------
def run_test(df, filename, apply_corrections=True):
    """
    Runs the workflow:
    - If apply_corrections=False: just compute penalties/trust score without changing df
    - If apply_corrections=True: apply dataset corrections and recompute trust score
    """
    state = {"dataframe": df.copy(), "filename": filename}

    if apply_corrections:
        # This will modify df as needed
        result = app.invoke(state)
    else:
        # Only calculate penalties, ignore corrections
        # Temporarily patch agents to skip dataset changes
        original_agents = [n[1] for n in agents]

        def no_change_agent(agent_func):
            def wrapper(state):
                result = agent_func(state.copy())
                # Keep original dataframe
                result["dataframe"] = state["dataframe"]
                return result

            return wrapper

        # Patch agents
        for i, (name, func) in enumerate(agents):
            workflow.update_node(name, no_change_agent(func))
        result = app.invoke(state)
        # Restore agents
        for i, (name, func) in enumerate(agents):
            workflow.update_node(name, func)

    trust_score = result.get("trust_score", 0)
    all_insights = result.get("all_insights", [])
    return result.get("dataframe", df), trust_score, all_insights


# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="📊 Data Trust Auditor", layout="wide")
st.title("📊 AI Data Trust Auditor & Auto-Corrector")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Original Dataset Preview")
    st.dataframe(df.head(10))

    corrected_df, trust_score, insights = run_test(df, uploaded_file.name)

    st.write(f"### ✅ Corrected Dataset (Updated Trust Score: {trust_score}/100)")
    st.dataframe(corrected_df.head(10))

    st.write("### 🔍 Issues Found / Fixed")
    for insight in insights:
        st.write(f"● {insight}")

    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Corrected CSV",
        data=convert_df_to_csv(corrected_df),
        file_name="corrected_dataset.csv",
        mime="text/csv"
    )

    st.success(f"🎉 Trust Score after corrections: {trust_score}/100")
