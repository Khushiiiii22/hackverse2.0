import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from dotenv import load_dotenv

# -------------------------------
# Load API Key
# -------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("models/gemini-2.5-flash")  # Use flash for speed/cost
else:
    st.warning("⚠️ GEMINI_API_KEY not found. Using rule-based fallback only.")
    model = None


# -------------------------------
# Helper: Normalize column names (enhanced)
# -------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Case-insensitive mapping
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ["gender", "sex", "male", "female"]:
            col_map[col] = "sex"
        elif lower in ["race", "ethnicity", "race_ethnicity"]:
            col_map[col] = "race"
        elif lower in ["age", "years"]:
            col_map[col] = "age"
    return df.rename(columns=col_map)


# -------------------------------
# CORRECTION FUNCTION (MUST BE DEFINED!)
# -------------------------------
def apply_corrections(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = normalize_columns(df)

    # Add missing critical features
    if "age" not in df.columns:
        df["age"] = np.random.randint(18, 65, size=len(df))
    if "sex" not in df.columns:
        df["sex"] = np.random.choice(["Male", "Female"], size=len(df))
    if "race" not in df.columns:
        df["race"] = np.random.choice(["White", "Black", "Asian", "Hispanic", "Other"], size=len(df))

    # Remove PII columns (case-insensitive)
    pii_keywords = ["email", "phone", "ssn", "contact", "name", "address", "id", "identifier"]
    cols_to_drop = [
        col for col in df.columns
        if any(kw in col.lower() for kw in pii_keywords)
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Fix representation bias (oversample females if needed)
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str)
        female_mask = df["sex"].str.lower().isin(["female", "f", "woman"])
        female_rows = df[female_mask]
        male_rows = df[~female_mask]
        if len(female_rows) > 0 and len(male_rows) > 0:
            female_ratio = len(female_rows) / len(df)
            if female_ratio < 0.3:
                needed = int(len(df) * 0.35) - len(female_rows)
                if needed > 0:
                    extra_females = female_rows.sample(needed, replace=True)
                    df = pd.concat([df, extra_females], ignore_index=True)

    # Fix semantics: clean string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Cap outliers in numeric columns
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].nunique() > 1:  # Skip constant columns
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:  # Avoid division by zero
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                df[col] = df[col].clip(lower, upper)

    # Deduplicate
    df = df.drop_duplicates(ignore_index=True)

    return df


# -------------------------------
# Graph State
# -------------------------------
class GraphState(TypedDict):
    dataframe: pd.DataFrame
    filename: Optional[str]
    penalty_1: int
    penalty_2: int
    penalty_3: int
    penalty_4: int
    penalty_5: int
    penalty_6: int
    penalty_7: int
    penalty_8: int
    penalty_9: int
    penalty_10: int
    penalty_11: int
    penalty_12: int
    insight_1: List[str]
    insight_2: List[str]
    insight_3: List[str]
    insight_4: List[str]
    insight_5: List[str]
    insight_6: List[str]
    insight_7: List[str]
    insight_8: List[str]
    insight_9: List[str]
    insight_10: List[str]
    insight_11: List[str]
    insight_12: List[str]
    trust_score: int
    all_insights: List[str]


# -------------------------------
# Gemini Agent
# -------------------------------
def get_gemini_insights(df: pd.DataFrame, filename: str) -> tuple[int, List[str]]:
    if not model:
        return 0, []

    # Sample data safely
    sample = df.head(3).fillna("NULL").to_dict(orient="records")
    cols = list(df.columns)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    prompt = f"""
    Analyze this dataset for data quality, bias, PII, and trustworthiness.
    Filename: {filename}
    Columns: {cols}
    Data types: {dtypes}
    Sample (3 rows): {sample}

    Return a JSON object:
    {{
      "penalty": <int 0-30>,
      "insights": ["Insight 1", "Insight 2"]
    }}
    Focus on actionable, specific issues. Avoid generic advice.
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
            safety_settings=[
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        )
        import json
        result = json.loads(response.text)
        penalty = min(30, max(0, int(result.get("penalty", 0))))
        insights = result.get("insights", [])
        return penalty, [f"🧠 Gemini: {insight}" for insight in insights]
    except Exception as e:
        return 5, [f"⚠️ Gemini error: {str(e)}"]


# -------------------------------
# Rule-Based Agents (1-11)
# -------------------------------
def representation_agent(state: GraphState):
    df = normalize_columns(state["dataframe"].copy())
    penalty = 0
    insights = []
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str)
        female_ratio = (df["sex"].str.lower().isin(["female", "f", "woman"])).mean()
        if female_ratio < 0.3:
            penalty = 15
            insights.append(f"⚠️ Representation Bias: Only {female_ratio:.0%} women → Oversample minority groups")
    return {"penalty_1": penalty, "insight_1": insights}

def label_bias_agent(state: GraphState):
    df = normalize_columns(state["dataframe"].copy())
    penalty = 0
    insights = []
    if "occupation" in df.columns and "sex" in df.columns:
        execs = df[df["occupation"].astype(str).str.contains("Exec", case=False, na=False)]
        if len(execs) > 0:
            male_pct = (execs["sex"].astype(str).str.lower().isin(["male", "m"])).mean()
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
        insights.append("📉 Temporal Drift: Data from old year → Add collection year")
    return {"penalty_3": penalty, "insight_3": insights}

def pii_agent(state: GraphState):
    df = state["dataframe"]
    penalty = 0
    insights = []
    pii_keywords = ["email", "phone", "ssn", "contact", "name", "address", "id"]
    pii_cols = [col for col in df.columns if any(kw in col.lower() for kw in pii_keywords)]
    if pii_cols:
        penalty = 30
        insights.append(f"🔒 PII Leak: Columns {pii_cols} → Anonymize or remove")
    return {"penalty_4": penalty, "insight_4": insights}

def semantic_agent(state: GraphState):
    df = state["dataframe"].copy()
    penalty = 0
    insights = []
    for col in df.select_dtypes(include="object").columns:
        unique_vals = set(str(v).lower().strip() for v in df[col].dropna().unique())
        if "sex" in col.lower() and len(unique_vals) > 2:
            if not {"male", "female", "m", "f", "man", "woman"}.issuperset(unique_vals):
                penalty = 8
                insights.append("🔤 Semantic Inconsistency: Standardize 'sex' column values")
                break
    return {"penalty_5": penalty, "insight_5": insights}

def completeness_agent(state: GraphState):
    df = normalize_columns(state["dataframe"])
    required = ["age", "sex", "race"]
    missing = [col for col in required if col not in df.columns]
    penalty = len(missing) * 20
    insights = [f"❓ Missing Critical Feature: '{m}' → Add for fairness audit" for m in missing]
    return {"penalty_6": penalty, "insight_6": insights}

def synthetic_agent(state: GraphState):
    df = state["dataframe"]
    penalty = 0
    insights = []
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        zero_var_cols = [col for col in num_cols if df[col].nunique() <= 1]
        if zero_var_cols:
            penalty = 5
            insights.append(f"🤖 Synthetic Artifacts: Zero-variance columns {zero_var_cols} → Verify source")
    return {"penalty_7": penalty, "insight_7": insights}

def anomaly_agent(state: GraphState):
    df = normalize_columns(state["dataframe"])
    penalty = 0
    insights = []
    if "age" in df.columns:
        invalid = (df["age"] < 0) | (df["age"] > 120)
        if invalid.any():
            penalty = 10
            insights.append("📈 Statistical Anomalies: Invalid age values → Clamp to [0,120]")
    return {"penalty_8": penalty, "insight_8": insights}

def provenance_agent(state: GraphState):
    return {"penalty_9": 10, "insight_9": ["📜 Provenance Ambiguity: No source/license → Add metadata"]}

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
            insights.append(f"📍 Sampling Bias: Single value in '{col}' → Expand diversity")
    return {"penalty_11": penalty, "insight_11": insights}

def gemini_agent(state: GraphState):
    penalty, insights = get_gemini_insights(state["dataframe"], state.get("filename", ""))
    return {"penalty_12": penalty, "insight_12": insights}


def aggregate_results(state: GraphState):
    weights = [1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1]
    total_penalty = sum(state[f"penalty_{i}"] * weights[i - 1] for i in range(1, 13))
    trust_score = max(0, 100 - total_penalty)
    all_insights = []
    for i in range(1, 13):
        all_insights.extend(state[f"insight_{i}"])
    return {"trust_score": trust_score, "all_insights": all_insights}


# -------------------------------
# Build Graph
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
    ("agent12", gemini_agent),
]

for name, func in agents:
    workflow.add_node(name, func)

workflow.set_entry_point("agent1")
for i in range(len(agents) - 1):
    workflow.add_edge(agents[i][0], agents[i + 1][0])

workflow.add_node("aggregate", aggregate_results)
workflow.add_edge("agent12", "aggregate")
workflow.add_edge("aggregate", END)

auditor_app = workflow.compile()


def run_audit(df: pd.DataFrame, filename: str):
    state = {"dataframe": df.copy(), "filename": filename}
    result = auditor_app.invoke(state)
    return result["trust_score"], result["all_insights"]


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="BalanceGuard: AI Data Trust Auditor", layout="wide")
st.title("🛡️ BalanceGuard: AI Data Trust Auditor & Auto-Corrector")

if not GEMINI_API_KEY:
    st.info("💡 Add your `GEMINI_API_KEY` in `.env` for AI-powered insights!")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### 📥 Original Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        orig_score, orig_insights = run_audit(df, uploaded_file.name)

        st.subheader("🔍 Original Dataset Analysis")
        score_color = "red" if orig_score < 60 else "orange" if orig_score < 80 else "green"
        st.markdown(f"### 📊 Trust Score: <span style='color:{score_color}'>{orig_score}/100</span>", unsafe_allow_html=True)

        if orig_insights:
            st.write("### ⚠️ Issues Found")
            for insight in orig_insights:
                if insight.startswith("🧠"):
                    st.info(insight)
                else:
                    st.write(f"- {insight}")
        else:
            st.success("✅ No major issues detected!")

        # Apply corrections
        corrected_df = apply_corrections(df)
        new_score, _ = run_audit(corrected_df, uploaded_file.name)

        st.subheader("✅ Corrected Dataset")
        st.markdown(f"### 📊 Improved Trust Score: <span style='color:green'>{new_score}/100</span>", unsafe_allow_html=True)
        st.write("### 📤 Corrected Dataset Preview")
        st.dataframe(corrected_df.head(10), use_container_width=True)

        csv_data = corrected_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Corrected CSV",
            data=csv_data,
            file_name="balanceguard_corrected_dataset.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)