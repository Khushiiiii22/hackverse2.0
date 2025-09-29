def representation_agent(state):
    df = state["data"]
    penalty = 0
    insights = []
    if "sex" in df.columns:
        female_ratio = (df["sex"] == "Female").mean()
        if female_ratio < 0.3:
            penalty = 15
            insights.append("⚠️ Representation Bias: Only {:.0%} women → Oversample minority groups".format(female_ratio))
    return {"penalty_1": penalty, "insight_1": insights}


def label_bias_agent(state):
    df = state["data"]
    penalty = 0
    insights = []
    if "occupation" in df.columns and "sex" in df.columns:
        nurses = df[df["occupation"].str.contains("Exec", case=False, na=False)]
        if len(nurses) > 0:
            male_pct = (nurses["sex"] == "Male").mean()
            if male_pct > 0.9:
                penalty = 10
                insights.append("⚠️ Label Bias: Executive roles labeled as male-dominated")
    return {"penalty_2": penalty, "insight_2": insights}


def temporal_drift_agent(state):
    # Simulate: if filename has "201" → old data
    penalty = 10 if "201" in state.get("filename", "") else 0
    insights = ["📉 Temporal Drift: Data from 2010 → Add collection year"] if penalty else []
    return {"penalty_3": penalty, "insight_3": insights}


def pii_agent(state):
    df = state["data"]
    penalty = 0
    insights = []
    # Check values for SSN-like patterns
    for col in df.columns:
        sample = str(df[col].iloc[0]) if len(df) > 0 else ""
        if "SSN" in sample or (sample.isdigit() and len(sample) == 9):
            penalty = 20
            insights.append(f"🔒 PII Leak: '{col}' column → Anonymize or remove")
            break
    return {"penalty_4": penalty, "insight_4": insights}


def semantic_agent(state):
    df = state["data"]
    penalty = 0
    insights = []
    if "sex" in df.columns:
        unique_vals = set(str(v).lower() for v in df["sex"].unique())
        if len(unique_vals) > 2 and not {"male", "female"}.issuperset(unique_vals):
            penalty = 8
            insights.append("🔤 Semantic Inconsistency: Standardize 'sex' column values")
    return {"penalty_5": penalty, "insight_5": insights}


def completeness_agent(state):
    df = state["data"]
    required = ["age", "sex", "race"]
    missing = [c for c in required if c not in df.columns]
    penalty = len(missing) * 10
    insights = [f"❓ Missing Feature: No '{m}' → Add for fairness audit" for m in missing]
    return {"penalty_6": penalty, "insight_6": insights}


def synthetic_agent(state):
    df = state["data"]
    penalty = 5 if len(df) > 0 and df.select_dtypes(include='number').std().min() == 0 else 0
    insights = ["🤖 Synthetic Artifacts: Verify data source authenticity"] if penalty else []
    return {"penalty_7": penalty, "insight_7": insights}


def anomaly_agent(state):
    df = state["data"]
    penalty = 0
    insights = []
    if "age" in df.columns:
        if (df["age"] < 0).any() or (df["age"] > 120).any():
            penalty = 10
            insights.append("📈 Statistical Anomalies: Invalid age values → Clamp to [0,120]")
    return {"penalty_8": penalty, "insight_8": insights}


def provenance_agent(state):
    penalty = 10
    insights = ["📜 Provenance Ambiguity: No source or license → Add metadata"]
    return {"penalty_9": penalty, "insight_9": insights}


def concept_drift_agent(state):
    penalty = 8 if "201" in state.get("filename", "") else 0
    insights = ["🔄 Concept Drift: Feature meanings may be outdated → Version definitions"] if penalty else []
    return {"penalty_10": penalty, "insight_10": insights}

def sampling_agent(state):
    df = state["data"]
    penalty = 12 if "native-country" in df.columns and df["native-country"].nunique() == 1 else 0
    insights = ["📍 Sampling Bias: Single country → Expand to diverse regions"] if penalty else []
    return {"penalty_11": penalty, "insight_11": insights}


def duplicate_agent(state):
    df = state["data"]
    dupes = df.duplicated().sum()
    penalty = min(10, dupes // 100)
    insights = [f"🔄 Duplicates: {dupes} rows → Deduplicate"] if dupes > 0 else []
    return {"penalty_12": penalty, "insight_12": insights}


def representation_agent():
    return None


def temporal_drift_agent():
    return None