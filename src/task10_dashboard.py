"""
task10_dashboard.py  --  Academic Risk Early-Warning Dashboard (Streamlit)

Run (from the project root, after running src/task10_early_warning.py):
    pip install streamlit
    streamlit run src/task10_dashboard.py

Three tabs:
  1. Department overview  - KPIs, tier mix, probation rate by tier
  2. Advisor worklist     - filterable list of flagged students with "why flagged"
  3. What-if calculator   - score one student profile and test an attendance improvement
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent.parent
SCORED = BASE / "outputs" / "tables" / "task10_scored_students.csv"
MODEL = BASE / "outputs" / "models" / "ews_model.joblib"
COLORS = {"Green": "#2E8B57", "Amber": "#E8A317", "Red": "#C0392B"}

st.set_page_config(page_title="Academic Risk Early-Warning", layout="wide")
st.title("Academic Risk Early-Warning Dashboard")
st.caption("BCSE Department decision-support prototype. The score is a screening aid for "
           "advisors, NOT a verdict on any student. Model AUC is about 0.70 (modest).")


@st.cache_data
def load():
    return pd.read_csv(SCORED)


@st.cache_resource
def load_model():
    return joblib.load(MODEL)


if not SCORED.exists() or not MODEL.exists():
    st.error("Run `python src/task10_early_warning.py` first to create the scored data and model.")
    st.stop()

df = load()
bundle = load_model()

# ---------------- sidebar: policy controls ----------------
st.sidebar.header("Policy settings")
amber = st.sidebar.slider("Amber cutoff (risk >=)", 0.10, 0.50, float(bundle["amber"]), 0.01)
red = st.sidebar.slider("Red cutoff (risk >=)", 0.20, 0.80, float(bundle["red"]), 0.01)
capacity = st.sidebar.number_input("Advisor capacity (students / month)", 10, 500, 60, 10)
if red <= amber:
    st.sidebar.warning("Red cutoff should be higher than Amber.")

df["tier"] = np.where(df.risk_prob >= red, "Red", np.where(df.risk_prob >= amber, "Amber", "Green"))

tab1, tab2, tab3 = st.tabs(["Department overview", "Advisor worklist", "What-if calculator"])

# ---------------- tab 1 ----------------
with tab1:
    n = len(df)
    c = st.columns(4)
    c[0].metric("Students scored", f"{n:,}")
    c[1].metric("Red (urgent)", int((df.tier == "Red").sum()))
    c[2].metric("Amber (monitor)", int((df.tier == "Amber").sum()))
    c[3].metric("Average attendance", f"{df.attendance.mean():.1f}%")

    left, right = st.columns(2)
    mix = df.tier.value_counts().reindex(["Green", "Amber", "Red"]).fillna(0)
    left.subheader("Students per tier")
    left.bar_chart(mix)
    rate = (df.assign(p=(df.probation == "Yes")).groupby("tier").p.mean()
              .reindex(["Green", "Amber", "Red"]) * 100)
    right.subheader("Actual probation rate by tier (%)")
    right.bar_chart(rate)
    st.info(f"At the current settings {int((df.tier == 'Red').sum())} students are Red. "
            f"Advisor capacity of {capacity}/month means the Red list takes about "
            f"{int(np.ceil((df.tier == 'Red').sum() / capacity))} month(s) to clear.")

# ---------------- tab 2 ----------------
with tab2:
    st.subheader("Advisor worklist (highest risk first)")
    f1, f2, f3 = st.columns(3)
    tiers = f1.multiselect("Tier", ["Red", "Amber", "Green"], default=["Red", "Amber"])
    sch = f2.selectbox("Scholarship", ["All", "Yes", "No"])
    max_att = f3.slider("Attendance at most (%)", 0, 100, 100)
    view = df[df.tier.isin(tiers) & (df.attendance <= max_att)]
    if sch != "All":
        view = view[view.scholarship == sch]
    cols = ["student_id", "tier", "risk_prob", "top_risk_drivers", "attendance",
            "semester", "scholarship", "teacher_consultancy"]
    st.dataframe(view.sort_values("risk_prob", ascending=False)[cols], use_container_width=True, hide_index=True)
    st.download_button("Download worklist (CSV)", view[cols].to_csv(index=False), "ews_worklist.csv")
    st.caption("Reasons list only levers an advisor can act on. Student IDs are synthetic row labels.")

# ---------------- tab 3 ----------------
with tab3:
    st.subheader("Score one student profile")
    feats = bundle["features"]
    a, b, c3 = st.columns(3)
    attendance = a.slider("Attendance (%)", 0, 100, 80)
    semester = a.slider("Semester", 1, 24, 6)
    credits = a.slider("Credits completed", 0, 150, 60)
    social = b.slider("Social media hours/day", 0, 20, 3)
    study_h = b.slider("Study hours/day", 0, 13, 2)
    sessions = b.slider("Study sessions/day", 0, 6, 2)
    skill = b.slider("Skill-dev hours/day", 0, 8, 1)
    english = c3.selectbox("English proficiency", ["Basic", "Intermediate", "Advance"], index=1)
    scholarship = c3.checkbox("Has scholarship")
    transport = c3.checkbox("Uses university transport")
    online = c3.checkbox("Online learning mode")
    consult = c3.checkbox("Attends teacher consultation")
    cocur = c3.checkbox("Co-curricular activities")

    vals = {"attendance": attendance, "study_hours": study_h, "study_sessions": sessions,
            "social_media_hours": social, "skill_dev_hours": skill, "semester": semester,
            "credits_completed": credits,
            "english_proficiency_ord": {"Basic": 0, "Intermediate": 1, "Advance": 2}[english],
            "scholarship_yes": int(scholarship), "uses_transport_yes": int(transport),
            "learning_mode_online": int(online), "teacher_consultancy_yes": int(consult),
            "co_curricular_yes": int(cocur)}
    x = pd.DataFrame([[vals[f] for f in feats]], columns=feats)
    p0 = float(bundle["pipeline"].predict_proba(x)[0, 1])
    tier0 = "Red" if p0 >= red else "Amber" if p0 >= amber else "Green"
    st.markdown(f"### Risk score: **{p0:.0%}**  |  Tier: **:{ {'Red':'red','Amber':'orange','Green':'green'}[tier0] }[{tier0}]**")

    bump = st.slider("What if attendance improved by (points)?", 0, 30, 10)
    x2 = x.copy()
    x2["attendance"] = min(100, attendance + bump)
    p1 = float(bundle["pipeline"].predict_proba(x2)[0, 1])
    st.write(f"Estimated risk after +{bump} pts attendance: **{p1:.0%}** (change {p1 - p0:+.0%}). "
             "This is an association from observational data, not a guaranteed causal effect.")
