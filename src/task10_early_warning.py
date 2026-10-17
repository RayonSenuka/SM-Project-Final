"""
task10_early_warning.py
Task 10: Industry Innovation Proposal -- Academic Risk Early-Warning System (EWS)

What this script does (the "engine" behind the proposed dashboard):
  1. Builds the DEPLOYMENT model: a standardized logistic regression for
     P(probation) using only ACTIONABLE / ETHICALLY DEFENSIBLE predictors
     (gender and family income are removed -- see FAIRNESS CHECK).
  2. Scores every student with an out-of-fold (cross-validated) probability so the
     numbers are honest (no student is scored by a model that saw them).
  3. Converts probabilities into 3 risk tiers (Green / Amber / Red) and checks
     that the tiers really separate high-risk from low-risk students.
  4. Compares operating strategies with an expected-cost analysis (links to the
     Bayesian decision-making section of Task 8).
  5. Runs a what-if attendance scenario for the Red/Amber group.
  6. Saves: scored students CSV, fitted model (.joblib), figures, results log.

Run (from the project root):
    python src/task10_early_warning.py

Outputs:
    outputs/tables/task10_*.csv, task10_results.txt
    outputs/figures/task10_*.png
    outputs/models/ews_model.joblib
"""

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
import model_prep  # noqa: E402

FIG_DIR = BASE / "outputs" / "figures"
TAB_DIR = BASE / "outputs" / "tables"
MOD_DIR = BASE / "outputs" / "models"
for d in (FIG_DIR, TAB_DIR, MOD_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---- Design choices (state these in the report) ---------------------------
EXCLUDED_FOR_FAIRNESS = ["gender_male", "log_family_income"]
FEATURES = [c for c in model_prep.ALL_PREDICTORS if c not in EXCLUDED_FOR_FAIRNESS]

AMBER_CUTOFF = 0.25   # P(probation) >= 25%  -> Amber ("monitor / light-touch outreach")
RED_CUTOFF = 0.40     # P(probation) >= 40%  -> Red   ("advisor meeting within 2 weeks")
COST_MISS = 5         # cost of missing an at-risk student (same illustrative matrix as Task 8)
COST_FALSE_ALARM = 1  # cost of one unnecessary advising slot


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def assign_tier(p):
    return np.where(p >= RED_CUTOFF, "Red", np.where(p >= AMBER_CUTOFF, "Amber", "Green"))


def main():
    X_full, y, df = model_prep.get_classification_data()
    df = df.copy()
    df["student_id"] = [f"S{str(i + 1).zfill(4)}" for i in range(len(df))]  # dataset has no ID column
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    # ------------------------------------------------------------------
    banner("10A. FAIRNESS CHECK -- does dropping gender & family income hurt accuracy?")
    rows = []
    oof = {}
    for name, cols in [("Full 15 predictors", model_prep.ALL_PREDICTORS),
                       (f"Deployment {len(FEATURES)} predictors (no gender/income)", FEATURES)]:
        p = cross_val_predict(make_model(), df[cols].astype(float).values, y, cv=cv,
                              method="predict_proba")[:, 1]
        oof[name] = p
        rows.append({"model": name, "cv_auc": round(roc_auc_score(y, p), 4),
                     "brier": round(brier_score_loss(y, p), 4)})
    fair = pd.DataFrame(rows)
    print(fair.to_string(index=False))
    fair.to_csv(TAB_DIR / "task10_fairness_check.csv", index=False)
    print("-> Removing gender and family income costs essentially no predictive power, so the "
          "deployed model does not need them (ethical + practical win).")
    p_oof = oof[f"Deployment {len(FEATURES)} predictors (no gender/income)"]
    df["risk_prob"] = p_oof
    df["risk_tier"] = assign_tier(p_oof)

    # ------------------------------------------------------------------
    banner("10B. Do the risk tiers separate students? (5-fold out-of-fold scores)")
    tier_tab = (df.groupby("risk_tier")
                  .agg(students=("probation_bin", "size"),
                       actual_probation_rate=("probation_bin", "mean"),
                       mean_predicted_risk=("risk_prob", "mean"),
                       mean_attendance=("attendance", "mean"))
                  .reindex(["Green", "Amber", "Red"]))
    tier_tab["share_of_students"] = tier_tab["students"] / len(df)
    tier_tab["probation_cases_captured"] = (tier_tab["actual_probation_rate"] * tier_tab["students"]).round(0)
    tier_tab = tier_tab.round(3)
    print(tier_tab.to_string())
    tier_tab.to_csv(TAB_DIR / "task10_tier_summary.csv")
    base = y.mean()
    red = tier_tab.loc["Red"]
    print(f"\nBase probation rate = {base:.1%}. Red tier probation rate = {red.actual_probation_rate:.1%} "
          f"({red.actual_probation_rate / base:.1f}x base rate) while covering only "
          f"{red.share_of_students:.1%} of students.")
    ra = tier_tab.loc[["Red", "Amber"]]
    print(f"Red+Amber together = {ra.share_of_students.sum():.1%} of students and capture "
          f"{ra.probation_cases_captured.sum() / y.sum():.1%} of all probation cases.")

    # Top-k lift table (useful when advisors have limited capacity)
    order = np.argsort(-p_oof)
    lift_rows = []
    for k in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        n = int(round(k * len(y)))
        hit = y[order[:n]].sum()
        lift_rows.append({"top_share_flagged": f"{int(k * 100)}%", "students_flagged": n,
                          "precision": round(hit / n, 3), "recall": round(hit / y.sum(), 3),
                          "lift_vs_random": round((hit / n) / base, 2)})
    lift = pd.DataFrame(lift_rows)
    print("\nCapacity table: if advisors can only see the top-k% riskiest students\n" + lift.to_string(index=False))
    lift.to_csv(TAB_DIR / "task10_capacity_lift.csv", index=False)

    # ------------------------------------------------------------------
    banner("10C. Expected-cost comparison of operating strategies (Bayesian decision view)")
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    cost_rows = [
        {"strategy": "Do nothing (reactive, status quo)", "students_flagged": 0,
         "missed_at_risk": n_pos, "false_alarms": 0,
         "total_cost": n_pos * COST_MISS},
        {"strategy": "Intervene with everyone", "students_flagged": len(y),
         "missed_at_risk": 0, "false_alarms": n_neg,
         "total_cost": n_neg * COST_FALSE_ALARM},
    ]
    for label, cut in [("EWS: Red only", RED_CUTOFF), ("EWS: Red + Amber", AMBER_CUTOFF),
                       (f"EWS: cost-optimal cutoff (1/(1+{COST_MISS}) = {1 / (1 + COST_MISS):.3f})",
                        COST_FALSE_ALARM / (COST_FALSE_ALARM + COST_MISS))]:
        flag = p_oof >= cut
        missed = int(((~flag) & (y == 1)).sum())
        fa = int((flag & (y == 0)).sum())
        cost_rows.append({"strategy": label, "students_flagged": int(flag.sum()),
                          "missed_at_risk": missed, "false_alarms": fa,
                          "total_cost": missed * COST_MISS + fa * COST_FALSE_ALARM})
    cost = pd.DataFrame(cost_rows)
    cost["cost_saving_vs_do_nothing_%"] = ((cost.loc[0, "total_cost"] - cost["total_cost"]) /
                                           cost.loc[0, "total_cost"] * 100).round(1)
    print(f"Illustrative cost matrix: miss = {COST_MISS}, false alarm = {COST_FALSE_ALARM} "
          "(set with the department, NOT derived from data)")
    print(cost.to_string(index=False))
    cost.to_csv(TAB_DIR / "task10_cost_comparison.csv", index=False)
    print("\nHONEST NOTE: because the model's AUC is only ~0.70, the cost-optimal cutoff flags a large "
          "share of students. In practice advisor capacity decides the cutoff -> the 3-tier design "
          "(Red = urgent, Amber = monitor) is the realistic operating mode.")

    # ------------------------------------------------------------------
    banner("10D. Fit final deployment model on all students + driver (odds-ratio) table")
    final = make_model().fit(df[FEATURES].astype(float).values, y)
    scaler, lr = final.named_steps["standardscaler"], final.named_steps["logisticregression"]
    coef = pd.DataFrame({"feature": FEATURES, "std_coef": lr.coef_[0],
                         "odds_ratio_per_1SD": np.exp(lr.coef_[0]),
                         "raw_coef": lr.coef_[0] / scaler.scale_}).sort_values("std_coef")
    print(coef.round(4).to_string(index=False))
    coef.to_csv(TAB_DIR / "task10_final_model_coefficients.csv", index=False)
    joblib.dump({"pipeline": final, "features": FEATURES, "amber": AMBER_CUTOFF, "red": RED_CUTOFF},
                MOD_DIR / "ews_model.joblib")

    # per-student "reason codes": the 2 features pushing risk UP the most
    Z = scaler.transform(df[FEATURES].astype(float).values) * lr.coef_[0]
    pretty = {"attendance": "Low attendance", "study_hours": "Few study hours",
              "study_sessions": "Study-session pattern", "social_media_hours": "High social-media time",
              "skill_dev_hours": "Skill-development time", "semester": "Later semester / progression",
              "credits_completed": "Low credits completed", "english_proficiency_ord": "Lower English proficiency",
              "scholarship_yes": "No scholarship", "uses_transport_yes": "No university transport",
              "learning_mode_online": "Online learning mode", "teacher_consultancy_yes": "No teacher consultation",
              "co_curricular_yes": "Co-curricular load"}
    # Reason codes use ONLY levers an advisor can act on. semester / credits_completed still
    # feed the score (they carry exposure-to-risk) but are not shown as "reasons".
    actionable = ["attendance", "study_hours", "social_media_hours", "english_proficiency_ord",
                  "scholarship_yes", "uses_transport_yes", "learning_mode_online",
                  "teacher_consultancy_yes"]
    act_idx = [FEATURES.index(a) for a in actionable]
    reasons = []
    for row in Z:
        ranked = sorted(act_idx, key=lambda i: -row[i])[:2]
        reasons.append(" + ".join(pretty[FEATURES[i]] for i in ranked if row[i] > 0.02) or "No dominant actionable driver")
    df["top_risk_drivers"] = reasons

    keep = ["student_id", "risk_prob", "risk_tier", "top_risk_drivers", "attendance", "semester",
            "credits_completed", "social_media_hours", "study_hours", "english_proficiency",
            "scholarship", "learning_mode", "teacher_consultancy", "probation"]
    out = df[keep].sort_values("risk_prob", ascending=False)
    out["risk_prob"] = out["risk_prob"].round(3)
    out.to_csv(TAB_DIR / "task10_scored_students.csv", index=False)
    print(f"\nSaved scored list for {len(out)} students -> outputs/tables/task10_scored_students.csv")
    print(out.head(8).to_string(index=False))

    # ------------------------------------------------------------------
    banner("10E. What-if scenario: raise attendance of Red/Amber students (ASSOCIATIONAL, not causal)")
    att_i = FEATURES.index("attendance")
    flagged = df["risk_tier"].isin(["Red", "Amber"]).values
    Xf = df[FEATURES].astype(float).values.copy()
    base_p = final.predict_proba(Xf)[:, 1]
    scen_rows = []
    for bump in (5, 10, 15):
        X2 = Xf.copy()
        X2[flagged, att_i] = np.minimum(X2[flagged, att_i] + bump, 100)
        p2 = final.predict_proba(X2)[:, 1]
        exp_before = base_p[flagged].sum()
        exp_after = p2[flagged].sum()
        scen_rows.append({"attendance_increase_pts": bump, "flagged_students": int(flagged.sum()),
                          "expected_probation_cases_before": round(exp_before, 1),
                          "expected_probation_cases_after": round(exp_after, 1),
                          "cases_avoided": round(exp_before - exp_after, 1),
                          "relative_reduction_%": round((exp_before - exp_after) / exp_before * 100, 1)})
    scen = pd.DataFrame(scen_rows)
    print(scen.to_string(index=False))
    scen.to_csv(TAB_DIR / "task10_attendance_scenario.csv", index=False)
    print("Caveat: the model shows association; a causal claim would need the experimental design "
          "discussed in Task 6 (e.g., an RCBD pilot of attendance coaching).")

    # ------------------------------------------------------------------
    banner("Making figures")
    colors = {"Green": "#2E8B57", "Amber": "#E8A317", "Red": "#C0392B"}

    # Fig 1: tier validation
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    tt = tier_tab.reset_index()
    ax[0].bar(tt["risk_tier"], tt["students"], color=[colors[t] for t in tt["risk_tier"]])
    ax[0].set_title("Students per risk tier")
    ax[0].set_ylabel("Number of students")
    for i, v in enumerate(tt["students"]):
        ax[0].text(i, v + 8, f"{v} ({v / len(df):.0%})", ha="center")
    ax[1].bar(tt["risk_tier"], tt["actual_probation_rate"] * 100, color=[colors[t] for t in tt["risk_tier"]])
    ax[1].axhline(base * 100, ls="--", color="gray")
    ax[1].text(2.45, base * 100 + 1.2, f"overall {base:.0%}", ha="right", color="gray")
    ax[1].set_title("Actual probation rate by tier (out-of-fold)")
    ax[1].set_ylabel("% on probation")
    for i, v in enumerate(tt["actual_probation_rate"] * 100):
        ax[1].text(i, v + 1, f"{v:.0f}%", ha="center")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task10_risk_tiers.png", dpi=200)
    plt.close()

    # Fig 2: calibration
    frac, mean_pred = calibration_curve(y, p_oof, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(mean_pred, frac, "o-", color="#1F4E79", label="EWS model (5-fold OOF)")
    ax.set_xlabel("Predicted probability of probation")
    ax.set_ylabel("Observed probation rate")
    ax.set_title("Calibration of the risk score")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task10_calibration.png", dpi=200)
    plt.close()

    # Fig 3: cost comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    cc = cost.copy()
    cc["short"] = ["Do nothing", "Intervene all", "EWS Red", "EWS Red+Amber", "EWS cost-optimal"]
    ax.barh(cc["short"][::-1], cc["total_cost"][::-1], color="#1F4E79")
    for i, v in enumerate(cc["total_cost"][::-1]):
        ax.text(v + 8, i, str(int(v)), va="center")
    ax.set_xlabel(f"Total expected cost (miss={COST_MISS}, false alarm={COST_FALSE_ALARM})")
    ax.set_title("Cost of different intervention strategies (lower is better)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task10_cost_comparison.png", dpi=200)
    plt.close()

    # Fig 4: driver chart
    cs = coef.copy()
    cs["label"] = cs["feature"].map(pretty)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(cs["label"], cs["std_coef"], color=["#2E8B57" if v < 0 else "#C0392B" for v in cs["std_coef"]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Standardized log-odds effect on probation risk (red = raises, green = lowers)")
    ax.set_title("What drives the risk score?")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task10_risk_drivers.png", dpi=200)
    plt.close()
    print("Saved 4 figures to outputs/figures/ (task10_*.png)")


if __name__ == "__main__":
    log_path = TAB_DIR / "task10_results.txt"
    with open(log_path, "w") as f:
        sys.stdout = Tee(sys.__stdout__, f)
        try:
            main()
        finally:
            sys.stdout = sys.__stdout__
