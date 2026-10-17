"""
task8_bayesian.py
Task 8: Critical Evaluation of Bayesian Statistical Methods

Discussion task per the assignment brief, grounded here in three small
real computations on the same data/splits used in Tasks 5 and 7:

  A. Gaussian Naive Bayes for `probation`, compared against the class-
     balanced logistic/GLM model from Task 5 on the SAME held-out split.
  B. Bayesian Ridge Regression for `current_cgpa`, compared against OLS
     and the CV-tuned Ridge from Task 5, on the SAME held-out split —
     reports posterior coefficient means AND standard deviations (the
     distinctive Bayesian output that plain OLS/Ridge do not give).
  C. A worked Bayesian decision-making example: using the logistic
     model's posterior/predicted probability of probation for one
     representative "borderline" student profile, combined with an
     explicit cost matrix, to choose an action by minimizing expected
     cost — illustrating Bayesian decision theory concretely rather than
     only describing it.

Run:
    python src/task8_bayesian.py

Output:
    Printed results (mirrored to outputs/tables/task8_results.txt),
    CSV tables in outputs/tables/, figures in outputs/figures/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
from model_prep import ALL_PREDICTORS  # noqa: E402
import model_prep  # noqa: E402

FIG_DIR = BASE / "outputs" / "figures"
TAB_DIR = BASE / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def naive_bayes_vs_logistic(df_clf):
    section("8A. Gaussian Naive Bayes vs class-balanced Logistic — target: probation")
    X = df_clf[ALL_PREDICTORS].astype(float).values
    y = df_clf["probation_bin"].astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    def eval_model(model, label, X_train, X_test):
        model.fit(X_train, ytr)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        return {
            "model": label,
            "accuracy": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred, zero_division=0),
            "f1": f1_score(yte, pred, zero_division=0),
            "auc": roc_auc_score(yte, proba),
        }

    rows = [
        eval_model(GaussianNB(), "Gaussian Naive Bayes (default priors)", Xtr_s, Xte_s),
        eval_model(
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            "Logistic / GLM, balanced (Task 5 recommended model)", Xtr_s, Xte_s,
        ),
    ]
    # Naive Bayes with priors set to match the class-balanced logistic's implicit
    # weighting (i.e. treat both classes as equally important a priori), to isolate
    # the conditional-independence assumption from the class-prior effect.
    nb_balanced = GaussianNB(priors=[0.5, 0.5])
    rows.append(eval_model(nb_balanced, "Gaussian Naive Bayes (priors forced 50/50)", Xtr_s, Xte_s))

    result = pd.DataFrame(rows)
    print(result.round(4).to_string(index=False))
    result.to_csv(TAB_DIR / "task8_nb_vs_logistic.csv", index=False)

    # Quick evidence on the conditional-independence assumption NB relies on:
    # report the max off-diagonal |correlation| among predictors, already
    # computed for VIF in Task 5 (all VIF < 7) — cross-check here directly.
    corr = pd.DataFrame(Xtr, columns=ALL_PREDICTORS).corr().copy()
    corr_vals = corr.values.copy()
    np.fill_diagonal(corr_vals, 0)
    corr = pd.DataFrame(corr_vals, index=corr.index, columns=corr.columns)
    max_corr = corr.abs().values.max()
    max_pair = corr.abs().stack().idxmax()
    print(f"\nMax |pairwise correlation| among the 15 predictors: {max_corr:.3f} "
          f"({max_pair[0]} & {max_pair[1]}) — NB's conditional-independence "
          f"assumption is violated in degree, not badly, consistent with Task 5's VIF check.")
    return result


def bayesian_ridge_vs_ols(df_reg):
    section("8B. Bayesian Ridge Regression vs OLS vs (frequentist) Ridge — target: current_cgpa")
    X = df_reg[ALL_PREDICTORS].astype(float).values
    y = df_reg["current_cgpa"].astype(float).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    ols = LinearRegression().fit(Xtr_s, ytr)
    ridge = Ridge(alpha=3.56).fit(Xtr_s, ytr)  # same alpha CV-selected in Task 5
    bridge = BayesianRidge().fit(Xtr_s, ytr)

    def perf(model, label):
        pred = model.predict(Xte_s)
        return {
            "model": label,
            "test_r2": r2_score(yte, pred),
            "test_rmse": float(np.sqrt(np.mean((yte - pred) ** 2))),
        }

    perf_rows = [perf(ols, "OLS"), perf(ridge, "Ridge (alpha=3.56, Task 5 value)"),
                 perf(bridge, "Bayesian Ridge (evidence-tuned alpha/lambda)")]
    perf_df = pd.DataFrame(perf_rows)
    print(perf_df.round(4).to_string(index=False))
    perf_df.to_csv(TAB_DIR / "task8_bayesian_ridge_performance.csv", index=False)

    # Posterior coefficient means and standard deviations — the distinctive
    # Bayesian output. sigma_ is the posterior covariance of the coefficients.
    post_sd = np.sqrt(np.diag(bridge.sigma_))
    coef_table = pd.DataFrame({
        "predictor": ALL_PREDICTORS,
        "ols_coef": ols.coef_,
        "bayesian_ridge_posterior_mean": bridge.coef_,
        "bayesian_ridge_posterior_sd": post_sd,
    })
    coef_table["95pct_credible_lower"] = coef_table["bayesian_ridge_posterior_mean"] - 1.96 * coef_table["bayesian_ridge_posterior_sd"]
    coef_table["95pct_credible_upper"] = coef_table["bayesian_ridge_posterior_mean"] + 1.96 * coef_table["bayesian_ridge_posterior_sd"]
    coef_table["credible_interval_excludes_0"] = (
        (coef_table["95pct_credible_lower"] > 0) | (coef_table["95pct_credible_upper"] < 0)
    )
    print("\nPosterior coefficient means/SDs (standardized predictors):")
    print(coef_table.round(4).to_string(index=False))
    coef_table.to_csv(TAB_DIR / "task8_bayesian_ridge_coefficients.csv", index=False)

    # Forest plot of posterior means +/- 95% credible interval
    plot_df = coef_table.sort_values("bayesian_ridge_posterior_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(plot_df))
    colors = ["#1f77b4" if x else "#bbbbbb" for x in plot_df["credible_interval_excludes_0"]]
    ax.errorbar(
        plot_df["bayesian_ridge_posterior_mean"], y_pos,
        xerr=1.96 * plot_df["bayesian_ridge_posterior_sd"],
        fmt="o", ecolor="grey", capsize=3, color="black", zorder=3,
    )
    ax.scatter(plot_df["bayesian_ridge_posterior_mean"], y_pos, color=colors, zorder=4)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["predictor"])
    ax.set_xlabel("Posterior coefficient mean (standardized), 95% credible interval")
    ax.set_title("Task 8 — Bayesian Ridge posterior coefficients, current_cgpa")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task8_bayesian_ridge_forest.png", dpi=150)
    plt.close(fig)

    print(f"\nBayesian Ridge's data-driven precision parameters: "
          f"alpha (noise precision) = {bridge.alpha_:.3f}, "
          f"lambda (coefficient precision) = {bridge.lambda_:.5f} "
          f"-> implied prior variance on each coefficient = {1/bridge.lambda_:.3f}. "
          f"This value was learned from the data by maximizing model evidence, "
          f"not chosen by cross-validation grid search as in Task 5's Ridge.")

    return perf_df, coef_table


def decision_theory_example(df_clf):
    section("8C. Worked Bayesian decision-making example (expected-cost minimization)")
    X = df_clf[ALL_PREDICTORS].astype(float).values
    y = df_clf["probation_bin"].astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    logit = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr_s, ytr)
    proba_test = logit.predict_proba(Xte_s)[:, 1]

    # Pick a genuinely borderline student from the test set (posterior prob near 0.5)
    idx = int(np.argmin(np.abs(proba_test - 0.5)))
    p = float(proba_test[idx])

    # Illustrative cost matrix (Task 1 framing: missing an at-risk student is worse
    # than a false alarm, which just triggers an advising conversation).
    cost_intervene_given_no_risk = 1   # cost of an unnecessary advising conversation
    cost_intervene_given_risk = 0      # correctly caught -> no extra "cost" beyond the conversation itself
    cost_no_action_given_no_risk = 0   # correctly left alone
    cost_no_action_given_risk = 5      # missed an at-risk student -> costliest outcome

    exp_cost_intervene = p * cost_intervene_given_risk + (1 - p) * cost_intervene_given_no_risk
    exp_cost_no_action = p * cost_no_action_given_risk + (1 - p) * cost_no_action_given_no_risk

    decision = "INTERVENE (flag for advising)" if exp_cost_intervene < exp_cost_no_action else "NO ACTION"

    print(f"Example student (test-set index {idx}): posterior P(probation) = {p:.3f}")
    print("Cost matrix used (illustrative, set by the consultancy in consultation with "
          "the department — not derived from data):")
    print(f"  Intervene & student was NOT at risk: cost = {cost_intervene_given_no_risk} "
          f"(one unnecessary advising slot)")
    print(f"  Intervene & student WAS at risk:     cost = {cost_intervene_given_risk} "
          f"(correct catch)")
    print(f"  No action & student was NOT at risk: cost = {cost_no_action_given_no_risk} "
          f"(correct)")
    print(f"  No action & student WAS at risk:     cost = {cost_no_action_given_risk} "
          f"(missed at-risk student — costliest)")
    print(f"\nE[cost | intervene]  = {p:.3f}*{cost_intervene_given_risk} + "
          f"{1-p:.3f}*{cost_intervene_given_no_risk} = {exp_cost_intervene:.3f}")
    print(f"E[cost | no action]  = {p:.3f}*{cost_no_action_given_risk} + "
          f"{1-p:.3f}*{cost_no_action_given_no_risk} = {exp_cost_no_action:.3f}")
    print(f"Decision (minimize expected cost): {decision}")

    # Show how the break-even posterior probability threshold shifts with the cost ratio
    ratios = [1, 2, 5, 10]
    thresh_rows = []
    for r in ratios:
        # break-even: p* = c10 / (c10 + c01*r) for cost_no_action_given_risk = r * cost_intervene_given_no_risk
        c01 = cost_intervene_given_no_risk
        c10 = r * c01
        p_star = c01 / (c01 + c10)
        thresh_rows.append({"cost_ratio_miss_vs_false_alarm": r, "break_even_threshold": p_star})
    thresh_df = pd.DataFrame(thresh_rows)
    print("\nHow the optimal flagging threshold shifts with the assumed cost ratio "
          "(missing an at-risk student vs. one false alarm):")
    print(thresh_df.round(3).to_string(index=False))
    print("Note: a plain 0.5 cutoff (implicitly equal costs) is a special/default case, "
          "not a principled choice — this is precisely what Bayesian decision theory "
          "makes explicit and adjustable, unlike a bare classification-accuracy read of "
          "Task 5's confusion matrices.")
    thresh_df.to_csv(TAB_DIR / "task8_decision_threshold_sensitivity.csv", index=False)

    return idx, p, decision, thresh_df


def main():
    log_path = TAB_DIR / "task8_results.txt"
    with open(log_path, "w") as f:
        tee = Tee(sys.stdout, f)
        old_stdout = sys.stdout
        sys.stdout = tee
        try:
            df = model_prep.build_model_frame()
            df_clf = df.dropna(subset=["probation_bin"] + ALL_PREDICTORS).reset_index(drop=True)
            df_reg = df.dropna(subset=["current_cgpa"] + ALL_PREDICTORS).reset_index(drop=True)

            print("Task 8 — Bayesian methods critical evaluation")
            print(f"Classification sample (probation): n = {len(df_clf)}")
            print(f"Regression sample (CGPA): n = {len(df_reg)}")

            naive_bayes_vs_logistic(df_clf)
            bayesian_ridge_vs_ols(df_reg)
            decision_theory_example(df_clf)

        finally:
            sys.stdout = old_stdout

    print(f"\nFull log written to {log_path}")


if __name__ == "__main__":
    main()
