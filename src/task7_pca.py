"""
task7_pca.py
Task 7: Critical Evaluation of Principal Component Analysis (PCA)

This is a *discussion* task (per the assignment brief), but the discussion
is grounded here in an actual PCA run on the same 15-predictor set used in
Task 5, rather than argued from first principles alone. Specifically:

  A. Standardize the 15 predictors (9 numeric + 6 binary dummies) used
     throughout Tasks 4-5.
  B. Run PCA, report the explained-variance / scree profile and loadings.
  C. Re-fit the CGPA regression and probation classifier on the top-k
     principal components (Principal Component Regression / PCR, and
     PC-logistic) and compare predictive performance directly against the
     original-variable models from Task 5, on the SAME held-out test split.
  D. Report loss of interpretability alongside any predictive gain/loss.

Run:
    python src/task7_pca.py

Output:
    Printed results (mirrored to outputs/tables/task7_results.txt),
    CSV tables in outputs/tables/, figures in outputs/figures/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
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
    """Mirror stdout to a log file, matching the style of task4/task5 scripts."""

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


def run_pca_profile(X_std, feature_names):
    section("7A. PCA on the 15-predictor set (standardized)")
    pca = PCA(n_components=len(feature_names), random_state=RANDOM_STATE)
    pca.fit(X_std)

    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    profile = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(len(ev))],
            "eigenvalue": pca.explained_variance_,
            "explained_variance_ratio": ev,
            "cumulative_variance": cum,
        }
    )
    print(profile.round(4).to_string(index=False))
    profile.to_csv(TAB_DIR / "task7_pca_variance.csv", index=False)

    n_kaiser = int((pca.explained_variance_ > 1.0).sum())
    n_80 = int(np.searchsorted(cum, 0.80) + 1)
    n_90 = int(np.searchsorted(cum, 0.90) + 1)
    print(f"\nComponents with eigenvalue > 1 (Kaiser criterion): {n_kaiser} of {len(feature_names)}")
    print(f"Components needed for >=80% cumulative variance: {n_80} of {len(feature_names)}")
    print(f"Components needed for >=90% cumulative variance: {n_90} of {len(feature_names)}")

    # Scree plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(ev) + 1), ev, "o-", label="Explained variance ratio")
    ax.plot(range(1, len(ev) + 1), cum, "s--", label="Cumulative variance", color="darkorange")
    ax.axhline(0.80, color="grey", linestyle=":", linewidth=1)
    ax.axhline(0.90, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Proportion of variance explained")
    ax.set_title("Task 7 — PCA scree / cumulative-variance plot (15 predictors)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task7_pca_scree.png", dpi=150)
    plt.close(fig)

    # Loadings for the first 5 PCs
    loadings = pd.DataFrame(
        pca.components_[:5].T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(5)],
    )
    print("\nLoadings, first 5 components (variables with |loading| >= 0.30 shown per PC):")
    for pc in loadings.columns:
        top = loadings[pc][loadings[pc].abs() >= 0.30].sort_values(key=np.abs, ascending=False)
        print(f"\n{pc} ({ev[int(pc[2:]) - 1]*100:.1f}% of variance):")
        if top.empty:
            print("  (no single predictor loads >= 0.30 — variance is spread thinly)")
        else:
            for name, val in top.items():
                print(f"  {name:28s} {val:+.3f}")
    loadings.to_csv(TAB_DIR / "task7_pca_loadings.csv")

    return pca, profile, loadings, n_80, n_90


def compare_regression(X_std, y_reg, mask_reg, pca, n_80, feature_names):
    section("7B. CGPA regression: original 15 predictors vs top-PC regression (same test split)")
    Xr = X_std[mask_reg]
    yr = y_reg[mask_reg]
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=RANDOM_STATE)

    # Original-variable OLS (matches Task 5's full model, same split logic)
    ols = LinearRegression().fit(Xtr, ytr)
    r2_orig = r2_score(yte, ols.predict(Xte))
    rmse_orig = np.sqrt(np.mean((yte - ols.predict(Xte)) ** 2))

    rows = [{"model": "OLS on original 15 predictors", "k_components_or_vars": 15,
             "test_r2": r2_orig, "test_rmse": rmse_orig}]

    for k in [n_80, 10, 15]:
        if k > 15 or k in [r["k_components_or_vars"] for r in rows if "PC" in r.get("model", "")]:
            continue
        pca_k = PCA(n_components=k, random_state=RANDOM_STATE).fit(Xtr)
        Ztr, Zte = pca_k.transform(Xtr), pca_k.transform(Xte)
        pcr = LinearRegression().fit(Ztr, ytr)
        r2_k = r2_score(yte, pcr.predict(Zte))
        rmse_k = np.sqrt(np.mean((yte - pcr.predict(Zte)) ** 2))
        rows.append({"model": f"PCR — top {k} PCs", "k_components_or_vars": k,
                     "test_r2": r2_k, "test_rmse": rmse_k})

    result = pd.DataFrame(rows)
    print(result.round(4).to_string(index=False))
    result.to_csv(TAB_DIR / "task7_pcr_vs_ols.csv", index=False)
    return result


def compare_classification(X_std, y_clf, mask_clf, pca, n_80, feature_names):
    section("7C. Probation classification: original 15 predictors vs top-PC logistic (same test split)")
    Xc = X_std[mask_clf]
    yc = y_clf[mask_clf]
    Xtr, Xte, ytr, yte = train_test_split(
        Xc, yc, test_size=0.2, random_state=RANDOM_STATE, stratify=yc
    )

    def eval_model(model, Xtr_, Xte_, label, k):
        model.fit(Xtr_, ytr)
        pred = model.predict(Xte_)
        proba = model.predict_proba(Xte_)[:, 1]
        return {
            "model": label,
            "k_components_or_vars": k,
            "accuracy": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred, zero_division=0),
            "f1": f1_score(yte, pred, zero_division=0),
            "auc": roc_auc_score(yte, proba),
        }

    rows = [eval_model(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        Xtr, Xte, "Logistic on original 15 predictors (balanced)", 15,
    )]

    for k in sorted(set([n_80, 10])):
        if k >= 15:
            continue
        pca_k = PCA(n_components=k, random_state=RANDOM_STATE).fit(Xtr)
        Ztr, Zte = pca_k.transform(Xtr), pca_k.transform(Xte)
        rows.append(eval_model(
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            Ztr, Zte, f"PC-logistic — top {k} PCs (balanced)", k,
        ))

    result = pd.DataFrame(rows)
    print(result.round(4).to_string(index=False))
    result.to_csv(TAB_DIR / "task7_pc_logistic_vs_logistic.csv", index=False)
    return result


def main():
    log_path = TAB_DIR / "task7_results.txt"
    with open(log_path, "w") as f:
        tee = Tee(sys.stdout, f)
        old_stdout = sys.stdout
        sys.stdout = tee
        try:
            df = model_prep.build_model_frame()

            # Regression (CGPA) mask/target
            df_reg = df.dropna(subset=["current_cgpa"] + ALL_PREDICTORS).reset_index(drop=True)
            X_reg_raw = df_reg[ALL_PREDICTORS].astype(float).values
            y_reg = df_reg["current_cgpa"].astype(float).values

            # Classification (probation) mask/target
            df_clf = df.dropna(subset=["probation_bin"] + ALL_PREDICTORS).reset_index(drop=True)
            X_clf_raw = df_clf[ALL_PREDICTORS].astype(float).values
            y_clf = df_clf["probation_bin"].astype(int).values

            # Standardize each design matrix independently (as Task 5 does for regularization)
            scaler_reg = StandardScaler().fit(X_reg_raw)
            X_reg_std = scaler_reg.transform(X_reg_raw)

            scaler_clf = StandardScaler().fit(X_clf_raw)
            X_clf_std = scaler_clf.transform(X_clf_raw)

            print("Task 7 — PCA critical evaluation")
            print(f"Predictor set (15, same as Task 5): {ALL_PREDICTORS}")
            print(f"Regression sample (CGPA): n = {X_reg_std.shape[0]}")
            print(f"Classification sample (probation): n = {X_clf_std.shape[0]}")

            # Run the PCA profile on the regression sample (larger, near-identical structure)
            pca, profile, loadings, n_80, n_90 = run_pca_profile(X_reg_std, ALL_PREDICTORS)

            reg_result = compare_regression(X_reg_std, y_reg, np.arange(len(y_reg)), pca, n_80, ALL_PREDICTORS)
            clf_result = compare_classification(X_clf_std, y_clf, np.arange(len(y_clf)), pca, n_80, ALL_PREDICTORS)

            section("7D. Summary")
            print(f"- {n_80} of 15 PCs are needed to reach 80% cumulative variance;"
                  f" {n_90} needed for 90%.")
            print("- No small number of components dominates: variance is spread across"
                  " most components (see task7_pca_variance.csv / scree plot).")
            print("- Predictive performance: see task7_pcr_vs_ols.csv and"
                  " task7_pc_logistic_vs_logistic.csv for the side-by-side test-set comparison.")

        finally:
            sys.stdout = old_stdout

    print(f"\nFull log written to {log_path}")


if __name__ == "__main__":
    main()
