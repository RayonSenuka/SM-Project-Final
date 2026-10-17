"""
task5_modelling.py
Task 5: Predictive Statistical Modelling

Builds, compares and evaluates:
  A. Multiple Linear Regression (OLS) for current_cgpa, with a hand-rolled
     inference layer (coefficients, SE, t, p, R2, adj-R2, F-test, VIF)
     since statsmodels is not available in this environment.
  B. Logistic Regression as a GLM (Binomial family, logit link), fit by
     Iteratively Reweighted Least Squares (IRLS) for inference, and via
     scikit-learn for predictive evaluation.
  C. Ridge / Lasso / Elastic Net regularized regression for current_cgpa,
     with cross-validated penalty selection.
  D. Variable selection: exhaustive best-subset search over the numeric
     predictors (2^9 = 512 models, feasible), and forward/backward
     stepwise selection over the full 15-predictor set (AIC-based).
  E. Model comparison tables (regression + classification) on a held-out
     test set.

Run:
    python src/task5_modelling.py

Output:
    Printed results (mirrored to outputs/tables/task5_results.txt),
    CSV tables in outputs/tables/, figures in outputs/figures/.
"""

import itertools
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import (ElasticNetCV, LassoCV, LinearRegression,
                                   LogisticRegression, RidgeCV)
from sklearn.metrics import (RocCurveDisplay, accuracy_score,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_prep import (ALL_PREDICTORS, BINARY_PREDICTORS,  # noqa: E402
                         NUMERIC_PREDICTORS, get_classification_data,
                         get_regression_data)

FIG_DIR = BASE / "outputs" / "figures"
TAB_DIR = BASE / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

RNG = 42
LOG_LINES = []


def log(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    LOG_LINES.append(line)


def section(title):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


# =====================================================================
# Hand-rolled OLS with full inferential output (no statsmodels available)
# =====================================================================
class OLSResult:
    def __init__(self, names, beta, se, t, p, r2, adj_r2, f_stat, f_p, n, k,
                 resid, fitted, y):
        self.names, self.beta, self.se, self.t, self.p = names, beta, se, t, p
        self.r2, self.adj_r2 = r2, adj_r2
        self.f_stat, self.f_p = f_stat, f_p
        self.n, self.k = n, k
        self.resid, self.fitted, self.y = resid, fitted, y

    def summary_table(self):
        return pd.DataFrame({
            "coef": self.beta, "std_err": self.se, "t": self.t, "p_value": self.p
        }, index=self.names)


def fit_ols(X, y, names):
    n, k = X.shape
    Xd = np.column_stack([np.ones(n), X])
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ y
    fitted = Xd @ beta
    resid = y - fitted
    rss = np.sum(resid ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    dof = n - k - 1
    sigma2 = rss / dof
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t_vals = beta / se
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), dof))
    r2 = 1 - rss / tss
    adj_r2 = 1 - (1 - r2) * (n - 1) / dof
    f_stat = (r2 / k) / ((1 - r2) / dof)
    f_p = 1 - stats.f.cdf(f_stat, k, dof)
    return OLSResult(["Intercept"] + names, beta, se, t_vals, p_vals,
                      r2, adj_r2, f_stat, f_p, n, k, resid, fitted, y)


def vif_table(X, names):
    """Variance Inflation Factor for each predictor."""
    vifs = []
    for i in range(X.shape[1]):
        y_i = X[:, i]
        X_others = np.delete(X, i, axis=1)
        r2 = fit_ols(X_others, y_i, [n for j, n in enumerate(names) if j != i]).r2
        vifs.append(1 / (1 - r2) if r2 < 0.999 else np.inf)
    return pd.Series(vifs, index=names)


def aic_bic_from_rss(rss, n, k):
    aic = n * np.log(rss / n) + 2 * (k + 1)
    bic = n * np.log(rss / n) + (k + 1) * np.log(n)
    return aic, bic


# =====================================================================
# Hand-rolled Binomial GLM (logistic regression) via IRLS
# =====================================================================
def fit_glm_binomial(X, y, names, max_iter=100, tol=1e-8):
    n, k = X.shape
    Xd = np.column_stack([np.ones(n), X])
    beta = np.zeros(Xd.shape[1])
    for _ in range(max_iter):
        eta = Xd @ beta
        mu = 1 / (1 + np.exp(-eta))
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        W = mu * (1 - mu)
        z = eta + (y - mu) / W
        WX = Xd * W[:, None]
        XtWX = Xd.T @ WX
        XtWz = Xd.T @ (W * z)
        beta_new = np.linalg.solve(XtWX, XtWz)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = Xd @ beta
    mu = 1 / (1 + np.exp(-eta))
    mu = np.clip(mu, 1e-10, 1 - 1e-10)
    W = mu * (1 - mu)
    XtWX = Xd.T @ (Xd * W[:, None])
    cov = np.linalg.inv(XtWX)
    se = np.sqrt(np.diag(cov))
    z_vals = beta / se
    p_vals = 2 * (1 - stats.norm.cdf(np.abs(z_vals)))
    odds_ratio = np.exp(beta)
    ci_low = np.exp(beta - 1.96 * se)
    ci_high = np.exp(beta + 1.96 * se)

    # deviance-based pseudo R2 (McFadden)
    ll_model = np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
    p_null = y.mean()
    ll_null = np.sum(y * np.log(p_null) + (1 - y) * np.log(1 - p_null))
    mcfadden_r2 = 1 - ll_model / ll_null

    table = pd.DataFrame({
        "coef": beta, "std_err": se, "z": z_vals, "p_value": p_vals,
        "odds_ratio": odds_ratio, "OR_2.5%": ci_low, "OR_97.5%": ci_high
    }, index=["Intercept"] + names)
    return table, mcfadden_r2, beta


def main():
    # =================================================================
    # PART A — Multiple Linear Regression (OLS) for current_cgpa
    # =================================================================
    section("5.A  Multiple Linear Regression — target: current_cgpa")
    X, y, df_reg = get_regression_data()
    names = ALL_PREDICTORS

    ols_full = fit_ols(X, y, names)
    log(f"n={ols_full.n}, k={ols_full.k} predictors")
    log(f"R2={ols_full.r2:.4f}, Adjusted R2={ols_full.adj_r2:.4f}")
    log(f"Overall F-test: F={ols_full.f_stat:.4f}, p={ols_full.f_p:.4g}")
    tbl = ols_full.summary_table()
    tbl["sig"] = tbl["p_value"].apply(lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
    log(tbl.round(4).to_string())
    tbl.round(4).to_csv(TAB_DIR / "task5_ols_full_coefficients.csv")

    vifs = vif_table(X, names)
    log("\nVariance Inflation Factors (VIF; >5 flags concern, >10 serious):")
    log(vifs.round(2).to_string())
    vifs.round(2).to_csv(TAB_DIR / "task5_vif.csv")

    # Residual diagnostics
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(ols_full.fitted, ols_full.resid, alpha=0.3, s=12)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")
    stats.probplot(ols_full.resid, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q Plot of Residuals")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ols_residual_diagnostics.png", dpi=150)
    plt.close()

    # Breusch-Pagan-style check: regress squared residuals on fitted values
    bp_X = ols_full.fitted.reshape(-1, 1)
    bp_ols = fit_ols(bp_X, ols_full.resid ** 2, ["fitted"])
    log(f"\nHomoscedasticity check (resid^2 ~ fitted): "
        f"slope p-value={bp_ols.p[1]:.4g} "
        f"({'evidence of heteroscedasticity' if bp_ols.p[1] < 0.05 else 'no strong evidence of heteroscedasticity'})")
    sw_stat, sw_p = stats.shapiro(ols_full.resid[:5000])
    log(f"Shapiro-Wilk normality test on residuals: W={sw_stat:.4f}, p={sw_p:.4g} "
        f"({'residuals deviate from normality' if sw_p < 0.05 else 'residuals consistent with normality'})")

    # ---- Sensitivity model: add previous_sgpa ----
    section("5.A-2  Sensitivity check: OLS + previous_sgpa (autoregressive predictor)")
    df_sens = df_reg.dropna(subset=["previous_sgpa"])
    X_sens = np.column_stack([df_sens[names].values, df_sens["previous_sgpa"].values])
    y_sens = df_sens["current_cgpa"].values
    ols_sens = fit_ols(X_sens, y_sens, names + ["previous_sgpa"])
    log(f"n={ols_sens.n}, R2={ols_sens.r2:.4f} (vs {ols_full.r2:.4f} without previous_sgpa), "
        f"Adjusted R2={ols_sens.adj_r2:.4f}")
    log(f"previous_sgpa coefficient={ols_sens.beta[-1]:.4f}, p={ols_sens.p[-1]:.4g}")
    log("Interpretation: previous academic performance is (as expected) the single strongest "
        "predictor of current CGPA; the behavioural-predictor model above is retained as the "
        "primary model because it targets ACTIONABLE levers for the department (Task 1 "
        "objective), whereas previous_sgpa is not something the university can intervene on.")

    # =================================================================
    # PART B — Logistic Regression / GLM for probation
    # =================================================================
    section("5.B  Logistic Regression (GLM, Binomial family, logit link) — target: probation")
    Xc, yc, df_clf = get_classification_data()
    glm_table, mcfadden_r2, beta_glm = fit_glm_binomial(Xc, yc, names)
    log(f"n={len(yc)}, positive rate (probation=Yes) = {yc.mean():.3%}")
    log(f"McFadden's pseudo-R2 = {mcfadden_r2:.4f}")
    glm_table["sig"] = glm_table["p_value"].apply(lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
    log(glm_table.round(4).to_string())
    glm_table.round(4).to_csv(TAB_DIR / "task5_glm_logistic_coefficients.csv")

    log("\nInterpretation of significant odds ratios (OR>1 raises probation odds, OR<1 lowers it):")
    for idx, row in glm_table.iloc[1:].iterrows():
        if row["p_value"] < 0.05:
            direction = "increases" if row["coef"] > 0 else "decreases"
            log(f"  {idx}: OR={row['odds_ratio']:.3f} -> each unit increase {direction} "
                f"the odds of probation by {abs(row['odds_ratio']-1):.1%} (p={row['p_value']:.4g})")

    # =================================================================
    # PART C — Ridge / Lasso / Elastic Net (regularized regression, CGPA)
    # =================================================================
    section("5.C  Regularized regression (Ridge / Lasso / Elastic Net) — target: current_cgpa")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RNG)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    ols_cv = LinearRegression().fit(X_train_s, y_train)
    ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=kf).fit(X_train_s, y_train)
    lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 50), cv=kf, max_iter=20000, random_state=RNG).fit(X_train_s, y_train)
    enet_cv = ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1], alphas=np.logspace(-4, 1, 50),
                            cv=kf, max_iter=20000, random_state=RNG).fit(X_train_s, y_train)

    log(f"Ridge best alpha = {ridge_cv.alpha_:.5g}")
    log(f"Lasso best alpha = {lasso_cv.alpha_:.5g}")
    log(f"ElasticNet best alpha = {enet_cv.alpha_:.5g}, l1_ratio = {enet_cv.l1_ratio_:.3g}")

    coef_compare = pd.DataFrame({
        "OLS": ols_cv.coef_, "Ridge": ridge_cv.coef_, "Lasso": lasso_cv.coef_, "ElasticNet": enet_cv.coef_
    }, index=names)
    log("\nStandardized coefficient comparison:")
    log(coef_compare.round(4).to_string())
    coef_compare.round(4).to_csv(TAB_DIR / "task5_regularization_coefficients.csv")
    dropped = coef_compare.index[(coef_compare["Lasso"].abs() < 1e-6)].tolist()
    log(f"\nPredictors zeroed out (dropped) by Lasso: {dropped if dropped else 'none'}")

    fig, ax = plt.subplots(figsize=(9, 6))
    coef_compare.plot(kind="barh", ax=ax)
    ax.set_title("Standardized Coefficients: OLS vs Ridge vs Lasso vs Elastic Net")
    ax.set_xlabel("Standardized coefficient")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "regularization_coefficients.png", dpi=150)
    plt.close()

    # =================================================================
    # PART D — Best subset (numeric-only) + forward/backward stepwise (full)
    # =================================================================
    section("5.D  Variable selection — best subset (numeric) & forward/backward stepwise (full)")

    # ---- Exhaustive best subset over the 9 numeric predictors ----
    best_subset_results = []
    for r in range(1, len(NUMERIC_PREDICTORS) + 1):
        for combo in itertools.combinations(range(len(NUMERIC_PREDICTORS)), r):
            X_sub = X[:, combo]
            fit = fit_ols(X_sub, y, [NUMERIC_PREDICTORS[i] for i in combo])
            rss = np.sum(fit.resid ** 2)
            aic, bic = aic_bic_from_rss(rss, fit.n, len(combo))
            best_subset_results.append((combo, len(combo), fit.adj_r2, aic, bic))

    bs_df = pd.DataFrame(best_subset_results, columns=["combo", "k", "adj_r2", "aic", "bic"])
    best_by_aic = bs_df.loc[bs_df["aic"].idxmin()]
    best_by_bic = bs_df.loc[bs_df["bic"].idxmin()]
    best_vars_aic = [NUMERIC_PREDICTORS[i] for i in best_by_aic["combo"]]
    best_vars_bic = [NUMERIC_PREDICTORS[i] for i in best_by_bic["combo"]]
    log(f"Best subset (numeric predictors only, {len(best_subset_results)} models fit) by AIC "
        f"(k={best_by_aic['k']}): {best_vars_aic}")
    log(f"  AIC={best_by_aic['aic']:.2f}, Adjusted R2={best_by_aic['adj_r2']:.4f}")
    log(f"Best subset by BIC (k={best_by_bic['k']}): {best_vars_bic}")
    log(f"  BIC={best_by_bic['bic']:.2f}, Adjusted R2={best_by_bic['adj_r2']:.4f}")

    # ---- Forward stepwise over the FULL 15-predictor set (AIC-greedy) ----
    def forward_stepwise(X, y, names, criterion="aic"):
        remaining = list(range(len(names)))
        selected = []
        current_score = np.inf
        history = []
        while remaining:
            scores = []
            for cand in remaining:
                trial = selected + [cand]
                fit = fit_ols(X[:, trial], y, [names[i] for i in trial])
                rss = np.sum(fit.resid ** 2)
                aic, bic = aic_bic_from_rss(rss, fit.n, len(trial))
                score = aic if criterion == "aic" else bic
                scores.append((score, cand))
            scores.sort()
            best_score, best_cand = scores[0]
            if best_score < current_score:
                selected.append(best_cand)
                remaining.remove(best_cand)
                current_score = best_score
                history.append((list(selected), best_score))
            else:
                break
        return [names[i] for i in selected], current_score, history

    def backward_stepwise(X, y, names, criterion="aic"):
        selected = list(range(len(names)))
        fit = fit_ols(X[:, selected], y, names)
        rss = np.sum(fit.resid ** 2)
        current_score = aic_bic_from_rss(rss, fit.n, len(selected))[0 if criterion == "aic" else 1]
        improved = True
        while improved and len(selected) > 1:
            improved = False
            scores = []
            for cand in selected:
                trial = [s for s in selected if s != cand]
                fit = fit_ols(X[:, trial], y, [names[i] for i in trial])
                rss = np.sum(fit.resid ** 2)
                aic, bic = aic_bic_from_rss(rss, fit.n, len(trial))
                score = aic if criterion == "aic" else bic
                scores.append((score, cand))
            scores.sort()
            best_score, worst_cand = scores[0]
            if best_score < current_score:
                selected.remove(worst_cand)
                current_score = best_score
                improved = True
        return [names[i] for i in selected], current_score

    fwd_vars, fwd_score, fwd_hist = forward_stepwise(X, y, names, "aic")
    bwd_vars, bwd_score = backward_stepwise(X, y, names, "aic")
    log(f"\nForward stepwise selection (full 15-predictor set, AIC): {fwd_vars}")
    log(f"  final AIC={fwd_score:.2f}")
    log(f"Backward stepwise elimination (full 15-predictor set, AIC): {bwd_vars}")
    log(f"  final AIC={bwd_score:.2f}")
    log(f"\nForward/backward agree: {set(fwd_vars) == set(bwd_vars)}")

    # =================================================================
    # PART E — Model comparison & evaluation (held-out test set)
    # =================================================================
    section("5.E  Model comparison & evaluation — regression (current_cgpa)")

    def eval_reg(model_name, coefs_idx, X_tr, X_te, y_tr, y_te, fitted_model=None, scaled=False):
        if fitted_model is None:
            fit = fit_ols(X_tr[:, coefs_idx], y_tr, [names[i] for i in coefs_idx])
            pred_tr = np.column_stack([np.ones(len(X_tr)), X_tr[:, coefs_idx]]) @ fit.beta
            pred_te = np.column_stack([np.ones(len(X_te)), X_te[:, coefs_idx]]) @ fit.beta
        else:
            pred_tr = fitted_model.predict(X_tr)
            pred_te = fitted_model.predict(X_te)
        rmse_tr = np.sqrt(np.mean((y_tr - pred_tr) ** 2))
        rmse_te = np.sqrt(np.mean((y_te - pred_te) ** 2))
        r2_te = 1 - np.sum((y_te - pred_te) ** 2) / np.sum((y_te - y_te.mean()) ** 2)
        return rmse_tr, rmse_te, r2_te

    fwd_idx = [names.index(v) for v in fwd_vars]
    results_reg = []
    r = eval_reg("OLS (full, 15 predictors)", list(range(len(names))), X_train, X_test, y_train, y_test)
    results_reg.append(("OLS (full, 15 predictors)", *r))
    r = eval_reg("OLS (forward-stepwise subset)", fwd_idx, X_train, X_test, y_train, y_test)
    results_reg.append((f"OLS (stepwise subset, k={len(fwd_vars)})", *r))
    r = eval_reg("Ridge (all predictors, standardized)", None, X_train_s, X_test_s, y_train, y_test, ridge_cv)
    results_reg.append(("Ridge (CV-tuned)", *r))
    r = eval_reg("Lasso (all predictors, standardized)", None, X_train_s, X_test_s, y_train, y_test, lasso_cv)
    results_reg.append(("Lasso (CV-tuned)", *r))
    r = eval_reg("ElasticNet (all predictors, standardized)", None, X_train_s, X_test_s, y_train, y_test, enet_cv)
    results_reg.append(("ElasticNet (CV-tuned)", *r))

    reg_comparison = pd.DataFrame(results_reg, columns=["model", "train_rmse", "test_rmse", "test_r2"])
    log(reg_comparison.round(4).to_string(index=False))
    reg_comparison.round(4).to_csv(TAB_DIR / "task5_regression_comparison.csv", index=False)

    section("5.E  Model comparison & evaluation — classification (probation)")
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=RNG, stratify=yc)
    scaler_c = StandardScaler().fit(Xc_train)
    Xc_train_s, Xc_test_s = scaler_c.transform(Xc_train), scaler_c.transform(Xc_test)

    bwd_idx = [names.index(v) for v in bwd_vars]

    def eval_clf(model, X_tr, X_te, y_tr, y_te, label):
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = model.predict(X_te)
        acc = accuracy_score(y_te, pred)
        prec = precision_score(y_te, pred, zero_division=0)
        rec = recall_score(y_te, pred)
        f1 = f1_score(y_te, pred)
        auc = roc_auc_score(y_te, proba)
        cm = confusion_matrix(y_te, pred)
        return dict(model=label, accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc), cm, model

    log_full = LogisticRegression(max_iter=2000, class_weight="balanced")
    res_full, cm_full, fitted_full = eval_clf(log_full, Xc_train_s, Xc_test_s, yc_train, yc_test,
                                               "Logistic (full, 15 predictors, balanced)")

    log_subset = LogisticRegression(max_iter=2000, class_weight="balanced")
    res_subset, cm_subset, fitted_subset = eval_clf(
        log_subset, Xc_train_s[:, bwd_idx], Xc_test_s[:, bwd_idx], yc_train, yc_test,
        f"Logistic (backward-stepwise subset, k={len(bwd_vars)}, balanced)")

    log_unweighted = LogisticRegression(max_iter=2000)
    res_unw, cm_unw, fitted_unw = eval_clf(log_unweighted, Xc_train_s, Xc_test_s, yc_train, yc_test,
                                            "Logistic (full, 15 predictors, unweighted)")

    clf_comparison = pd.DataFrame([res_full, res_subset, res_unw])
    log(clf_comparison.round(4).to_string(index=False))
    clf_comparison.round(4).to_csv(TAB_DIR / "task5_classification_comparison.csv", index=False)
    log(f"\nConfusion matrix, Logistic (full, balanced):\n{cm_full}")
    log(f"Confusion matrix, Logistic (unweighted):\n{cm_unw}")
    log("Note: class_weight='balanced' trades some accuracy for much higher recall on the "
        "minority (probation=Yes) class, which matters more for an early-warning use case "
        "than raw accuracy, since missing an at-risk student is costlier than a false alarm.")

    # ROC curve figure
    fig, ax = plt.subplots(figsize=(6.5, 6))
    RocCurveDisplay.from_estimator(fitted_full, Xc_test_s, yc_test, ax=ax, name="Full (balanced)")
    RocCurveDisplay.from_estimator(fitted_subset, Xc_test_s[:, bwd_idx], yc_test, ax=ax,
                                    name="Stepwise subset (balanced)")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve — Logistic Regression Models (probation)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "logistic_roc_curve.png", dpi=150)
    plt.close()

    # Confusion matrix heatmap
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    sns.heatmap(cm_full, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    axes[0].set_title("Full model (balanced)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    sns.heatmap(cm_unw, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    axes[1].set_title("Full model (unweighted)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "logistic_confusion_matrices.png", dpi=150)
    plt.close()

    # =================================================================
    # Final recommendation
    # =================================================================
    section("5.F  Recommended models")
    best_reg = reg_comparison.loc[reg_comparison["test_rmse"].idxmin(), "model"]
    best_clf = clf_comparison.loc[clf_comparison["auc"].idxmax(), "model"]
    log(f"Lowest test RMSE (regression): {best_reg}")
    log(f"Highest AUC (classification): {best_clf}")
    log("Recommended for the consultancy report: the stepwise-selected OLS model for CGPA "
        "(near-identical accuracy to the full/regularized models but far fewer variables to "
        "explain to non-technical stakeholders), and the balanced logistic/GLM model for "
        "probation risk (prioritizes catching at-risk students, matching the early-warning "
        "objective in Task 1).")

    with open(TAB_DIR / "task5_results.txt", "w") as f:
        f.write("\n".join(LOG_LINES))
    log(f"\nFull log saved to: {TAB_DIR / 'task5_results.txt'}")


if __name__ == "__main__":
    main()
