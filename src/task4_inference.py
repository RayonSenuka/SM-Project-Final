"""
task4_inference.py
Task 4: Statistical Inference

Comparison of means, proportions, and variances, plus one-way ANOVA, on the
cleaned dataset. Uses scipy.stats only (no statsmodels dependency).

Run:
    python src/task4_inference.py

Output:
    Printed results (also mirrored to outputs/tables/task4_results.txt)
    and two new figures in outputs/figures/.
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

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_prep import build_model_frame  # noqa: E402

FIG_DIR = BASE / "outputs" / "figures"
TAB_DIR = BASE / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

LOG_LINES = []


def log(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    LOG_LINES.append(line)


def hedges_g(a, b):
    """Hedges' g effect size for two independent samples."""
    na, nb = len(a), len(b)
    pooled_sd = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / pooled_sd
    correction = 1 - (3 / (4 * (na + nb) - 9))
    return d * correction


def cramers_v(chi2, n, table_shape):
    r, k = table_shape
    return np.sqrt(chi2 / (n * (min(r - 1, k - 1))))


def section(title):
    log("\n" + "=" * 70)
    log(title)
    log("=" * 70)


def main():
    df = build_model_frame()

    # -----------------------------------------------------------------
    # 4.1 Comparison of MEANS — Test A: CGPA by scholarship status
    # -----------------------------------------------------------------
    section("4.1a  Independent-samples t-test: current_cgpa by scholarship")
    a = df.loc[df["scholarship"] == "Yes", "current_cgpa"].dropna()
    b = df.loc[df["scholarship"] == "No", "current_cgpa"].dropna()
    log(f"H0: mu(scholarship=Yes) = mu(scholarship=No)   [current CGPA]")
    log(f"H1: mu(scholarship=Yes) != mu(scholarship=No)")
    log(f"n(Yes)={len(a)}, mean={a.mean():.4f}, sd={a.std(ddof=1):.4f}")
    log(f"n(No) ={len(b)}, mean={b.mean():.4f}, sd={b.std(ddof=1):.4f}")

    lev_stat, lev_p = stats.levene(a, b)
    log(f"\nLevene's test for equal variances: F={lev_stat:.4f}, p={lev_p:.4g}")
    equal_var = lev_p > 0.05
    log(f" -> equal variances assumption {'HOLDS' if equal_var else 'VIOLATED'} at alpha=0.05; "
        f"using {'Student' if equal_var else 'Welch'} t-test accordingly.")

    t_stat, t_p = stats.ttest_ind(a, b, equal_var=equal_var)
    g = hedges_g(a.values, b.values)
    log(f"t-statistic={t_stat:.4f}, df~{len(a)+len(b)-2}, p-value={t_p:.4g}")
    log(f"Hedges' g (effect size) = {g:.4f}")
    log("Interpretation: " + (
        "Reject H0 — scholarship holders' mean CGPA differs significantly from non-holders'."
        if t_p < 0.05 else
        "Fail to reject H0 — no statistically significant mean CGPA difference by scholarship status."
    ))

    # -----------------------------------------------------------------
    # 4.1b Comparison of MEANS — Test B: attendance by probation status
    # -----------------------------------------------------------------
    section("4.1b  Independent-samples t-test: attendance by probation status")
    c = df.loc[df["probation"] == "Yes", "attendance"].dropna()
    d = df.loc[df["probation"] == "No", "attendance"].dropna()
    log("H0: mu(attendance | probation=Yes) = mu(attendance | probation=No)")
    log("H1: mu(attendance | probation=Yes) < mu(attendance | probation=No)  [one-tailed]")
    log(f"n(Yes)={len(c)}, mean={c.mean():.4f}, sd={c.std(ddof=1):.4f}")
    log(f"n(No) ={len(d)}, mean={d.mean():.4f}, sd={d.std(ddof=1):.4f}")

    lev_stat2, lev_p2 = stats.levene(c, d)
    log(f"\nLevene's test for equal variances: F={lev_stat2:.4f}, p={lev_p2:.4g}")
    equal_var2 = lev_p2 > 0.05
    log(f" -> equal variances assumption {'HOLDS' if equal_var2 else 'VIOLATED'} at alpha=0.05; "
        f"using {'Student' if equal_var2 else 'Welch'} t-test accordingly.")

    t_stat2, t_p2_two = stats.ttest_ind(c, d, equal_var=equal_var2)
    t_p2_one = t_p2_two / 2 if t_stat2 < 0 else 1 - t_p2_two / 2
    g2 = hedges_g(c.values, d.values)
    log(f"t-statistic={t_stat2:.4f}, two-tailed p={t_p2_two:.4g}, one-tailed p={t_p2_one:.4g}")
    log(f"Hedges' g (effect size) = {g2:.4f}")
    log("Interpretation: " + (
        "Reject H0 — students with a probation history have significantly lower mean attendance."
        if t_p2_one < 0.05 else
        "Fail to reject H0 — attendance means are not significantly lower for the probation group."
    ))

    # -----------------------------------------------------------------
    # 4.2 Comparison of PROPORTIONS — Test A: probation rate by scholarship
    # -----------------------------------------------------------------
    section("4.2a  Chi-square test of independence: probation vs scholarship")
    ct1 = pd.crosstab(df["scholarship"], df["probation"])
    log(ct1)
    chi2_1, p_1, dof_1, exp_1 = stats.chi2_contingency(ct1)
    log(f"\nH0: probation status is independent of scholarship status")
    log(f"H1: probation status is associated with scholarship status")
    log(f"chi2={chi2_1:.4f}, df={dof_1}, p-value={p_1:.4g}")
    log(f"Min expected cell count = {exp_1.min():.2f} (chi-square assumption check, should be >=5)")
    v1 = cramers_v(chi2_1, ct1.values.sum(), ct1.shape)
    log(f"Cramer's V (effect size) = {v1:.4f}")
    rate_yes = (df.loc[df.scholarship == "Yes", "probation"] == "Yes").mean()
    rate_no = (df.loc[df.scholarship == "No", "probation"] == "Yes").mean()
    log(f"Probation rate | scholarship=Yes: {rate_yes:.3%}")
    log(f"Probation rate | scholarship=No : {rate_no:.3%}")
    log("Interpretation: " + (
        "Reject H0 — probation rate significantly differs by scholarship status."
        if p_1 < 0.05 else
        "Fail to reject H0 — no significant association between scholarship and probation."
    ))

    # -----------------------------------------------------------------
    # 4.2b Comparison of PROPORTIONS — Test B: suspension rate by gender
    # -----------------------------------------------------------------
    section("4.2b  Suspension rate by gender")
    ct2 = pd.crosstab(df["gender"], df["suspension"])
    log(ct2)
    chi2_2, p_2, dof_2, exp_2 = stats.chi2_contingency(ct2)
    log(f"\nMin expected cell count = {exp_2.min():.2f}")
    use_fisher = exp_2.min() < 5
    log(f" -> {'Below 5: chi-square assumption violated, using Fisher exact test instead.' if use_fisher else 'Chi-square assumption satisfied.'}")
    if use_fisher and ct2.shape == (2, 2):
        odds_ratio, p_fisher = stats.fisher_exact(ct2)
        log(f"H0: suspension is independent of gender")
        log(f"H1: suspension is associated with gender")
        log(f"Fisher exact test: odds ratio={odds_ratio:.4f}, p-value={p_fisher:.4g}")
        final_p = p_fisher
    else:
        log(f"chi2={chi2_2:.4f}, df={dof_2}, p-value={p_2:.4g}")
        final_p = p_2
    rate_m = (df.loc[df.gender == "Male", "suspension"] == "Yes").mean()
    rate_f = (df.loc[df.gender == "Female", "suspension"] == "Yes").mean()
    log(f"Suspension rate | Male:   {rate_m:.3%}")
    log(f"Suspension rate | Female: {rate_f:.3%}")
    log("Interpretation: " + (
        "Reject H0 — suspension rate significantly differs by gender."
        if final_p < 0.05 else
        "Fail to reject H0 — no statistically significant gender difference in suspension rate "
        "(note: suspension is rare, 46/1194 cases, limiting statistical power)."
    ))

    # -----------------------------------------------------------------
    # 4.3 One-way ANOVA: CGPA across English proficiency levels
    # -----------------------------------------------------------------
    section("4.3  One-way ANOVA: current_cgpa by english_proficiency")
    groups = {}
    for level in ["Basic", "Intermediate", "Advance"]:
        groups[level] = df.loc[df["english_proficiency"] == level, "current_cgpa"].dropna()
        log(f"{level}: n={len(groups[level])}, mean={groups[level].mean():.4f}, sd={groups[level].std(ddof=1):.4f}")

    log("\nH0: mean CGPA is equal across all three English-proficiency groups")
    log("H1: at least one group mean differs")

    lev_stat3, lev_p3 = stats.levene(*groups.values())
    log(f"\nLevene's test for equal variances across groups: F={lev_stat3:.4f}, p={lev_p3:.4g}")

    f_stat, f_p = stats.f_oneway(*groups.values())
    log(f"ANOVA: F={f_stat:.4f}, p-value={f_p:.4g}")

    grand_mean = df["current_cgpa"].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups.values())
    ss_total = ((df["current_cgpa"].dropna() - grand_mean) ** 2).sum()
    eta_sq = ss_between / ss_total
    log(f"Eta-squared (effect size) = {eta_sq:.4f}")

    log("\nPost-hoc pairwise t-tests with Bonferroni correction (3 comparisons, alpha_adj=0.0167):")
    pairs = list(itertools.combinations(groups.keys(), 2))
    for g1, g2 in pairs:
        t_p2, pval = stats.ttest_ind(groups[g1], groups[g2], equal_var=True)
        sig = "SIGNIFICANT" if pval < 0.05 / len(pairs) else "not significant"
        log(f"  {g1} vs {g2}: t={t_p2:.4f}, p={pval:.4g}  -> {sig} (Bonferroni-adjusted)")

    log("\nInterpretation: " + (
        "Reject H0 — mean CGPA differs significantly across English-proficiency groups; "
        "see pairwise comparisons above for which groups drive the difference."
        if f_p < 0.05 else
        "Fail to reject H0 — mean CGPA does not differ significantly across English-proficiency groups."
    ))

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6.5, 5))
    order = ["Basic", "Intermediate", "Advance"]
    sns.boxplot(data=df, x="english_proficiency", y="current_cgpa", order=order)
    plt.title("Current CGPA by English Proficiency (Task 4 ANOVA)")
    plt.xlabel("English proficiency")
    plt.ylabel("CGPA")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cgpa_by_english_proficiency.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    rates = df.groupby("scholarship")["probation"].apply(lambda s: (s == "Yes").mean())
    rates = rates.reindex(["No", "Yes"])
    ax = rates.plot(kind="bar", color=["#4C72B0", "#DD8452"])
    plt.title("Probation Rate by Scholarship Status (Task 4 chi-square)")
    plt.xlabel("Has scholarship?")
    plt.ylabel("Proportion ever on probation")
    plt.xticks(rotation=0)
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "probation_rate_by_scholarship.png", dpi=150)
    plt.close()

    with open(TAB_DIR / "task4_results.txt", "w") as f:
        f.write("\n".join(LOG_LINES))

    log(f"\nFigures saved to: {FIG_DIR}")
    log(f"Full log saved to: {TAB_DIR / 'task4_results.txt'}")


if __name__ == "__main__":
    main()
