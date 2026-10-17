# Task 4 — Statistical Inference

Run `python src/task4_inference.py` (after `clean_data.py`) to regenerate every
number below. Full console output is also saved to
`outputs/tables/task4_results.txt`. Tests use `scipy.stats` (independent
t-tests, Levene's test, chi-square/Fisher, one-way ANOVA).

All tests below were chosen to speak directly to Task 1's consultancy
objectives: which behavioural/demographic factors are associated with
academic outcomes (CGPA, probation, suspension), and are they statistically
reliable given the sample, or just noise?

## 4.1 Comparison of means

### 4.1a — Current CGPA by scholarship status (independent-samples t-test)
**H0:** mean CGPA is equal for scholarship holders and non-holders.
**H1:** mean CGPA differs by scholarship status (two-tailed).

| Group | n | Mean CGPA | SD |
|---|---|---|---|
| Scholarship = Yes | 493 | 3.4005 | 0.4552 |
| Scholarship = No | 662 | 3.1772 | 0.4784 |

Levene's test for equal variances: F = 0.7735, p = 0.379 → assumption
**holds**, so a standard (Student's) t-test is used.

**Result:** t = 8.011, df ≈ 1153, **p = 2.77 × 10⁻¹⁵**. Hedges' g = 0.476
(medium effect).

**Interpretation:** We reject H0. Scholarship holders have a significantly
higher mean CGPA (3.40 vs 3.18) than non-holders. This is consistent with
either (a) scholarships being awarded on merit/prior performance, and/or
(b) the financial and motivational support of a scholarship helping
maintain performance — the data cannot distinguish direction of causality,
only association (see Task 6 for why a controlled experiment would be
needed to establish causality).

### 4.1b — Attendance by probation history (independent-samples t-test)
**H0:** mean attendance is equal for students with and without a probation
history. **H1 (one-tailed):** students with a probation history have
*lower* mean attendance.

| Group | n | Mean attendance (%) | SD |
|---|---|---|---|
| Probation = Yes | 298 | 82.06 | 19.10 |
| Probation = No | 896 | 90.32 | 13.21 |

Levene's test: F = 49.92, p = 2.72 × 10⁻¹² → assumption **violated**
(probation group is noticeably more variable), so **Welch's t-test** is
used instead of Student's.

**Result:** t = −6.938, one-tailed **p = 8.16 × 10⁻¹²**. Hedges' g = −0.554
(medium-large effect).

**Interpretation:** We reject H0. Students who have ever been on probation
attended class roughly 8 percentage points less, on average, than students
who never have — and the gap is both statistically and practically
meaningful. This directly supports Task 1's premise that attendance is a
usable early-warning signal.

## 4.2 Comparison of proportions

### 4.2a — Probation rate by scholarship status (chi-square test of independence)
**H0:** probation status is independent of scholarship status.
**H1:** the two are associated.

| | Probation = No | Probation = Yes |
|---|---|---|
| Scholarship = No | 448 | 223 |
| Scholarship = Yes | 448 | 75 |

χ² = 55.01, df = 1, **p = 1.20 × 10⁻¹³** (minimum expected cell count =
130.5, well above the rule-of-thumb minimum of 5, so the chi-square
assumption is satisfied). Cramér's V = 0.215 (small-to-medium association).

- Probation rate | scholarship = Yes: **14.3%**
- Probation rate | scholarship = No: **33.2%**

**Interpretation:** We reject H0. Non-scholarship students are more than
twice as likely to have a probation history as scholarship holders — a
substantial, policy-relevant gap for the scholarship committee stakeholder
identified in Task 1.

### 4.2b — Suspension rate by gender (chi-square test)
**H0:** suspension status is independent of gender. **H1:** associated.

| | Suspension = No | Suspension = Yes |
|---|---|---|
| Female | 508 | 14 |
| Male | 640 | 32 |

Minimum expected count = 20.1 (chi-square assumption satisfied, Fisher's
exact test not required). χ² = 2.89, df = 1, **p = 0.089**.

- Suspension rate | Male: 4.76%
- Suspension rate | Female: 2.68%

**Interpretation:** We fail to reject H0 at α = 0.05 — the apparent gap
between genders is not statistically significant. Suspension is a rare
event in this sample (46 of 1,194 students, 3.9%), which limits statistical
power; this test should not be read as proof of "no real difference," only
that this sample cannot confirm one.

## 4.3 Comparison of variances (reported alongside 4.1a/4.1b above)
Levene's test was used as the formal variance-comparison procedure ahead of
each t-test, since the correct t-test variant (Student vs Welch) itself
depends on the outcome of a variance-equality test:
- CGPA by scholarship: variances **statistically indistinguishable**
  (F = 0.77, p = 0.38).
- Attendance by probation: variances **significantly different**
  (F = 49.92, p < 0.001) — the probation group's attendance is far more
  spread out (SD 19.1 vs 13.2), which is itself a useful finding: at-risk
  students are a more heterogeneous group, not just a uniformly
  lower-attendance one.

## 4.4 One-way ANOVA — Current CGPA by English proficiency
**H0:** mean CGPA is equal across Basic / Intermediate / Advance English
proficiency. **H1:** at least one group mean differs.

| Group | n | Mean CGPA | SD |
|---|---|---|---|
| Basic | 291 | 3.163 | 0.577 |
| Intermediate | 650 | 3.305 | 0.420 |
| Advance | 214 | 3.324 | 0.491 |

Levene's test flags unequal variances across groups (F = 10.83, p < 0.001)
— a caveat on the classic ANOVA's equal-variance assumption, noted here
rather than hidden; with group sizes this large (n ≥ 214 per group) ANOVA's
F-test is reasonably robust to moderate variance heterogeneity, but a
Welch's ANOVA would be a natural robustness check if this were taken
further.

**Result:** F = 10.28, **p = 3.75 × 10⁻⁵**. Eta-squared = 0.0175.

**Post-hoc pairwise comparisons** (independent t-tests, Bonferroni-adjusted
α = 0.05/3 = 0.0167 for 3 comparisons):

| Comparison | t | p | Significant? |
|---|---|---|---|
| Basic vs Intermediate | −4.218 | 2.70 × 10⁻⁵ | Yes |
| Basic vs Advance | −3.280 | 0.00111 | Yes |
| Intermediate vs Advance | −0.550 | 0.582 | No |

**Interpretation:** We reject H0. Students with only "Basic" English
proficiency have significantly lower CGPA than both the Intermediate and
Advance groups, which do not differ meaningfully from each other. However,
the effect size is small (η² = 0.0175, i.e. English proficiency explains
under 2% of the variance in CGPA) — statistically real, given the sample
size, but not a primary lever for the department compared with attendance
or scholarship status. This is a good example of statistical significance
not implying large practical importance, and it is reported as such rather
than overstated.

## 4.5 Summary of Task 4 findings for the consultancy report
| Finding | Direction | Strength |
|---|---|---|
| Scholarship → higher CGPA | Yes | Medium effect, highly significant |
| Probation history → lower attendance | Yes | Medium-large effect, highly significant |
| Scholarship → lower probation rate | Yes | Small-medium association, highly significant |
| Gender → suspension rate | No (not confirmed) | Not significant, low power |
| English proficiency → CGPA | Yes (Basic vs others only) | Small effect, statistically significant |

These results feed directly into the predictor set and interpretation used
in Task 5.
