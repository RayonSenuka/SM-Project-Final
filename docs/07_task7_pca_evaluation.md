# Task 7 — Critical Evaluation of Principal Component Analysis (PCA)

Run `python src/task7_pca.py` (after `clean_data.py`) to regenerate every
number below. Full console output is saved to
`outputs/tables/task7_results.txt`; component variance, loadings and
model-comparison tables are saved as CSVs in `outputs/tables/`.

The assignment brief only requires a *critical discussion* of whether PCA
would help this dataset, not a required deliverable model. Rather than
argue this abstractly, this section runs PCA on the same 15-predictor set
used throughout Tasks 4–5 and 7 and lets the actual numbers settle the
question, in the same evidence-based style as the rest of this report.

## 7.1 Is dimensionality reduction necessary here?

The short answer is **no, not really — and the data shows why.**

PCA earns its keep when a set of predictors is **highly correlated /
redundant**, so that a handful of components can stand in for most of the
original variables without much information loss. Task 5's VIF check
(§5.1.1) already hinted at the answer: **all 15 predictors have VIF < 7**,
with only `semester` (5.39) and `credits_completed` (6.16) elevated — and
even that pair, while the strongest correlation in the whole set
(r = 0.908, both tracking a student's academic progress), is a single
pair, not a systemic redundancy problem. The Task 3 correlation heatmap
(`outputs/figures/correlation_heatmap.png`) shows the same pattern among
the purely numeric variables: most pairwise correlations are under 0.2.

Running PCA directly on the standardized 15-predictor matrix confirms
this quantitatively:

| Component | Eigenvalue | Variance explained | Cumulative |
|---|---|---|---|
| PC1 | 2.62 | 17.5% | 17.5% |
| PC2 | 1.53 | 10.2% | 27.7% |
| PC3 | 1.18 | 7.9% | 35.5% |
| PC4 | 1.15 | 7.7% | 43.2% |
| PC5 | 1.11 | 7.4% | 50.6% |
| PC6–PC9 | 0.86–0.99 | 6.0–6.6% each | up to 75.7% |
| PC10–PC15 | 0.09–0.85 | 0.6–5.7% each | up to 100% |

**10 of the 15 components are needed to reach 80% cumulative variance, and
12 are needed for 90%** (Kaiser's eigenvalue > 1 rule keeps only 5
components, and even those explain barely half the total variance). The
scree plot (`outputs/figures/task7_pca_scree.png`) is essentially flat
after PC1 — there is no "elbow" that would justify keeping a small number
of components. This is the signature of a predictor set that is **already
close to its own effective dimensionality**: PCA has almost nothing
redundant to compress. That is itself a useful, quantified finding for the
consultancy report, not a null result — it confirms that the 15 predictors
selected in Task 1/5 were reasonably non-overlapping in the first place.

## 7.2 Possible advantages (and why they don't apply strongly here)

The textbook advantages of PCA are real in general, but each one is weak
in this specific case:

- **Multicollinearity reduction** — would help most models that suffer
  from unstable, inflated-variance coefficients under correlated
  predictors. Task 5's OLS and logistic models already have stable,
  interpretable coefficients with modest standard errors (§5.1, §5.2)
  precisely because multicollinearity was never severe here.
- **Noise reduction / compact representation** — useful when many
  variables carry redundant signal; the flat scree profile shows that is
  not the case.
- **Visualization in 2–3 dimensions** — PC1–PC2 together explain only
  27.7% of the variance, so a 2D PCA scatterplot of students would discard
  more than 70% of the predictor information — not a faithful summary
  for this dataset (unlike, e.g., high-dimensional genomic or image data
  where the first few PCs often dominate).
- **Speed** — irrelevant at n ≈ 1,200 rows and 15 predictors; none of
  Task 5's models were computationally constrained.

## 7.3 Limitations, made concrete with real loadings

Beyond the general critique that PCA is unsupervised (it maximizes
variance in the predictors, not their relationship to CGPA or probation)
and sensitive to variable scaling (mitigated here by standardizing first,
as in Task 5's regularization), the actual loadings show a second, more
practical problem for *this* consultancy: **the components mix variables
from unrelated domains**, which works against the report's main goal of
identifying specific, actionable levers.

| Component | Variance | Variables that load ≥ 0.30 |
|---|---|---|
| PC1 | 17.5% | `credits_completed` (+0.56), `semester` (+0.53), `scholarship_yes` (−0.36), `log_family_income` (+0.31) |
| PC2 | 10.2% | `study_hours` (+0.60), `study_sessions` (+0.58), `skill_dev_hours` (+0.33) |
| PC3 | 7.9% | `gender_male` (+0.61), `log_family_income` (−0.39), `social_media_hours` (−0.34), `skill_dev_hours` (+0.32) |
| PC4 | 7.7% | `co_curricular_yes` (+0.59), `uses_transport_yes` (+0.47), `english_proficiency` (+0.35), `teacher_consultancy_yes` (+0.34) |
| PC5 | 7.4% | `learning_mode_online` (+0.62), `social_media_hours` (+0.51), `attendance` (−0.42) |

PC1 blends academic progress, a demographic scholarship indicator, and
family income into a single axis; PC5 blends learning mode, social media
use, and attendance with opposite signs. None of these components has a
clean, presentable label a department head could act on — compare this to
Task 5's finding that **attendance and scholarship status** are, on their
own, the two clearest actionable levers. Compressing them into
multi-variable components would make Task 12's final recommendations
*harder* to communicate to non-technical stakeholders (the department
administration and scholarship committee named in Task 1), not easier —
directly working against a consultancy's core deliverable.

## 7.4 Expected impact on predictive modelling — tested directly

The 15-predictor Task 5 models were re-fit on principal components instead
of the original variables, using the **same train/test split and
evaluation metrics** as Task 5, so the comparison is apples-to-apples.

**Regression (`current_cgpa`), test set:**

| Model | Predictors/components | Test R² | Test RMSE |
|---|---|---|---|
| OLS, original 15 predictors (Task 5 baseline) | 15 | 0.164 | 0.402 |
| PCR, top 10 PCs (80% variance) | 10 | **0.130** | **0.410** |
| PCR, all 15 PCs | 15 | 0.164 | 0.402 |

**Classification (`probation`), test set:**

| Model | Predictors/components | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| Logistic, original 15 predictors, balanced (Task 5 baseline) | 15 | 0.636 | 0.366 | 0.617 | 0.460 | 0.701 |
| PC-logistic, top 10 PCs, balanced | 10 | 0.661 | 0.390 | 0.617 | 0.477 | **0.693** |

Two things are worth noting. First, using *all 15* components gives
identical performance to the original variables (expected — PCA at full
rank is just a rotation of the same information, so there is never a
predictive-accuracy argument for using all components). Second, truncating
to the components needed for 80% variance (10 of 15) **costs some
predictive performance on the regression task** (test R² drops from 0.164
to 0.130) for classification the picture is more mixed — accuracy ticks up
slightly but AUC (the more appropriate metric for a probability-ranking
early-warning tool, per Task 5.5) drops slightly. In neither task does
truncated PCA provide a predictive-accuracy case for its use here.

## 7.5 When PCA should — and should not — be used (situating this dataset)

Drawing on the pattern documented across statistical-modelling and
educational-data-mining literature (e.g. applications of PCA in
high-dimensional learning-analytics settings with dozens to hundreds of
correlated behavioural/clickstream features), PCA earns its place when:

- The predictor set is large relative to the sample size (p approaching or
  exceeding n), where dimensionality reduction is close to mandatory for
  stable estimation.
- Predictors are strongly, systematically correlated (e.g. repeated
  measures of the same underlying construct — several highly correlated
  test-score items, or many correlated sensor/pixel readings).
- The downstream use case genuinely does not require interpreting
  individual variables (e.g. PCA as a pre-processing step purely to feed
  a black-box predictive model, with no need to explain *why*).

None of these conditions hold strongly for this project: p = 15 is small
relative to n ≈ 1,200; the predictors were deliberately curated in Task 1
to cover distinct behavioural/demographic domains and are, as shown above,
only mildly correlated; and the consultancy's stated objective (Task 1) is
explicitly to identify **specific, actionable, explainable factors** for
department policy — the opposite of a black-box use case.

## 7.6 Conclusion for this consultancy

**PCA is not recommended for this project.** The dataset does not exhibit
the multicollinearity or dimensionality problem PCA is designed to solve
(all VIFs < 7, flat scree profile, 10–12 of 15 components needed for
80–90% variance), truncated PCA measurably costs predictive performance
rather than improving it, and the resulting components are harder to
interpret and communicate than the original named variables — a direct
cost given this project's consultancy purpose. The original 15-predictor
models from Task 5 remain the recommended approach, consistent with
Task 5.6's model recommendation.
