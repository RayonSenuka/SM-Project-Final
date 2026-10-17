# Task 8 — Critical Evaluation of Bayesian Statistical Methods

Run `python src/task8_bayesian.py` (after `clean_data.py`) to regenerate
every number below. Full console output is saved to
`outputs/tables/task8_results.txt`; comparison and coefficient tables are
saved as CSVs in `outputs/tables/`, and the posterior coefficient forest
plot is saved to `outputs/figures/task8_bayesian_ridge_forest.png`.

As with Task 7, this is a discussion task, grounded here in three small,
real computations on the same predictor set and train/test splits used in
Tasks 5 and 7, so the evaluation is evidence-based rather than purely
theoretical.

## 8.1 Naïve Bayes

### What it is and why it might apply
Naïve Bayes classifies by applying Bayes' theorem and assuming predictors
are **conditionally independent given the class**. For `probation`
(binary), Gaussian Naïve Bayes models each numeric/dummy predictor as
class-conditionally Gaussian.

### Is the independence assumption reasonable here?
Not perfectly, but not badly either. The strongest pairwise correlation
among the 15 predictors is `semester`/`credits_completed` at **r = 0.908**
(the same pair flagged by Task 5's VIF check and Task 7's PCA loadings) —
a real, known violation of conditional independence, since both track
academic progress. Every other pair is far weaker. So Naïve Bayes'
core assumption is **violated in degree, not comprehensively** — exactly
the situation the method is known to be fairly robust to in practice: NB
frequently performs competitively even under moderate correlation because
it only needs the class-conditional *decision boundary* implied by the
independence assumption to be reasonable, not the assumption itself to be
literally true.

### Head-to-head result
Gaussian Naïve Bayes was fit on the same standardized 15-predictor matrix
and the same held-out test split as Task 5's class-balanced logistic
model:

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Gaussian NB (default priors) | 0.799 | 0.625 | 0.500 | 0.556 | **0.727** |
| Logistic/GLM, balanced (Task 5 recommended) | 0.636 | 0.366 | 0.617 | 0.460 | 0.701 |
| Gaussian NB (priors forced 50/50) | 0.665 | 0.396 | 0.633 | 0.487 | 0.727 |

Two findings stand out. First, **NB's AUC (0.727) is slightly higher than
logistic's (0.701)** despite the independence-assumption violation —
its underlying ability to *rank* students by probation risk is, if
anything, marginally better here. Second, NB's default class priors match
the sample's true 75/25 split, which is why the default-prior NB has high
accuracy but only catches half of at-risk students (recall 0.500); forcing
50/50 priors — the fairer comparison to logistic's `class_weight =
"balanced"`, which does the analogous re-weighting — brings NB's recall
(0.633) in line with logistic's (0.617), at similar accuracy, with NB
still keeping its AUC edge.

### Advantages, limitations, and recommendation
**Advantages:** extremely fast to train (closed-form, no iterative
optimization, unlike logistic's IRLS in Task 5); performs well with modest
sample sizes; a competitive, non-obvious alternative candidate here.
**Limitations:** the class-conditional Gaussian assumption is a real
simplification for binary dummy predictors (e.g. `scholarship_yes`),
which are not remotely Gaussian; NB does not produce directly
interpretable effect sizes analogous to logistic's odds ratios (§5.2),
which matter for explaining findings to non-technical stakeholders (Task
1); and NB's probability estimates are known to be poorly *calibrated*
even when its *ranking* (AUC) is good — a caveat before using its raw
probabilities in the decision-theory framework of §8.3 without
recalibration.
**Recommendation:** NB is worth reporting as a strong, cheap benchmark
that cross-validates logistic regression's basic conclusion (attendance-
and scholarship-linked variables separate the two risk groups), but the
**class-balanced logistic/GLM model remains the recommended model** for
the final consultancy deliverable (Task 5.6), because its odds ratios are
directly interpretable for the department stakeholders, whereas NB is
best used as a supporting cross-check.

## 8.2 Bayesian Regression

### What it is and why it might apply
Bayesian (Ridge) regression places a prior distribution over the
regression coefficients and returns a full **posterior distribution**
(mean and uncertainty) for each coefficient, rather than a single point
estimate. Applied here to `current_cgpa`, it is directly comparable to
Task 5's OLS and CV-tuned Ridge.

### Result vs OLS and frequentist Ridge

| Model | Test R² | Test RMSE |
|---|---|---|
| OLS (Task 5 baseline) | 0.164 | 0.402 |
| Ridge, α = 3.56 (Task 5's CV-selected value) | 0.165 | 0.402 |
| Bayesian Ridge (evidence-tuned) | **0.169** | **0.401** |

Bayesian Ridge edges out both by a small margin — and, notably, it reached
its regularization strength **without a cross-validation grid search**:
its precision parameters (noise precision α = 4.73, coefficient precision
λ = 287.4, implying a prior coefficient variance of ≈ 0.003) were learned
directly from the training data by maximizing model evidence (the
marginal likelihood). This is a genuine practical advantage over Task 5's
Ridge/Lasso/Elastic Net, which needed a separate 5-fold CV loop over a
penalty grid (§5.3) to achieve a comparable result.

### What the posterior adds beyond a point estimate
The posterior standard deviation on each coefficient gives a 95%
**credible interval** — the Bayesian analogue of a confidence interval,
but interpretable directly as "there is a 95% probability the true
(standardized) effect lies in this range," conditional on the model and
prior. Ranking predictors by whether their credible interval excludes
zero reproduces Task 5's significance pattern almost exactly:

**Credible interval excludes 0 (i.e. a robust, non-zero effect):**
`credits_completed`, `scholarship_yes`, `attendance`,
`english_proficiency`, `social_media_hours`, `learning_mode_online`,
`semester`.

**Credible interval includes 0 (no robust effect):** `study_hours`,
`study_sessions`, `skill_dev_hours`, `log_family_income`,
`uses_transport_yes`, `teacher_consultancy_yes`, `co_curricular_yes`,
`gender_male`.

This is exactly the same seven "real effect" vs eight "no robust effect"
split that Task 5's OLS p-values (§5.1) and Lasso's coefficient shrinkage
(§5.3) independently produced — a third, philosophically different
inferential framework (Bayesian credible intervals, vs frequentist
p-values, vs an L1 penalty selecting a sparse model) converging on the
same substantive conclusion is a strong form of robustness evidence for
the consultancy report.

### Advantages, limitations, and recommendation
**Advantages:** automatic, data-driven regularization strength; full
uncertainty quantification on every coefficient "for free," useful for
communicating confidence (or lack of it) to stakeholders; naturally
handles the mild multicollinearity flagged in Task 5/Task 7 without a
separate VIF-driven decision.
**Limitations:** requires choosing (or accepting default, weakly
informative) priors, which a stakeholder unfamiliar with Bayesian
methods may find harder to sign off on than a plain OLS coefficient
table; credible intervals here still rest on a Gaussian-likelihood
assumption, so they inherit the same residual non-normality caveat noted
for OLS in §5.1.1; and the *predictive* gain over Task 5's models is
small (ΔR² ≈ 0.005) — not transformative.
**Recommendation:** Bayesian Ridge is a reasonable **complement** to
Task 5's stepwise OLS model, particularly for the credible-interval
uncertainty statements it offers "for free," but given the negligible
accuracy gain, the **stepwise OLS model from Task 5.6 remains recommended
as the primary model** for its simplicity and stakeholder familiarity.

## 8.3 Bayesian Decision-Making

### Why this matters more than either model above
Task 5.5 already showed that the choice between the class-balanced and
unweighted logistic model is really a choice about a **cost trade-off**
(missing at-risk students vs raising false alarms), decided there by
qualitative argument. Bayesian decision theory makes that trade-off
explicit and quantitative: given a posterior probability p = P(probation |
student's data) from either the logistic or NB model, and an explicit cost
for each of the four possible outcomes, the optimal action is the one that
**minimizes expected cost**, not the one implied by an arbitrary 0.5
probability cutoff.

### Worked example
A genuinely borderline student from the held-out test set (posterior
probability closest to 0.5, actual p = 0.502) was used, together with an
illustrative cost matrix reflecting Task 1's early-warning framing (a
missed at-risk student is treated as five times costlier than one
unnecessary advising conversation):

| Outcome | Cost |
|---|---|
| Intervene, student was not at risk | 1 (one unneeded advising slot) |
| Intervene, student was at risk | 0 (correct catch) |
| No action, student was not at risk | 0 (correct) |
| No action, student was at risk | 5 (missed at-risk student) |

E[cost \| intervene] = 0.502 × 0 + 0.498 × 1 = **0.498**
E[cost \| no action] = 0.502 × 5 + 0.498 × 0 = **2.508**

→ **Intervene** minimizes expected cost, even though the posterior
probability (0.502) is barely above chance — because the cost of a miss is
so much higher than the cost of a false alarm. A plain 0.5 accuracy-style
cutoff would have reached the same decision here only by coincidence (this
case sits almost exactly at 0.5); the framework's real value shows up when
p is lower.

### Threshold sensitivity
Solving for the break-even posterior probability (the threshold at which
the two expected costs are equal) at different assumed cost ratios shows
how much the "right" flagging threshold moves:

| Cost ratio (miss : false alarm) | Break-even threshold |
|---|---|
| 1:1 | 0.500 |
| 2:1 | 0.333 |
| 5:1 | 0.167 |
| 10:1 | 0.091 |

If the department genuinely believes a missed at-risk student is ten times
costlier than one unnecessary advising conversation, the model should flag
*any* student with a posterior probability above **0.091** — far more
permissive than the default 0.5 cutoff, and this is precisely the
reasoning that justified using the *class-balanced* logistic model over
the higher-accuracy *unweighted* one in Task 5.5, now made explicit and
adjustable rather than argued qualitatively.

### Advantages, limitations, and recommendation
**Advantages:** turns a vague "accuracy vs recall" trade-off into an
explicit, auditable, adjustable policy lever the department can own
(they set the cost ratio, not the statistician); works with the posterior
probabilities from *either* the logistic or NB model above.
**Limitations:** the cost matrix here is illustrative, not derived from
data — a real deployment would need the department to estimate actual
costs (advisor time, and the cost of a missed intervention in terms of
eventual dropout/extra semesters), which is a genuine, non-trivial
elicitation exercise; and as noted in §8.1, if NB's probabilities are used
here they should first be checked for calibration, since NB is known to
produce good rankings but sometimes poorly calibrated probabilities.
**Recommendation:** adopt this framework, not a fixed 0.5 cutoff, for any
operational deployment of the early-warning system proposed in Task 10 —
it is the clearest, most directly actionable output of this entire
Bayesian evaluation.

## 8.4 Summary for the consultancy report

| Method | Verdict |
|---|---|
| Naïve Bayes | Strong, fast benchmark; slightly better AUC than logistic despite the independence assumption being mildly violated; kept as a cross-check, not the primary model (interpretability favours logistic) |
| Bayesian regression | Small predictive edge over OLS/Ridge; its main value is automatic regularization tuning and credible intervals that independently confirm Task 5's significant predictors |
| Bayesian decision-making | The most directly useful output of Task 8 — replaces Task 5.5's qualitative balanced-vs-unweighted argument with an explicit, department-owned cost-based threshold, recommended for the Task 10 early-warning system |
