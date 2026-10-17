# Task 6 — Critical Evaluation of Experimental Design

No experiment is run for this task (per the assignment brief); this is a
critical discussion of whether, and how, a **Completely Randomized Design
(CRD)** or a **Randomized Complete Block Design (RCBD)** would suit this
consultancy context.

## 6.1 The current dataset is observational, not experimental
`Students_Performance_dataset.csv` is a **cross-sectional self-report
survey** — no variable was assigned to students by a researcher, and
nothing was randomized. Attendance, study habits, scholarship status, and
so on are all things students already had or did before being surveyed.

This matters directly for how Tasks 4 and 5's results should be read: every
significant t-test, chi-square test, ANOVA, and regression/GLM coefficient
in this project describes an **association**, not a proven causal effect.
For example, Task 4.1a shows scholarship holders have significantly higher
CGPA, but we cannot say the scholarship *causes* the higher CGPA — it could
equally be that scholarships are awarded to already-strong students
(reverse causality), or that some third factor (e.g. prior academic
preparation) drives both. Strictly speaking, **CRD and RCBD do not apply
to the analysis already performed**, because both are designs for
prospective experiments with researcher-controlled treatment assignment,
not tools for analyzing existing survey data. What follows is instead an
evaluation of how CRD or RCBD *would* apply if the department acted on
Task 1's proposal and tested an actual intervention.

## 6.2 What CRD and RCBD are
- **CRD (Completely Randomized Design):** experimental units are randomly
  assigned to treatment groups with no other structure imposed. Simple to
  run and analyze (one-way ANOVA/t-test), and appropriate when the units
  are reasonably homogeneous with respect to factors that also affect the
  outcome.
- **RCBD (Randomized Complete Block Design):** units are first grouped
  into **blocks** based on a known nuisance variable (something that
  affects the outcome but isn't the thing being tested), and treatment is
  randomized *within* each block. This removes the blocking variable's
  contribution to residual (error) variance, increasing the power to
  detect a real treatment effect for a given sample size.

## 6.3 A hypothetical intervention experiment
Task 1's second consultancy objective was to build a predictive
early-warning flag; a natural next step — beyond this project's own scope,
but a reasonable one to evaluate design options for — would be:

> Randomly assign students flagged as at-risk (via the Task 5 logistic/GLM
> model) to either a **structured teacher-consultation intervention** or
> **standard support** (active control, not no support at all — see
> ethics note in §6.5), then measure whether the intervention group shows
> a higher subsequent-semester CGPA and/or lower probation incidence than
> the control group.

## 6.4 CRD suitability
Under CRD, flagged at-risk students would simply be randomly split into
intervention vs control with no further structure.

**In favour:** simple to implement and administer; simple to analyze
(independent-samples t-test on the outcome, exactly the technique already
used in Task 4).

**Against, based on this project's own findings:** Tasks 4 and 5 already
show the at-risk population is **not** homogeneous on factors that matter
for the outcome:
- English proficiency (Task 4.4: significant CGPA differences between
  Basic and the other two levels)
- Scholarship status (Task 4.1a/4.2a: large, highly significant CGPA and
  probation-rate gaps)
- Semester / academic-progress stage (Task 5.1/5.2: significant predictor
  in both the CGPA and probation models)

With a real intervention cohort likely much smaller than this project's
1,194-row convenience sample (probably one department, one semester's
worth of flagged students), pure random assignment could easily leave one
arm with disproportionately more low-English-proficiency or
non-scholarship students by chance — inflating residual variance and
either masking a genuine intervention effect or wrongly crediting the
intervention for what is actually a pre-existing group imbalance. CRD's
simplicity is attractive but statistically inefficient here, given that
the department already has empirical grounds (from this very project) to
expect these nuisance variables to matter.

## 6.5 RCBD suitability (recommended)
Blocking on one or more of the empirically justified confounders above,
then randomizing intervention vs control **within each block**, directly
addresses CRD's weakness:

- **Primary block: scholarship status** (2 levels) — the single strongest,
  most significant confound found in this project (Task 4.1a, 4.2a).
- **Secondary block/covariate, resources permitting: semester band**
  (e.g. early / mid / late, binned from the continuous semester variable)
  — a significant predictor of both outcomes in Task 5, and a reasonable
  proxy for how much time a student has had to fall behind.
- English proficiency is a defensible third candidate but has the smallest
  effect size of the three (Task 4.4: η² = 0.0175) and would fragment an
  already-small at-risk cohort into more, thinner blocks — a tradeoff to
  weigh against available sample size.

**Benefit:** blocking removes known, already-quantified between-student
variance *before* testing the intervention, which increases statistical
power to detect a real effect for the same number of participants, and
produces genuinely causal evidence — something the current observational
analysis (however statistically significant) cannot provide, since no
variable was ever randomly assigned in it.

## 6.6 Practical and ethical constraints (both designs)
- **Ethics:** withholding a plausibly beneficial, low-cost support
  intervention from genuinely at-risk students purely for study design
  purposes is questionable. A **wait-list control** (control group
  receives the intervention on a delay, after outcomes are measured) or an
  **active control** (standard support vs the enhanced intervention, rather
  than a true no-treatment arm) is more defensible than a strict
  placebo-style control group.
- **Feasibility / block sizes:** only one semester's cohort of at-risk
  students is available at a time. In the full dataset, the smallest
  English-proficiency group (Advance, n=214 overall) is already the
  minority; the *at-risk* subset of it will be smaller still, and some
  block × treatment cells could end up sparse or empty. This may force
  collapsing blocks (e.g. combining Intermediate and Advance) or excluding
  underpopulated combinations, which should be planned for in advance
  rather than discovered mid-study.
- **Statistical power:** Task 4.4's ANOVA found a *significant but small*
  effect size for English proficiency (η² = 0.0175) despite a large sample
  (n=1,155). A future RCBD intervention trial, likely working with a much
  smaller at-risk cohort, would need its own formal power analysis to size
  blocks and replicates correctly — this project's convenience-sample
  effect sizes are a starting point for that calculation, not a
  substitute for it.

## 6.7 Recommendation
An **RCBD, blocking primarily on scholarship status** (and secondarily on
semester band, if the cohort is large enough to support it), is the more
statistically defensible design for testing the proposed early-warning
intervention. It directly neutralizes the two strongest, already-quantified
sources of between-student variability identified in Tasks 4 and 5, at the
cost of modest added logistical complexity in recruitment and assignment.
**CRD remains a reasonable fallback** only if the eventual intervention
cohort turns out to be small and reasonably homogeneous on these factors,
or if practical constraints make block-stratified recruitment infeasible —
but given what this project has already found about the population, that
would be a design choice made in spite of the evidence, not because of it.
