# Task 10 — Industry Innovation Proposal
**Solution:** Academic Risk Early-Warning System (EWS) with an Advisor Decision-Support Dashboard

## 1. Business need
The BCSE department currently reacts after probation or suspension is already recorded (Task 1). Our data show the scale of the problem: **298 of 1,194 students (25.0%) have been on academic probation**, and probation is strongly linked to low attendance (mean 82.1% vs 90.3% for students never on probation, Welch t-test p < 0.001, Task 4) and to scholarship status (probation rate 14.3% with scholarship vs 33.2% without, chi-square p < 0.001). Advisors have limited time, so the real need is not just a prediction but a **prioritised, explainable worklist** that tells them whom to contact first and why.

## 2. Proposed solution
A three-layer system (see `outputs/figures/task10_conceptual_framework.png`):

1. **Data layer** — attendance, semester, credits completed, study and social-media hours, English proficiency, scholarship, learning mode, teacher consultation. Gender and family income are deliberately excluded (Section 5).
2. **Analytics layer** — a standardised logistic regression (the GLM recommended in Task 5) producing a probability of probation for each student, scored with 5-fold cross-validation so no student is scored by a model that has seen them. Probabilities are converted to tiers:
   - **Red** (risk ≥ 40%): advisor meeting within two weeks
   - **Amber** (25–40%): attendance nudges, peer mentoring, monitoring
   - **Green** (< 25%): routine support
   Each flagged student receives **reason codes** limited to levers an advisor can act on (e.g. "Low attendance + No scholarship").
3. **Decision layer** — a Streamlit dashboard (`src/task10_dashboard.py`) with a department overview, a filterable advisor worklist (CSV export), a what-if calculator, and policy sliders (cutoffs, advisor capacity). The cost-based cutoff logic comes from the Bayesian decision analysis in Task 8.

## 3. Evidence that the tiers work (out-of-fold, n = 1,194)
| Tier | Students | Share | Actual probation rate |
|---|---|---|---|
| Green | 698 | 58% | 15.0% |
| Amber | 315 | 26% | 31.1% |
| Red | 181 | 15% | **52.5%** |

The Red tier has a probation rate **2.1× the base rate** while covering only 15% of students; Red + Amber (42% of students) contains **65% of all probation cases**. Model AUC is 0.706. The cross-validated AUC is unchanged when gender and income are removed (0.7059 vs 0.7063).

If advisors can only see the top 10% riskiest students, precision is 58.8% (lift 2.4×); the top 5% gives 65% precision.

## 4. Expected benefits
- **Earlier, targeted support:** advisors start from a ranked list instead of waiting for results.
- **Cost-efficiency (illustrative cost matrix: missing an at-risk student = 5, one unnecessary advising slot = 1):** doing nothing costs 1,490 units; flagging Red + Amber costs 828 (−44%); the cost-optimal cutoff costs 756 (−49%). *Caveat:* the cost-optimal cutoff flags 63% of students, so capacity, not cost alone, sets the real cutoff. The costs are assumptions to be agreed with the department, not data-derived.
- **Attendance scenario:** raising attendance by 10 points for Red/Amber students is associated with about 22 fewer expected probation cases (−11.6%) among the 496 flagged students; +15 points gives about 30 fewer (−15.5%). This is association, not proven causation (Task 6: a pilot RCBD would test it).
- **Transparency:** reason codes and the what-if tool let advisors discuss the result with the student.

## 5. Implementation challenges and mitigations
| Challenge | Mitigation |
|---|---|
| Modest accuracy (AUC ≈ 0.70, pseudo-R² ≈ 0.12) | Use as a *screening aid*, never an automatic decision; keep human review |
| Cross-sectional data; "ever on probation" accumulates with semesters, so semester dominates the score | Collect semester-by-semester records so the model predicts *future* probation (links to Task 9) |
| Fairness | Gender and family income excluded; audit tier rates by group each semester; support, never punishment |
| Data privacy | Role-based access, anonymised IDs, consent and policy notice to students |
| Staff adoption and workload | Pilot with one cohort; capacity slider; short training |
| Model drift | Retrain every semester; monitor calibration |

## 6. Required resources
- Data: attendance system export, registrar records (existing).
- Tools: Python (pandas, scikit-learn), Streamlit; optional Power BI for production.
- People: one data analyst (part-time), 2–3 advisors for the pilot, IT support for secure hosting.
- Cost: open-source software; main cost is staff time.

## 7. Organisational impact
Shifts the department from reactive to proactive support, gives scholarship and advising committees a consistent evidence base, and creates the longitudinal data needed for forecasting later.

*Add 1–2 learning-analytics / early-warning-system papers from your own literature search to support this section (verify each before citing).*
