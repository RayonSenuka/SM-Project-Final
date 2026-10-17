# Task 12 — Final Consultancy Recommendations
**Client:** BCSE department, university
**Audience:** senior management (Board)
**Question we answered:** can attendance, together with study habits and background, predict CGPA and probation risk early enough for the university to step in?

**Short answer:** yes, with limits. Attendance is the strongest controllable signal, the model ranks students usefully (AUC ≈ 0.70) but is not accurate enough to decide about any one student, and every result so far is an association. We therefore recommend an attendance-led, human-reviewed early-warning pilot, not an automatic system.

> Every recommendation below is tied to (1) statistical evidence from our own result files, (2) research literature from Task 2, and (3) industry expert feedback from Task 11. Section 7 is the place to record the expert feedback; it is still empty and must be completed before submission.

---

## 1. Evidence base at a glance

| Finding | Value | Source file |
|---|---|---|
| Students with a probation history | 298 of 1,194 (25.0%) | `task5_results.txt` |
| Mean attendance, probation vs no probation | 82.1% vs 90.3% (Welch t = −6.94, p < 0.001, g = −0.55) | `task4_results.txt` |
| Probation rate by scholarship | 14.3% with, 33.2% without (χ² = 55.0, V = 0.21) | `task4_results.txt` |
| Mean CGPA by scholarship | 3.40 vs 3.18 (t = 8.01, g = 0.48) | `task4_results.txt` |
| Attendance odds ratio (logistic GLM) | 0.972 per point (95% CI 0.963–0.980) | `task5_results.txt` |
| Scholarship odds ratio (logistic GLM) | 0.465 (95% CI 0.334–0.647) | `task5_results.txt` |
| Model skill | AUC ≈ 0.70; recall 0.62; test R² 0.164–0.177 | `task5_results.txt` |
| Risk tiers (out-of-fold) | Green 15.0%, Amber 31.1%, Red 52.5% probation | `task10_results.txt` |
| Red tier | 2.1× the base rate with 15.2% of students | `task10_results.txt` |
| Red + Amber | 41.6% of students hold 64.8% of cases | `task10_results.txt` |
| Gender and income removed | AUC 0.7059 vs 0.7063 (no loss) | `task10_results.txt` |

---

## 2. Strategic recommendations

### S1 — Adopt an attendance-led early-warning programme instead of blanket outreach
- **Statistical evidence:** attendance is the strongest controllable predictor in the deployment model (standardised coefficient −0.43; OR 0.65 per 1 SD). Probation is 56.1% among students at 60% attendance or below (n = 82) against 17.9% above 90% (n = 605).
- **Literature:** Stanca (2006) finds a positive attendance effect after controlling for unobserved ability; Karnik et al. (2020) find the benefit persists at higher grade percentiles; Marburger (2006) links mandatory attendance policies to better performance.
- **Industry expert feedback:** *[to be completed — Section 7]*
- **Decision for management:** approve attendance as the primary monitored signal.

### S2 — Prioritise low-attendance, non-scholarship students first
- **Statistical evidence:** probation is 14.3% with a scholarship and 33.2% without; scholarship OR 0.465. In OLS, a scholarship is associated with +0.25 CGPA points (p < 0.001). Combining both signals in the cleaned dataset (descriptive), probation is 60.6% for students at 60% attendance or below without a scholarship (n = 66), against 10.5% for those above 90% with one.
- **Literature:** family background effects are weak or inconsistent in published work (Neupane & Gurung, 2021). Family income is also weak in our data (OLS p = 0.59; logistic p = 0.079), so we recommend targeting by attendance and scholarship status, not by income.
- **Industry expert feedback:** *[to be completed — Section 7]*
- **Decision for management:** use attendance and scholarship status together when ranking who to contact.

### S3 — Set the flagging threshold by cost and advisor capacity, not by a default 0.5
- **Statistical evidence (Bayesian decision rule, Task 8):** the break-even risk threshold falls as a missed student becomes costlier relative to a false alarm:

| Cost of a miss vs a false alarm | Flag students at risk ≥ |
|---|---|
| 1 : 1 | 0.500 |
| 2 : 1 | 0.333 |
| 5 : 1 | 0.167 |
| 10 : 1 | 0.091 |

  With the illustrative 5 : 1 cost matrix, doing nothing costs 1,490 units, flagging Red + Amber costs 828 (−44.4%), and the cost-optimal cutoff costs 756 (−49.3%) but flags 748 students (63%). Capacity therefore sets the real cutoff: the top 5% flagged gives 65% precision (lift 2.6), the top 10% gives 58.8% (lift 2.36), and the top 20% gives 48.1% precision with 38.6% recall.
- **Literature:** decision-theoretic thresholds are standard in Bayesian decision analysis (Task 8); the cost values themselves are not from literature.
- **Industry expert feedback:** *[to be completed — Section 7]*
- **Decision for management:** agree the cost ratio and advisor capacity with the department. The costs are assumptions, not data.

### S4 — Do not build policy on social-media use or study hours alone
- **Statistical evidence:** social media is weakly negative for CGPA (−0.017 points per hour, p = 0.005) but not significant for probation (OR 0.967, p = 0.31). Study hours are not significant in either model (CGPA p = 0.91; probation p = 0.43).
- **Literature:** Masalimova et al. (2023) report mixed effects across 22 reviewed studies.
- **Industry expert feedback:** *[to be completed — Section 7]*
- **Decision for management:** monitor these variables but do not target them until the pilot shows an effect.

---

## 3. Operational recommendations

### O1 — Run a one-cohort pilot as an RCBD
- **Design:** among model-flagged students, block by scholarship status (primary) and semester band (secondary, if the cohort allows). Randomise within each block to enhanced advisor consultation versus standard support. Measure next-semester CGPA and probation status from institutional records.
- **Statistical evidence:** scholarship and semester are the two strongest, already-quantified nuisance variables (Tasks 4–5), so blocking on them removes known variance. The Task 10 what-if (association only) suggests raising attendance by 10 points for Red and Amber students is linked to about 22 fewer expected probation cases (−11.6% of 496 flagged students). The pilot tests whether that holds.
- **Ethics and feasibility:** use a wait-list or active control so no at-risk student goes unsupported (Task 6, §6.6). Run a formal power analysis before launch and fix the success rule in advance.
- **Industry expert feedback:** *[to be completed — Section 7]*

### O2 — Export attendance weekly and act by tier
- Red (risk ≥ 40%; 181 students, 15.2%): advisor meeting within two weeks.
- Amber (25–40%; 315 students, 26.4%): attendance nudges, peer mentoring, monitoring.
- Green (< 25%; 698 students, 58.5%): routine support.
- **Statistical evidence:** actual probation rates of 52.5%, 31.1% and 15.0% across the three tiers show the tiers separate students.

### O3 — Give advisors reason codes and a what-if view
- Limit reason codes to levers an advisor can act on (for example "low attendance + no scholarship"). Treat the what-if tool as associational, not causal.

### O4 — Retrain every semester and monitor calibration
- **Statistical evidence:** AUC is only ≈ 0.70 and the Brier score is 0.167, so drift would quickly reduce usefulness. Recompute tiers and re-check calibration each semester.

### O5 — Start collecting a semester-by-semester panel now
- Record student, semester, attendance, SGPA, probation flag and scholarship each term.
- **Statistical evidence:** the file has only two time points per student (previous SGPA and current CGPA, r = 0.66), and "ever on probation" accumulates with semester (semester OR 1.228), which dominates the current score. A panel lets the model predict future probation and enables the forecasting in Task 9.

---

## 4. Risk management considerations

| Risk | Evidence | Mitigation |
|---|---|---|
| Moderate accuracy | AUC ≈ 0.70; recall 0.617, so about 38% of at-risk students are missed; precision 0.366 (test set: 37 true flags, 64 false alarms) | Use as a screening aid with human review, never an automatic decision |
| Advisor overload | Cost-optimal cutoff flags 63% of students | Use the three-tier design and a capacity slider |
| Semester dominates the score | "Ever on probation" accumulates over time | Collect the semester panel (O5) |
| Self-reported data | Survey recall and reporting bias | Move to registrar and attendance-system records for the pilot |
| Model assumptions | Residuals non-normal (Shapiro p < 0.001) and heteroscedastic (p = 0.001); CGPA capped at 4.0; OLS R² = 0.146 | Report robust or Beta/Tobit models as a next step; treat CGPA predictions as indicative |
| Rare suspension | 46 cases (3.9%); gender difference p = 0.089 with low power | Do not model suspension; keep probation as the target |
| Missing CGPA | 39 zero rows (3.3%) treated as missing | CGPA models use n = 1,155; probation models use n = 1,194 |
| Association read as cause | Cross-sectional survey (Task 6, §6.1) | Present results as associations; test causality in the RCBD pilot |

---

## 5. Ethical considerations

1. **Support, never punishment.** A high-risk score triggers support only. It must not be used for disciplinary action or to withdraw a scholarship (scholarship status is protective in our data).
2. **Fairness by design.** Gender and family income are excluded from the deployed model. Cross-validated AUC is unchanged (0.7059 vs 0.7063) and the Brier score is slightly better (0.1669 vs 0.1676).
3. **Audit for proxies each semester.** Scholarship and English proficiency remain in the model and may proxy for background. Report tier rates by gender, scholarship and English level every semester and investigate gaps.
4. **Privacy and consent.** Attendance and probation records are personal data: role-based access for advisors only, anonymised IDs in analysis, and a clear notice to students.
5. **Transparency.** Advisors share the reason codes with the student so the score can be discussed and challenged.
6. **Research ethics.** Use a wait-list or active control in the pilot (Task 6, §6.6).

---

## 6. Future research opportunities

1. **Longitudinal panel and forecasting** (trend, seasonal, ARIMA/SARIMA) once several terms of data exist (Task 9).
2. **Causal evidence** from the RCBD pilot, with a formal power analysis (Task 6).
3. **Quantile regression** to test whether the attendance effect holds across CGPA levels in this programme, as in Karnik et al. (2020) and Pani & Kishore (2016). Our models estimate average effects only.
4. **Bounded-outcome models** (Beta or Tobit) for CGPA, given the cap at 4.0 and heteroscedastic residuals.
5. **Bayesian models with informed priors** once pilot data exist (Task 8).
6. **External and temporal validation** on another cohort or programme, with recalibration (Task 10).
7. **Richer behavioural data**, such as learning-platform activity, if the department can supply it.

---

## 7. Industry expert feedback — to complete (Task 11)

| Recommendation | Expert's comment | How it changed the recommendation |
|---|---|---|
| S1 Attendance-led programme | *[add]* | *[add]* |
| S2 Prioritise low attendance without a scholarship | *[add]* | *[add]* |
| S3 Cost- and capacity-based threshold | *[add]* | *[add]* |
| S4 No policy on social media or study hours | *[add]* | *[add]* |
| O1 RCBD pilot | *[add]* | *[add]* |
| O2–O5 Operations | *[add]* | *[add]* |
| Risk and ethics (Sections 4–5) | *[add]* | *[add]* |

Record the expert's name, role, date, method (email or meeting) and attach evidence. **Do not use the two WhatsApp images in the project folder:** they are emails about a different project (customer repurchase prediction, different student IDs), not feedback on this one.

---

## 8. Decisions we ask management to take

1. Approve a one-cohort RCBD pilot of the early-warning system (O1).
2. Approve a weekly attendance export and a semester-by-semester data record (O2, O5).
3. Agree the cost ratio and advisor capacity that set the flagging threshold (S3), and confirm the support-not-punishment rule (Section 5).

---

## References
- Karnik, A., Kishore, P., & Meraj, M. (2020). Examining the linkage between class attendance at university and academic performance in an international branch campus setting. *Research in Comparative and International Education, 15*(4), 371–390.
- Marburger, D. R. (2006). Does mandatory attendance improve student performance? *The Journal of Economic Education, 37*(2), 148–155.
- Masalimova, A. R., et al. (2023). Effects of social networking sites on university students' academic performance: A systematic review. *Online Journal of Communication and Media Technologies, 13*(3).
- Neupane, D., & Gurung, S. K. (2021). Influence of parental socio-economic status on students' academic performance. *International Journal of Multidisciplinary Perspectives in Higher Education, 6*(2), 50–67.
- Pani, P. K., & Kishore, P. (2016). Absenteeism and performance in a quantitative module: A quantile regression analysis. *Journal of Applied Research in Higher Education, 8*(3), 376–389.
- Stanca, L. (2006). The effects of attendance on academic performance: Panel data evidence for introductory microeconomics. *The Journal of Economic Education, 37*(3), 251–266.

*Add one or two learning-analytics fairness or early-warning papers from your own search to support Section 5 (verify each before citing).*
