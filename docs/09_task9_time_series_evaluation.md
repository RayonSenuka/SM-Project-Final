# Task 9 — Time Series Analysis (Discussion Only)

Per the assignment brief, no time series analysis is required or performed
here — `Students_Performance_dataset.csv` is a **single cross-sectional
survey wave** (each student appears once, at one point in time), so there
is no repeated, time-stamped outcome to analyze. This section instead
critically discusses how time series methods *could* apply if the
department collected longitudinal data, using this project's own findings
to motivate exactly what such data would need to contain and what it
would be used for.

## 9.1 Why the current dataset cannot support time series analysis

Time series methods require **repeated observations of the same unit
over ordered time points** (e.g. one student's GPA each semester). This
dataset has, for each student, only a single row with `current_cgpa`
(most recent semester) and `previous_sgpa` (one semester prior) — **two
time points**, and only two. That is enough to compute a single
difference or correlation (and indeed the Task 3 correlation heatmap
already shows `previous_sgpa` and `current_cgpa` correlate at r = 0.66,
by far the strongest relationship in the dataset — see
`outputs/figures/correlation_heatmap.png`), but it is not a time series:
proper trend/seasonality/forecasting analysis needs a run of **at least
several consecutive, evenly spaced observations per unit** (a bare
minimum of ~8–10 points for basic decomposition, and considerably more —
commonly recommended as 50+ observations — for a well-specified ARIMA
model). A two-point panel can only ever say "did the outcome go up or
down," never characterize a trend shape, a seasonal cycle, or generate a
calibrated multi-period forecast.

That said, the r = 0.66 correlation between the two available waves is
itself informative: it shows meaningful **persistence** in academic
performance from one semester to the next (consistent with Task 5.1.2's
finding that `previous_sgpa` is by far the strongest single predictor of
`current_cgpa`, raising R² from 0.146 to 0.605). This is exactly the kind
of autocorrelation structure that a full time series of semester GPAs
would let the department model and forecast properly, rather than glimpse
through a single lagged snapshot.

## 9.2 What longitudinal data would need to look like

To apply time series methods to this problem, the university would need
to restructure data collection from a one-off survey into a **panel /
repeated-measures dataset**: for each student, one row per semester (or
per term), covering their full enrolment history, e.g.:

`student_id, semester_number, calendar_term (e.g. 2023-Fall), attendance_%, study_hours, sgpa, cumulative_cgpa, probation_flag, scholarship_flag, ...`

This is a realistic extension, not a hypothetical one — the raw dataset
already contains `admission_year` and `current semester`, meaning the
institution evidently *could* track each student across semesters if it
retained each term's snapshot instead of only the latest one.

## 9.3 Trend analysis

With a per-student semester-by-semester CGPA series, **trend analysis**
would let the department distinguish a student on a genuinely
**declining trajectory** (a much stronger early-warning signal than a
single low CGPA snapshot) from one who is stable or recovering. This
directly extends Task 5.2's logistic/GLM probation model, which currently
uses only static, single-snapshot predictors (attendance, credits
completed, etc.) — a *slope* term (e.g. GPA change over the last two
semesters) would very plausibly outperform any single-wave predictor
currently in the model, since deterioration is intuitively more
predictive of an upcoming probation event than an absolute level.
Aggregated across the whole cohort, a **department-level trend** (mean
CGPA by cohort-semester) would also let administrators see whether
academic performance is drifting over calendar time, distinct from any
individual student's trajectory.

## 9.4 Seasonal analysis

Two plausible seasonal/cyclical patterns are worth naming for this
context:

- **Within-academic-year cyclicality**: performance, attendance, and
  workload plausibly differ systematically between a program's regular
  semesters and any shortened/summer term, or between early- and
  late-semester weeks (e.g. attendance often drops before finals). A
  single cross-sectional `Average attendance on class (%)` figure, as
  used in Task 3–5, cannot separate "this student is consistently
  disengaged" from "this snapshot happened to land in a low-attendance
  week" — a longitudinal, within-semester series would.
- **Cohort/intake seasonality**: if the university admits students in
  more than one intake per year (common in South Asian universities, per
  the `admission_year` field's format), outcomes may differ systematically
  by intake cohort — a pattern a classical seasonal decomposition (or
  simply cohort as a categorical control) would reveal, but which the
  current single-wave dataset cannot separate from a genuine calendar-time
  trend.

## 9.5 Forecasting and ARIMA modelling

Once several consecutive terms of the panel above exist, standard
forecasting approaches become applicable:

- **Per-student CGPA forecasting** (e.g. a simple exponential-smoothing or
  low-order ARIMA model per student, or a pooled panel model) to predict
  *next*-semester CGPA before it happens — the genuinely forward-looking
  version of Task 5's regression model, which currently only explains the
  *current* semester's CGPA using contemporaneous predictors.
- **Cohort/department-level forecasting**: an ARIMA (or seasonal SARIMA,
  if the seasonality in §9.4 is confirmed) model of aggregate quantities
  such as term-over-term probation rate or mean CGPA, useful for capacity
  planning (advising staff, scholarship budget) rather than individual
  student intervention.
- **Practical caveat**: ARIMA and similar methods need the series to be
  (or be made, via differencing) **stationary**, and enough historical
  terms to estimate autoregressive/moving-average orders reliably — a
  newly-started longitudinal collection effort would likely need several
  years of terms before ARIMA-class methods become reliable, so a
  university adopting this recommendation should expect a multi-year
  lead time before forecasting quality matures, not an immediate result.

## 9.6 Business applications and expected organizational benefits

Directly building on Task 10's proposed early-warning system, a
time-series-enabled version would let the department move from
**reactive** to **predictive** intervention:

- Flag students on a *declining trend* one or two semesters **before**
  they cross into probation, rather than only after the fact (the current
  dataset's `probation` field is itself a lagging indicator — it records
  that probation already happened).
- Forecast term-ahead aggregate probation/scholarship-risk rates to help
  the scholarship committee (a Task 1 stakeholder) budget proactively
  rather than reactively.
- Detect a *within-student* attendance or engagement decline in real time
  (if attendance were logged more frequently than once per survey), which
  the literature on learning-analytics early-warning systems consistently
  identifies as more actionable than single end-of-term summaries, since
  it leaves time to intervene within the same term rather than only after
  final grades are recorded.

## 9.7 Conclusion for this consultancy

Time series methods are **not applicable to the dataset as collected**
(a single cross-sectional wave with, at best, one lagged data point), and
no time series model is fitted in this project, consistent with the
assignment brief. However, the strong `previous_sgpa`–`current_cgpa`
correlation already observed (r = 0.66, and the dominant predictor in
Task 5.1.2's sensitivity check) is a concrete signal that performance is
serially correlated across semesters — exactly the structure a proper
longitudinal panel would let the department model with trend analysis,
seasonal decomposition, and short-horizon ARIMA-class forecasting.
**Recommendation:** the department should begin retaining each semester's
snapshot as a permanent panel (it already collects a compatible one-off
version of this data each term) so that, within a few academic years, the
early-warning system proposed in Task 10 can be upgraded from a
single-snapshot classifier to a genuinely predictive, trend-aware model.
