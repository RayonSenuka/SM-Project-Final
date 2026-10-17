# Task 1 — Understanding the Industry Problem

**Working title:** Academic Risk Prediction & Early Intervention for a University CSE Department

## Industry background
Higher education institutions in South Asia (and globally) face persistent student attrition and academic probation/suspension rates, often addressed reactively — after a student's CGPA has already dropped — rather than through early identification.

## Organizational context
The client is a university's BCSE (Computer Science & Engineering) department, which currently has no data-driven system to flag at-risk students before they enter academic probation. Interventions (counseling, tutoring, scholarship review) happen only after poor results are already recorded.

## Description of the business problem
The department cannot currently predict, using data it already collects (attendance, study habits, socioeconomic background), which students are likely to fall into academic probation or suspension. As a result, support resources are allocated reactively rather than proactively, and some at-risk students go unnoticed until it's too late to intervene meaningfully.

## Importance of solving the problem
Probation/suspension events are costly for both the student (delayed graduation, lost scholarships, mental health strain) and the institution (lower retention/graduation rates, reputational and accreditation impact, wasted teaching capacity on students who eventually drop out).

## Stakeholders affected
- Students and families
- Academic advisors/counselors
- Department administration (retention metrics)
- Scholarship committees (`meritorious scholarship` status is in the data)
- Teaching staff, who currently have no systematic way to flag struggling students

## Business decisions influenced
- Which students to proactively refer to teacher consultation/counseling?
- How to allocate scholarship and financial support (`family income` is in the data)
- Whether to adjust attendance policy enforcement
- How to design early-semester interventions using CGPA/SGPA trend data

## Consultancy objectives
1. Quantify how strongly attendance and behavioral factors (study hours, social media use, transportation, English proficiency) predict probation/suspension risk.
2. Build a predictive model (logistic regression/GLM) the department could realistically use as an early-warning flag.
3. Recommend which controllable factors the university should intervene on first, based on statistical effect size rather than intuition.

## Alternate framings considered (not chosen, kept for reference)
- **Performance/CGPA prediction for resource allocation** — CGPA as a
  continuous target, framed around tutoring budget allocation
- **Attendance behavior analysis** — flip the direction and ask what drives
  poor attendance (transportation access, social media use, scholarship status), framed as an engagement/operations problem

## Note for Task 12 (ethics)
`family income` should not be used in a way that penalizes lower-income students — frame recommendations around support, not exclusion.
