# Task 3 — Dataset Understanding and Descriptive Analysis

## Dataset description
`Students_Performance_dataset.csv` — 1,194 rows, 31 columns. 
Appears to be a real, self-administered survey from a South Asian university's BCSE
(Computer Science & Engineering) program (terminology: H.S.C., CGPA/SGPA,
"meritorious scholarship"). Not synthetic. No duplicate rows found.

Run `python src/eda.py` (after `clean_data.py`) to regenerate the numbers
below from the actual file and produce the plots referenced.

## Variable groups

**Numeric**
- Age (18–27)
- University Admission year (2013–2023) / H.S.C. passing year
- Current Semester
- Hours studied daily (0–13)
- Study sessions per day
- Hours spent daily on social media (0–20)
- Hours spent daily on skill development
- Average attendance on class (%) — *stored as text in raw file, see Data Quality below*
- Previous SGPA (0.00–4.00)
- Current CGPA (0.00–4.00) — primary performance target
- Credits completed
- Monthly family income (4,000–2,000,000)

**Categorical (ordinal)**
- English language proficiency (Basic/Intermediate/...)

**Categorical (nominal/binary)**
- Gender (Male: 672, Female: 522)
- Program (constant: "BCSE" for all 1,194 rows — **not usable as a grouping variable**, despite being present in the file)
- Meritorious scholarship (Yes: 523, No: 671)
- University transportation use
- Preferred learning mode (Online/Offline)
- Smartphone ownership, personal computer ownership
- Academic probation ever (Yes: 298, No: 896)
- Suspension ever (Yes/No)
- Teacher consultancy attendance
- Skills held, interested area
- Relationship status
- Co-curricular engagement
- Living situation
- Health issues, physical disabilities

## Data quality assessment

1. **`Average attendance on class` is stored as text, not numeric**, because one row contains the value `"94-98"` (a range) instead of a    single number. This forces the whole column to be read as a string. 
   **Fix:**
   replace with the midpoint (96) or drop/flag the row — implemented in `src/clean_data.py`.

2. **39 rows have `current CGPA == 0` and the same rows have `previous SGPA== 0`.** A true 0.00 CGPA is implausible for an enrolled student this almost certainly represents unreported/missing data coded as zero rather than genuine academic failure. 
  **Fix:** 
  treat these as missing (NaN), not as real zeros — implemented in `src/clean_data.py`. Including them as-is would badly distort any regression on CGPA.

3. **`Program` has only one value** across all 1,194 rows — exclude it from any ANOVA/grouping analysis despite its presence in the raw file.

4. **No missing values in any other column** (1 row has a missing value in "What are the skills do you have?" — negligible, single row).

5. No duplicate rows.

## Outlier detection (to run/confirm via `eda.py`)
- Check `hours studied daily` and `social media hours` for implausible extreme values (13 and 20 respectively are the observed maxima — plausible but worth boxplotting).

- `Monthly family income` spans 4,000 to 2,000,000 — a very wide range; consider a log transform before use in regression, and boxplot to check for a small number of high-income outliers skewing the mean.

## Descriptive statistics (from initial inspection — regenerate full table
via `eda.py`)

| Variable | Mean | Std | Min | Max |
|---|---|---|---|---|
| Current CGPA | 3.17 | 0.75 | 0.00* | 4.00 |
| Previous SGPA | 2.68 | 0.87 | 0.00* | 4.00 |
| Age | — | — | 18 | 27 |
| Hours studied daily | — | — | 0 | 13 |
| Social media hours daily | — | — | 0 | 20 |
| Monthly family income | — | — | 4,000 | 2,000,000 |

\* Zero values here are data-quality artifacts (see above), not genuine re-run stats after cleaning.

## Visualizations to generate (`eda.py` produces these in `outputs/figures/`)
- Histogram of CGPA
- Scatter of attendance vs. CGPA with trend line (headline chart)
- Boxplots of CGPA by: scholarship status, probation status, gender,
  learning mode
- Correlation heatmap across numeric variables (sets up Task 7 PCA
  justification)
- Bar chart of probation/suspension rates by scholarship status

## Initial business insights (provisional — confirm after running eda.py)
Attendance and study hours are expected to show a positive association with CGPA; probation history is expected to correlate with lower current CGPA and possibly with lower attendance. These are hypotheses to confirm with actual numbers, not asserted findings — do not finalize this section's wording
until `eda.py` has been run.
