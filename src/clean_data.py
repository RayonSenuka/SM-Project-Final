"""
clean_data.py

Cleans Students_Performance_dataset.csv:
1. Fixes the 'Average attendance on class' column (one row contains a range
   like "94-98" instead of a single number; replaced with the midpoint).
2. Treats CGPA == 0 and SGPA == 0 as missing data (NaN), not genuine zeros —
   see docs/03_task3_descriptive_analysis.md for the reasoning.
3. Renames columns to short, snake_case names for easier downstream analysis
   (original survey-question text is kept in COLUMN_MAP for reference).

Run:
    python src/clean_data.py

Output:
    data/Students_Performance_dataset_clean.csv
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "Students_Performance_dataset.csv"
CLEAN_PATH = RAW_PATH.parent / "Students_Performance_dataset_clean.csv"

# Map original survey-question column names to short working names.
# Edit this if your CSV's exact headers differ slightly.
COLUMN_MAP = {
    "University Admission year": "admission_year",
    "Gender": "gender",
    "Age": "age",
    "H.S.C passing year": "hsc_year",
    "Program": "program",
    "Current Semester": "semester",
    "Do you have meritorious scholarship ?": "scholarship",
    "Do you use University transportation?": "uses_transport",
    "How many hour do you study daily?": "study_hours",
    "How many times do you seat for study in a day?": "study_sessions",
    "What is your preferable learning mode?": "learning_mode",
    "Do you use smart phone?": "has_smartphone",
    "Do you have personal Computer?": "has_pc",
    "How many hour do you spent daily in social media?": "social_media_hours",
    "Status of your English language proficiency": "english_proficiency",
    "Average attendance on class": "attendance",
    "Did you ever fall in probation?": "probation",
    "Did you ever got suspension?": "suspension",
    "Do you attend in teacher consultancy for any kind of academical problems?": "teacher_consultancy",
    "What are the skills do you have ?": "skills",
    "How many hour do you spent daily on your skill development?": "skill_dev_hours",
    "What is you interested area?": "interested_area",
    "What is your relationship status?": "relationship_status",
    "Are you engaged with any co-curriculum activities?": "co_curricular",
    "With whom you are living with?": "living_with",
    "Do you have any health issues?": "health_issues",
    "What was your previous SGPA?": "previous_sgpa",
    "Do you have any physical disabilities?": "physical_disability",
    "What is your current CGPA?": "current_cgpa",
    "How many Credit did you have completed?": "credits_completed",
    "What is your monthly family income?": "family_income",
}


def parse_attendance(value):
    """Convert attendance values to a float, handling range strings like '94-98'."""
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    match = re.match(r"^(\d+)\s*-\s*(\d+)$", value)
    if match:
        low, high = int(match.group(1)), int(match.group(2))
        return (low + high) / 2
    try:
        return float(value)
    except ValueError:
        return np.nan


def main():
    df = pd.read_csv(RAW_PATH)

    n_before = len(df)

    # 1. Rename columns
    df = df.rename(columns=COLUMN_MAP)

    # 2. Fix attendance column
    df["attendance"] = df["attendance"].apply(parse_attendance)

    # 3. Treat CGPA/SGPA == 0 as missing
    zero_cgpa = (df["current_cgpa"] == 0).sum()
    zero_sgpa = (df["previous_sgpa"] == 0).sum()
    df.loc[df["current_cgpa"] == 0, "current_cgpa"] = np.nan
    df.loc[df["previous_sgpa"] == 0, "previous_sgpa"] = np.nan

    df.to_csv(CLEAN_PATH, index=False)

    print(f"Rows in:  {n_before}")
    print(f"Rows out: {len(df)}")
    print(f"Attendance values now numeric: {df['attendance'].notna().sum()} / {len(df)}")
    print(f"CGPA zeros converted to NaN:   {zero_cgpa}")
    print(f"SGPA zeros converted to NaN:   {zero_sgpa}")
    print(f"Saved cleaned file to: {CLEAN_PATH}")


if __name__ == "__main__":
    main()
