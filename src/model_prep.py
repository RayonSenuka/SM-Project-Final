"""
model_prep.py

Shared data preparation for Task 4 (statistical inference) and Task 5
(predictive modelling). Builds the curated predictor set justified in
docs/01_task1_industry_problem.md (attendance, study habits, social media,
transportation, English proficiency, scholarship, plus academic-progress
and demographic controls).

Predictors deliberately EXCLUDED, with reasons:
- previous_sgpa / current_cgpa as predictors of `probation`: temporal/
  circularity risk (probation and CGPA are two views of the same academic
  outcome) -> excluded from the probation model, used only as the target
  (current_cgpa) or as a separate sensitivity check for the CGPA model.
- program: constant (all "BCSE") -> no variance, cannot be used.
- skills, interested_area: free-text survey fields with very high
  cardinality -> not usable as categorical predictors without heavy NLP
  preprocessing, out of scope here.
- admission_year, hsc_year: near-duplicates of age/semester -> redundant,
  dropped to reduce multicollinearity.
- has_smartphone, has_pc, physical_disability: extremely low variance
  (e.g. has_smartphone: 1178 Yes / 16 No) -> unlikely to yield stable
  estimates, dropped for parsimony.
- living_with, relationship_status, health_issues: kept out of the main
  model for parsimony (already 15 predictors); noted as a limitation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CLEAN_PATH = BASE / "data" / "Students_Performance_dataset_clean.csv"

ENGLISH_MAP = {"Basic": 0, "Intermediate": 1, "Advance": 2}

NUMERIC_PREDICTORS = [
    "attendance",
    "study_hours",
    "study_sessions",
    "social_media_hours",
    "skill_dev_hours",
    "semester",
    "credits_completed",
    "log_family_income",
    "english_proficiency_ord",
]

BINARY_PREDICTORS = [
    "scholarship_yes",
    "uses_transport_yes",
    "learning_mode_online",
    "teacher_consultancy_yes",
    "co_curricular_yes",
    "gender_male",
]

ALL_PREDICTORS = NUMERIC_PREDICTORS + BINARY_PREDICTORS  # 15 predictors


def load_raw():
    return pd.read_csv(CLEAN_PATH)


def build_model_frame():
    """Returns the cleaned dataframe with engineered columns added."""
    df = load_raw()

    df["log_family_income"] = np.log1p(df["family_income"])
    df["english_proficiency_ord"] = df["english_proficiency"].map(ENGLISH_MAP)

    df["scholarship_yes"] = (df["scholarship"] == "Yes").astype(int)
    df["uses_transport_yes"] = (df["uses_transport"] == "Yes").astype(int)
    df["learning_mode_online"] = (df["learning_mode"] == "Online").astype(int)
    df["teacher_consultancy_yes"] = (df["teacher_consultancy"] == "Yes").astype(int)
    df["co_curricular_yes"] = (df["co_curricular"] == "Yes").astype(int)
    df["gender_male"] = (df["gender"] == "Male").astype(int)
    df["probation_bin"] = (df["probation"] == "Yes").astype(int)
    df["suspension_bin"] = (df["suspension"] == "Yes").astype(int)

    return df


def get_regression_data():
    """X, y for the CGPA regression task (drops rows with missing CGPA)."""
    df = build_model_frame()
    df = df.dropna(subset=["current_cgpa"] + ALL_PREDICTORS).reset_index(drop=True)
    X = df[ALL_PREDICTORS].astype(float).values
    y = df["current_cgpa"].astype(float).values
    return X, y, df


def get_classification_data():
    """X, y for the probation classification task (no missing target)."""
    df = build_model_frame()
    df = df.dropna(subset=["probation_bin"] + ALL_PREDICTORS).reset_index(drop=True)
    X = df[ALL_PREDICTORS].astype(float).values
    y = df["probation_bin"].astype(int).values
    return X, y, df
