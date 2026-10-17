"""
eda.py
//Task 3: Exploratory Data Analysis(Data Understanding & Descriptive Analysis)

Basic descriptive analysis for Task 3: missingness, descriptive statistics,
and headline visualizations. Run clean_data.py first.

Run:
    python src/eda.py

Output:
    Printed stats in the terminal, plus PNG plots in outputs/figures/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE = Path(__file__).resolve().parent.parent
CLEAN_PATH = BASE / "data" / "Students_Performance_dataset_clean.csv"
FIG_DIR = BASE / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = [
    "age",
    "study_hours",
    "study_sessions",
    "social_media_hours",
    "skill_dev_hours",
    "attendance",
    "previous_sgpa",
    "current_cgpa",
    "credits_completed",
    "family_income",
]


def main():
    df = pd.read_csv(CLEAN_PATH)

    print("=" * 60)
    print("SHAPE:", df.shape)

    print("\nMISSING VALUES PER COLUMN")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\nDESCRIPTIVE STATISTICS (numeric columns)")
    print(df[NUMERIC_COLS].describe().T)

    print("\nCATEGORICAL FREQUENCIES (key columns)")
    for col in ["gender", "scholarship", "probation", "suspension", "learning_mode"]:
        print(f"\n-- {col} --")
        print(df[col].value_counts())

    sns.set_theme(style="whitegrid")

    # Histogram of CGPA
    plt.figure(figsize=(7, 5))
    sns.histplot(df["current_cgpa"].dropna(), bins=30, kde=True)
    plt.title("Distribution of Current CGPA")
    plt.xlabel("CGPA")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cgpa_histogram.png", dpi=150)
    plt.close()

    # Attendance vs CGPA scatter
    plt.figure(figsize=(7, 5))
    sns.regplot(data=df, x="attendance", y="current_cgpa", scatter_kws={"alpha": 0.3})
    plt.title("Attendance vs. Current CGPA")
    plt.xlabel("Average attendance (%)")
    plt.ylabel("CGPA")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "attendance_vs_cgpa.png", dpi=150)
    plt.close()

    # CGPA by probation status
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x="probation", y="current_cgpa")
    plt.title("CGPA by Probation History")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cgpa_by_probation.png", dpi=150)
    plt.close()

    # CGPA by scholarship status
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x="scholarship", y="current_cgpa")
    plt.title("CGPA by Scholarship Status")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cgpa_by_scholarship.png", dpi=150)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(9, 7))
    corr = df[NUMERIC_COLS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (numeric variables)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"\nPlots saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
