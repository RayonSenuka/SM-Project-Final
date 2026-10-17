"""
task10_framework_diagram.py
Draws the conceptual framework required by Task 10 ("Include a conceptual framework
illustrating the proposed solution"). Output: outputs/figures/task10_conceptual_framework.png
Run: python src/task10_framework_diagram.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs" / "figures" / "task10_conceptual_framework.png"

fig, ax = plt.subplots(figsize=(14, 6.2))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6.2)
ax.axis("off")

NAVY, TEAL, GREY = "#1F4E79", "#2A7F8E", "#F2F4F7"


def box(x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=GREY, ec=color, lw=2))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.55), w, 0.55, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=color, ec=color))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", color="white",
            fontsize=11, fontweight="bold")
    ax.text(x + 0.15, y + h - 0.75, "\n".join(lines), ha="left", va="top", fontsize=9, linespacing=1.5)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, lw=2, color="#444"))


ax.text(7, 5.95, "Academic Risk Early-Warning System (EWS): Conceptual Framework",
        ha="center", fontsize=14, fontweight="bold", color=NAVY)

box(0.2, 2.3, 2.9, 3.0, "1. DATA", ["Attendance records", "Semester / credits", "Study & social-media habits",
                                     "English proficiency", "Scholarship status", "Consultation & mode",
                                     "(gender & income excluded)"], NAVY)
box(3.5, 2.3, 2.9, 3.0, "2. ANALYTICS", ["Cleaning & validation", "Logistic regression (GLM)",
                                          "5-fold cross-validation", "Calibrated risk probability",
                                          "Tiering: Green / Amber / Red", "Reason codes (actionable)"], TEAL)
box(6.8, 2.3, 2.9, 3.0, "3. DASHBOARD", ["Department overview", "Advisor worklist",
                                          "What-if calculator", "Policy sliders (cutoffs,",
                                          "capacity, cost ratio)"], NAVY)
box(10.1, 2.3, 3.7, 3.0, "4. INTERVENTION", ["Red: advisor meeting < 2 weeks", "Amber: attendance nudges,",
                                              "  peer mentoring, monitoring", "Green: routine support",
                                              "Scholarship / financial support", "  review (never punitive)"], TEAL)
for x1, x2 in [(3.1, 3.5), (6.4, 6.8), (9.7, 10.1)]:
    arrow(x1, 3.8, x2, 3.8)

# feedback loop
ax.add_patch(FancyArrowPatch((11.9, 2.25), (1.6, 2.25), connectionstyle="arc3,rad=-0.07",
                             arrowstyle="-|>", mutation_scale=18, lw=2, color="#C0392B", linestyle="--"))
ax.text(6.9, 0.75, "5. FEEDBACK LOOP: record outcomes, retrain each semester, audit fairness, "
        "collect longitudinal data (enables Task 9 time-series)", ha="center", fontsize=10,
        color="#C0392B", fontweight="bold")
ax.text(7, 0.25, "Governance: human-in-the-loop decisions | student data privacy | transparency to students",
        ha="center", fontsize=9, color="#555")
plt.tight_layout()
plt.savefig(OUT, dpi=200)
print("Saved", OUT)
