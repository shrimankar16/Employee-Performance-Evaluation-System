"""
Employee Performance Dataset Generator
Based on HR Analytics best practices + IBM HR Analytics dataset structure
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

np.random.seed(42)
N = 2000

# ── Departments & Roles ─────────────────────────────────────────────────────
departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations", "Support"]
roles = {
    "Engineering": ["Junior Dev", "Senior Dev", "Tech Lead", "Architect"],
    "Sales":        ["SDR", "Account Executive", "Sales Manager", "VP Sales"],
    "Marketing":    ["Content Writer", "Digital Marketer", "Brand Manager", "CMO"],
    "HR":           ["HR Associate", "HR Manager", "HRBP", "HR Director"],
    "Finance":      ["Analyst", "Senior Analyst", "Finance Manager", "CFO"],
    "Operations":   ["Ops Analyst", "Ops Manager", "Sr. Ops Manager", "COO"],
    "Support":      ["Support Agent", "Sr. Support", "Team Lead", "Support Manager"],
}
education = ["High School", "Bachelor's", "Master's", "PhD"]
gender     = ["Male", "Female", "Non-binary"]

dept_arr = np.random.choice(departments, N)

data = {
    "EmployeeID":         [f"EMP{str(i).zfill(4)}" for i in range(1, N+1)],
    "Age":                np.random.randint(22, 60, N),
    "Gender":             np.random.choice(gender, N, p=[0.48, 0.48, 0.04]),
    "Education":          np.random.choice(education, N, p=[0.10, 0.50, 0.30, 0.10]),
    "Department":         dept_arr,
    "JobRole":            [np.random.choice(roles[d]) for d in dept_arr],
    "YearsAtCompany":     np.random.randint(0, 25, N),
    "YearsInCurrentRole": np.random.randint(0, 12, N),
    "YearsSinceLastPromotion": np.random.randint(0, 10, N),
    "MonthlyIncome":      np.random.randint(25000, 200000, N),
    "JobLevel":           np.random.randint(1, 6, N),
    "NumCompaniesWorked": np.random.randint(0, 10, N),
    "TrainingHoursLastYear": np.random.randint(0, 80, N),
    "WorkLifeBalance":    np.random.randint(1, 5, N),
    "JobSatisfaction":    np.random.randint(1, 5, N),
    "RelationshipSatisfaction": np.random.randint(1, 5, N),
    "EnvironmentSatisfaction":  np.random.randint(1, 5, N),
    "OverTime":           np.random.choice(["Yes", "No"], N, p=[0.30, 0.70]),
    "BusinessTravel":     np.random.choice(["No Travel", "Rarely", "Frequently"], N, p=[0.30, 0.50, 0.20]),
    "Absences":           np.random.randint(0, 20, N),
    "TaskCompletionRate": np.round(np.random.uniform(50, 100, N), 1),
    "PeerRatingScore":    np.round(np.random.uniform(1, 5, N), 2),
    "ManagerRatingScore": np.round(np.random.uniform(1, 5, N), 2),
    "ProjectsCompleted":  np.random.randint(1, 20, N),
    "GoalAchievementPct": np.round(np.random.uniform(40, 100, N), 1),
}

df = pd.DataFrame(data)

# ── Fix logical inconsistencies ──────────────────────────────────────────────
df["YearsInCurrentRole"] = np.minimum(df["YearsInCurrentRole"], df["YearsAtCompany"])
df["YearsSinceLastPromotion"] = np.minimum(df["YearsSinceLastPromotion"], df["YearsAtCompany"])

# ── Derived performance score (ground truth) ─────────────────────────────────
score = (
    df["TaskCompletionRate"] * 0.25
    + df["GoalAchievementPct"] * 0.25
    + df["ManagerRatingScore"] * 10 * 0.20
    + df["PeerRatingScore"] * 10 * 0.15
    + (df["TrainingHoursLastYear"] / 80) * 100 * 0.08
    + df["ProjectsCompleted"] * 2 * 0.05
    - df["Absences"] * 1.5
    + (df["WorkLifeBalance"] - 1) * 2
    + (df["JobSatisfaction"] - 1) * 2
)

# Add noise
score += np.random.normal(0, 5, N)
score = np.clip(score, 0, 100)

# ── Rating categories ─────────────────────────────────────────────────────────
def score_to_rating(s):
    if s >= 85: return "Outstanding"
    elif s >= 70: return "Exceeds Expectations"
    elif s >= 55: return "Meets Expectations"
    elif s >= 40: return "Below Expectations"
    else: return "Needs Improvement"

df["PerformanceScore"] = np.round(score, 2)
df["PerformanceRating"] = df["PerformanceScore"].apply(score_to_rating)

# ── Save ─────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "data", "employee_performance.csv")
df.to_csv(out, index=False)
print(f"✅ Dataset saved → {out}")
print(f"   Shape: {df.shape}")
print(f"\nRating distribution:\n{df['PerformanceRating'].value_counts()}")
print(f"\nSample:\n{df.head(3)}")
