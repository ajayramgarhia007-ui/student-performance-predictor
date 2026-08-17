"""
generate_dataset.py
--------------------
Generates a realistic synthetic dataset of student academic records.
Run this first to create data/student_data.csv, which is then used by
train_model.py to train the prediction models.

Why synthetic data?
College courses often can't share real student records (privacy),
so this script builds a statistically realistic dataset using sensible
relationships between features and the final score (e.g. more study
hours + higher attendance -> higher score, plus random noise so the
problem is realistic and not "too easy" for the model).
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 1200  # number of students

# ---- Feature generation -----------------------------------------------
study_hours_per_week = np.clip(np.random.normal(12, 5, N), 0, 40)
attendance_percent = np.clip(np.random.normal(80, 12, N), 30, 100)
previous_score = np.clip(np.random.normal(65, 15, N), 0, 100)
sleep_hours = np.clip(np.random.normal(6.7, 1.3, N), 3, 10)
parental_education = np.random.choice(
    ["High School", "Bachelors", "Masters", "PhD"],
    size=N, p=[0.35, 0.35, 0.22, 0.08]
)
internet_access = np.random.choice(["Yes", "No"], size=N, p=[0.85, 0.15])
extracurricular = np.random.choice(["Yes", "No"], size=N, p=[0.45, 0.55])
part_time_job = np.random.choice(["Yes", "No"], size=N, p=[0.3, 0.7])
study_method = np.random.choice(
    ["Self-study", "Group Study", "Tutoring", "Online Courses"], size=N
)
absences = np.clip(np.round(np.random.poisson(4, N) + (100 - attendance_percent) / 10), 0, 40)

# Encode ordinal/binary effects for score simulation
parent_edu_score_boost = pd.Series(parental_education).map(
    {"High School": 0, "Bachelors": 3, "Masters": 5, "PhD": 7}
).values
internet_boost = np.where(internet_access == "Yes", 3, -2)
job_penalty = np.where(part_time_job == "Yes", -4, 0)
extracurricular_boost = np.where(extracurricular == "Yes", 1.5, 0)

# ---- Simulate final exam score (0-100) with realistic weighted formula ----
noise = np.random.normal(0, 6, N)

final_score = (
    0.9 * study_hours_per_week
    + 0.35 * attendance_percent
    + 0.30 * previous_score
    + 1.2 * sleep_hours
    + parent_edu_score_boost
    + internet_boost
    + job_penalty
    + extracurricular_boost
    - 0.4 * absences
    + noise
)

# Rescale to a believable 0-100 exam score
final_score = 15 + (final_score - final_score.min()) / (final_score.max() - final_score.min()) * 85
final_score = np.clip(final_score, 0, 100).round(2)

# Pass/Fail label (>=40 is pass, typical academic threshold)
pass_fail = np.where(final_score >= 40, "Pass", "Fail")

# Performance category for a classification variant
def categorize(score):
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Average"
    elif score >= 40:
        return "Below Average"
    else:
        return "Poor"

performance_category = [categorize(s) for s in final_score]

df = pd.DataFrame({
    "study_hours_per_week": study_hours_per_week.round(1),
    "attendance_percent": attendance_percent.round(1),
    "previous_score": previous_score.round(1),
    "sleep_hours": sleep_hours.round(1),
    "absences": absences.astype(int),
    "parental_education": parental_education,
    "internet_access": internet_access,
    "extracurricular_activities": extracurricular,
    "part_time_job": part_time_job,
    "study_method": study_method,
    "final_score": final_score,
    "pass_fail": pass_fail,
    "performance_category": performance_category,
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/student_data.csv", index=False)

print(f"Dataset created: data/student_data.csv  ({df.shape[0]} rows, {df.shape[1]} columns)")
print(df.head())
print("\nPass rate:", (df.pass_fail == "Pass").mean().round(3))
