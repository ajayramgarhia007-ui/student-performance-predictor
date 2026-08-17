"""
eda.py
------
Exploratory Data Analysis for the Student Performance dataset.
Produces charts saved to outputs/ that are useful to include in your
project report/presentation.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/student_data.csv")

print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

# 1. Distribution of final scores
plt.figure(figsize=(8, 5))
sns.histplot(df["final_score"], bins=30, kde=True, color="#4C72B0")
plt.title("Distribution of Final Exam Scores")
plt.xlabel("Final Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("outputs/01_score_distribution.png", dpi=150)
plt.close()

# 2. Study hours vs final score
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="study_hours_per_week", y="final_score", hue="pass_fail", alpha=0.6)
plt.title("Study Hours per Week vs Final Score")
plt.tight_layout()
plt.savefig("outputs/02_study_hours_vs_score.png", dpi=150)
plt.close()

# 3. Attendance vs final score
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="attendance_percent", y="final_score", hue="pass_fail", alpha=0.6)
plt.title("Attendance % vs Final Score")
plt.tight_layout()
plt.savefig("outputs/03_attendance_vs_score.png", dpi=150)
plt.close()

# 4. Correlation heatmap (numeric features)
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/04_correlation_heatmap.png", dpi=150)
plt.close()

# 5. Pass/Fail counts by parental education
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="parental_education", hue="pass_fail",
              order=["High School", "Bachelors", "Masters", "PhD"])
plt.title("Pass/Fail Count by Parental Education Level")
plt.tight_layout()
plt.savefig("outputs/05_parental_education_vs_result.png", dpi=150)
plt.close()

# 6. Performance category counts
plt.figure(figsize=(8, 5))
order = ["Poor", "Below Average", "Average", "Good", "Excellent"]
sns.countplot(data=df, x="performance_category", order=order, palette="viridis")
plt.title("Performance Category Distribution")
plt.tight_layout()
plt.savefig("outputs/06_performance_category_distribution.png", dpi=150)
plt.close()

print("\nEDA complete. Charts saved to outputs/")
