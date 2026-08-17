# 🎓 Student Performance Predictor

A complete Machine Learning project that predicts a student's **final exam score**
and **Pass/Fail result** from academic and lifestyle factors such as study
hours, attendance, previous scores, and more. Built with Python and
scikit-learn, with an interactive Streamlit web app for live predictions.

---

## 1. Project Overview

| | |
|---|---|
| **Problem type** | Supervised Machine Learning — Regression + Classification |
| **Goal (Regression)** | Predict a student's final exam score (0–100) |
| **Goal (Classification)** | Predict whether a student will Pass or Fail |
| **Dataset** | 1,200 synthetic but realistically-modeled student records |
| **Language / Libraries** | Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Streamlit |

This project demonstrates the full ML pipeline: data generation/cleaning,
exploratory data analysis (EDA), feature engineering, model training,
hyperparameter tuning, evaluation, explainability, and deployment via an
interactive multi-tab web app — making it suitable as a complete academic
submission.

**Key features:**
- Predicts both an exact **final score** (regression) and **Pass/Fail** (classification)
- A third model predicts a 5-tier **performance category** (Poor → Excellent)
- Hyperparameter tuning (`RandomizedSearchCV`) for the tree-based models
- **"Why this prediction?"** explainability chart showing which factors pushed the score up or down
- **Batch prediction**: upload a CSV of many students, get a downloadable results file
- **Prediction history** within your session, with a trend chart
- A **Model Insights** tab with comparison tables, feature importance, and dataset preview

---

## 2. Project Structure

```
student-performance-predictor/
├── data/
│   └── student_data.csv           # Generated dataset
├── models/
│   ├── best_regressor.joblib        # Saved best regression model
│   ├── best_classifier.joblib       # Saved best Pass/Fail model
│   ├── category_classifier.joblib   # Saved 5-tier category model
│   ├── regression_coefficients.json # Explainability data for "why this prediction"
│   └── metadata.json                # Feature lists, model info, dataset stats
├── outputs/
│   ├── 01_score_distribution.png
│   ├── 02_study_hours_vs_score.png
│   ├── 03_attendance_vs_score.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_parental_education_vs_result.png
│   ├── 06_performance_category_distribution.png
│   ├── regression_model_comparison.csv
│   ├── classification_model_comparison.csv
│   └── feature_importance.csv
├── generate_dataset.py            # Step 1: creates the dataset
├── eda.py                         # Step 2: exploratory data analysis
├── train_model.py                 # Step 3: trains & evaluates models
├── predict.py                     # Step 4: CLI prediction tool
├── app.py                         # Step 5: Streamlit web app
├── requirements.txt
└── README.md
```

---

## 3. How to Run (Step by Step)

### Setup
```bash
pip install -r requirements.txt
```

### Step 1 — Generate the dataset
```bash
python3 generate_dataset.py
```
Creates `data/student_data.csv` with 1,200 student records.

### Step 2 — Explore the data
```bash
python3 eda.py
```
Saves 6 charts to `outputs/` (score distribution, correlations, etc.)
— useful directly in your project report or slides.

### Step 3 — Train the models
```bash
python3 train_model.py
```
Trains **4 regression algorithms** (with hyperparameter tuning for the
tree-based ones), **4 classification algorithms**, and a **5-tier category
classifier**. Saves the best of each to `models/`, plus explainability data.
This step takes a bit longer than before due to the tuning search — that's expected.

### Step 4 — Predict from the command line
```bash
python3 predict.py           # interactive prompts
python3 predict.py --demo    # runs instantly with example data
```
Now also reports the predicted performance category (Poor → Excellent).

### Step 5 — Launch the web app
```bash
streamlit run app.py
```
Opens an interactive browser UI with two tabs:
- **Single Prediction** — sliders/dropdowns, predicted score, Pass/Fail,
  performance category badge, a "why this prediction?" explainability
  chart, a comparison to the dataset average, and a running history of
  everything you've predicted this session.
- **Batch Prediction (CSV)** — upload a CSV of many students (a template
  is provided) and download a results file with predictions for all of them.

Model comparison tables and feature importance are still generated as
CSV files in `outputs/` by `train_model.py` (useful for your report), even
though they're no longer shown as a tab in the app itself.

---

## 4. Dataset Description

Each row represents one student, with these features:

| Feature | Type | Description |
|---|---|---|
| `study_hours_per_week` | numeric | Hours spent studying per week |
| `attendance_percent` | numeric | Class attendance percentage |
| `previous_score` | numeric | Score in the previous exam/term |
| `sleep_hours` | numeric | Average hours of sleep per night |
| `absences` | numeric | Number of classes missed |
| `parental_education` | categorical | Highest parental education level |
| `internet_access` | categorical | Home internet access (Yes/No) |
| `extracurricular_activities` | categorical | Participates in extracurriculars |
| `part_time_job` | categorical | Has a part-time job |
| `study_method` | categorical | Preferred study method |
| `final_score` | target (regression) | Final exam score (0–100) |
| `pass_fail` | target (classification) | Pass if score ≥ 40, else Fail |
| `performance_category` | derived | Poor / Below Average / Average / Good / Excellent |

The scores were generated from a weighted formula (study hours, attendance,
prior performance, sleep, and other factors) plus random noise, so
relationships are realistic rather than perfectly deterministic — similar
to real academic data.

> **Using real data instead:** if your college provides an actual dataset
> (e.g. a CSV of real student records), just replace `data/student_data.csv`
> with it — keeping the same column names — and re-run `train_model.py`.

---

## 5. Model Results

### Regression — predicting exact final score

| Model | MAE | RMSE | R² | CV R² (5-fold) |
|---|---|---|---|---|
| **Linear Regression** ⭐ | 5.283 | 6.555 | **0.707** | 0.666 |
| Gradient Boosting | 5.478 | 6.949 | 0.671 | 0.608 |
| Random Forest | 5.948 | 7.528 | 0.614 | 0.567 |
| Decision Tree | 7.498 | 9.478 | 0.388 | 0.332 |

**Best model: Linear Regression** — the relationships in this dataset are
mostly linear (by design), so a simple, interpretable model wins over more
complex ones. R² of 0.71 means the model explains ~71% of the variance in
final scores.

### Classification — predicting Pass/Fail

| Model | Accuracy | Precision | Recall | F1 | CV F1 (5-fold) |
|---|---|---|---|---|---|
| **Random Forest** ⭐ | 0.896 | 0.906 | 0.986 | **0.945** | 0.953 |
| Logistic Regression | 0.892 | 0.913 | 0.972 | 0.942 | 0.958 |
| SVM | 0.879 | 0.908 | 0.963 | 0.935 | 0.952 |
| Decision Tree | 0.821 | 0.895 | 0.907 | 0.901 | 0.932 |

**Best model: Random Forest** — ~90% accuracy with a strong F1 score.

### Most important features (from Random Forest feature importance)
1. Study hours per week
2. Previous exam score
3. Attendance percentage
4. Sleep hours
5. Absences

These align with education-research intuition: consistent study habits,
prior performance, and attendance are the strongest predictors of results.

### Multi-class — predicting the 5-tier performance category

The `performance_category` model (Poor / Below Average / Average / Good /
Excellent) achieves roughly **60% accuracy**. This is expected and worth
discussing in your report: a 5-class problem is inherently harder than
binary Pass/Fail, and category boundaries near a threshold (e.g. a score of
84 vs. 85) are genuinely ambiguous even for a well-fit model. It's included
to demonstrate multi-class classification, not as the app's primary metric
— the score (regression) and Pass/Fail (binary classification) predictions
are the more reliable outputs.

### Explainability

Because Linear Regression is the best-performing regressor, its coefficients
double as a natural explanation: for any prediction, the app breaks the
score down into per-feature contributions (e.g. "+3.1 points from
attendance", "-0.8 points from parental education") relative to a baseline
intercept. This is exact for Linear Regression and is a standard way to
explain linear models without needing extra libraries like SHAP.

---

## 6. Methodology Notes (for your report)

- **Preprocessing**: numeric features are standardized (`StandardScaler`);
  categorical features are one-hot encoded (`OneHotEncoder`) inside a
  scikit-learn `Pipeline` + `ColumnTransformer`, so preprocessing and the
  model are saved together — no risk of train/test mismatch.
- **Train/test split**: 80/20, with stratification on the classification
  target to preserve the Pass/Fail ratio.
- **Validation**: in addition to a held-out test set, 5-fold cross-validation
  is reported for a more robust estimate of generalization.
- **Model selection**: the script automatically picks the model with the
  best test-set R² (regression) / F1 score (classification), so this project
  can be re-run on different or updated data and will always save the
  strongest model.

---

## 7. Possible Extensions (for extra credit / discussion)

- Swap in a real institutional dataset once column names are matched.
- Add hyperparameter tuning (`GridSearchCV` / `RandomizedSearchCV`).
- Add SHAP values for more detailed model explainability.
- Deploy the Streamlit app publicly (Streamlit Community Cloud, Render, etc.).
- Extend `performance_category` into a 5-class classifier as a stretch goal.

---

## 8. Author Notes

This project was scaffolded to be fully runnable end-to-end with the
included synthetic dataset, and easily adaptable to real student data for
a college course project or capstone submission.
