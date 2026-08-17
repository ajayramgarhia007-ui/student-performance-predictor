"""
train_model.py
---------------
Trains and evaluates machine learning models on the student performance
dataset, then saves the best-performing pipeline(s) to models/ for reuse
in predict.py and app.py.

Three tasks are solved:
  1. REGRESSION       -> predict the exact final_score (0-100)
  2. CLASSIFICATION    -> predict pass_fail (Pass / Fail)
  3. MULTI-CLASS       -> predict performance_category (Poor..Excellent)

Includes:
  - Hyperparameter tuning (RandomizedSearchCV) for the top regression models
  - Model explainability data (linear coefficients) saved for the app to
    show "why" a prediction came out the way it did
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

RANDOM_STATE = 42
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------
df = pd.read_csv("data/student_data.csv")

numeric_features = [
    "study_hours_per_week", "attendance_percent", "previous_score",
    "sleep_hours", "absences"
]
categorical_features = [
    "parental_education", "internet_access",
    "extracurricular_activities", "part_time_job", "study_method"
]

X = df[numeric_features + categorical_features]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# --------------------------------------------------------------------
# 2. REGRESSION: predict final_score (with hyperparameter tuning)
# --------------------------------------------------------------------
y_reg = df["final_score"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=RANDOM_STATE
)

regressors = {
    "Linear Regression": (LinearRegression(), None),
    "Decision Tree": (DecisionTreeRegressor(random_state=RANDOM_STATE), {
        "model__max_depth": [3, 5, 6, 8, 10, None],
        "model__min_samples_leaf": [1, 2, 4, 8],
    }),
    "Random Forest": (RandomForestRegressor(random_state=RANDOM_STATE), {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [4, 6, 8, 10, None],
        "model__min_samples_leaf": [1, 2, 4],
    }),
    "Gradient Boosting": (GradientBoostingRegressor(random_state=RANDOM_STATE), {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": [2, 3, 4],
    }),
}

reg_results = []
best_reg_name, best_reg_pipeline, best_reg_r2 = None, None, -np.inf

print("Training + tuning regression models (this may take a moment)...\n")

for name, (model, param_grid) in regressors.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

    if param_grid:
        search = RandomizedSearchCV(
            pipe, param_grid, n_iter=10, cv=3, scoring="r2",
            random_state=RANDOM_STATE, n_jobs=-1
        )
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
    else:
        pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    cv_scores = cross_val_score(pipe, X, y_reg, cv=5, scoring="r2")

    reg_results.append({
        "model": name, "MAE": round(mae, 3), "RMSE": round(rmse, 3),
        "R2": round(r2, 3), "CV_R2_mean": round(cv_scores.mean(), 3),
        "tuned": param_grid is not None
    })

    if r2 > best_reg_r2:
        best_reg_r2, best_reg_name, best_reg_pipeline = r2, name, pipe

reg_results_df = pd.DataFrame(reg_results).sort_values("R2", ascending=False)
print("=== Regression Results (predicting final_score) ===")
print(reg_results_df.to_string(index=False))
print(f"\nBest regressor: {best_reg_name} (R2 = {best_reg_r2:.3f})")

joblib.dump(best_reg_pipeline, "models/best_regressor.joblib")
reg_results_df.to_csv("outputs/regression_model_comparison.csv", index=False)

# --------------------------------------------------------------------
# 3. CLASSIFICATION: predict pass_fail
# --------------------------------------------------------------------
y_clf = df["pass_fail"].map({"Fail": 0, "Pass": 1})
X_train, X_test, y_train, y_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
)

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
}

clf_results = []
best_clf_name, best_clf_pipeline, best_clf_f1 = None, None, -np.inf

for name, model in classifiers.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cv_scores = cross_val_score(pipe, X, y_clf, cv=5, scoring="f1")

    clf_results.append({
        "model": name, "Accuracy": round(acc, 3), "Precision": round(prec, 3),
        "Recall": round(rec, 3), "F1": round(f1, 3), "CV_F1_mean": round(cv_scores.mean(), 3)
    })

    if f1 > best_clf_f1:
        best_clf_f1, best_clf_name, best_clf_pipeline = f1, name, pipe

clf_results_df = pd.DataFrame(clf_results).sort_values("F1", ascending=False)
print("\n=== Classification Results (predicting Pass/Fail) ===")
print(clf_results_df.to_string(index=False))
print(f"\nBest classifier: {best_clf_name} (F1 = {best_clf_f1:.3f})")

joblib.dump(best_clf_pipeline, "models/best_classifier.joblib")
clf_results_df.to_csv("outputs/classification_model_comparison.csv", index=False)

cm = confusion_matrix(y_test, best_clf_pipeline.predict(X_test))
print("\nConfusion Matrix (best classifier):\n", cm)

# --------------------------------------------------------------------
# 4. MULTI-CLASS: predict performance_category (5 tiers)
# --------------------------------------------------------------------
category_order = ["Poor", "Below Average", "Average", "Good", "Excellent"]
y_cat = df["performance_category"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cat
)

cat_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))
])
cat_pipe.fit(X_train, y_train)
cat_preds = cat_pipe.predict(X_test)
cat_acc = accuracy_score(y_test, cat_preds)
print(f"\n=== Multi-class Results (predicting performance_category) ===")
print(f"Accuracy: {cat_acc:.3f}")

joblib.dump(cat_pipe, "models/category_classifier.joblib")

# --------------------------------------------------------------------
# 5. Feature importance (Random Forest regressor)
# --------------------------------------------------------------------
try:
    rf_pipe = Pipeline([("preprocessor", preprocessor),
                         ("model", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE))])
    rf_pipe.fit(X, y_reg)
    feature_names = rf_pipe.named_steps["preprocessor"].get_feature_names_out()
    importances = rf_pipe.named_steps["model"].feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)
    fi_df.to_csv("outputs/feature_importance.csv", index=False)
    print("\nTop 5 most important features:\n", fi_df.head())
except Exception as e:
    print("Feature importance step skipped:", e)

# --------------------------------------------------------------------
# 6. Explainability: save linear coefficients for per-prediction breakdown
#    (used by app.py to show "why" a score came out the way it did)
# --------------------------------------------------------------------
model_step = best_reg_pipeline.named_steps["model"]
if hasattr(model_step, "coef_"):
    explain_pipe = best_reg_pipeline
    explain_source = best_reg_name
else:
    # Best model isn't linear -> fit a separate Linear Regression purely
    # for approximate, human-readable explanations
    explain_pipe = Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])
    explain_pipe.fit(X, y_reg)
    explain_source = "Linear Regression (approximation for explainability)"

feat_names = explain_pipe.named_steps["preprocessor"].get_feature_names_out().tolist()
coefs = explain_pipe.named_steps["model"].coef_.tolist()
intercept = float(explain_pipe.named_steps["model"].intercept_)

with open("models/regression_coefficients.json", "w") as f:
    json.dump({
        "source_model": explain_source,
        "intercept": intercept,
        "feature_names": feat_names,
        "coefficients": coefs,
    }, f, indent=2)

print(f"\nSaved explainability data (source: {explain_source})")

# --------------------------------------------------------------------
# 7. Save metadata (used by app.py / predict.py)
# --------------------------------------------------------------------
metadata = {
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "categorical_options": {c: sorted(df[c].unique().tolist()) for c in categorical_features},
    "category_order": category_order,
    "best_regressor": best_reg_name,
    "best_regressor_r2": round(best_reg_r2, 3),
    "best_classifier": best_clf_name,
    "best_classifier_f1": round(best_clf_f1, 3),
    "category_classifier_accuracy": round(cat_acc, 3),
    "dataset_stats": {
        col: {"mean": round(float(df[col].mean()), 2), "std": round(float(df[col].std()), 2)}
        for col in numeric_features
    },
}
with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved models/best_regressor.joblib, models/best_classifier.joblib,")
print("models/category_classifier.joblib, models/regression_coefficients.json, models/metadata.json")
