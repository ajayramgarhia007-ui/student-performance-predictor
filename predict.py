"""
predict.py
----------
Command-line tool to predict a NEW student's final score and pass/fail
result using the models trained by train_model.py.

Usage:
    python3 predict.py                # interactive prompts
    python3 predict.py --demo         # runs with example values, no prompts
"""

import argparse
import json
import joblib
import pandas as pd

MODEL_DIR = "models"


def load_artifacts():
    regressor = joblib.load(f"{MODEL_DIR}/best_regressor.joblib")
    classifier = joblib.load(f"{MODEL_DIR}/best_classifier.joblib")
    category_clf = joblib.load(f"{MODEL_DIR}/category_classifier.joblib")
    with open(f"{MODEL_DIR}/metadata.json") as f:
        metadata = json.load(f)
    return regressor, classifier, category_clf, metadata


def prompt_float(label, lo, hi):
    while True:
        try:
            val = float(input(f"{label} ({lo}-{hi}): "))
            if lo <= val <= hi:
                return val
            print(f"  Please enter a value between {lo} and {hi}.")
        except ValueError:
            print("  Please enter a number.")


def prompt_choice(label, options):
    print(f"{label}: {', '.join(options)}")
    while True:
        val = input(f"Choose {label}: ").strip()
        if val in options:
            return val
        print(f"  Please choose one of: {', '.join(options)}")


def collect_input_interactive(metadata):
    print("\nEnter the student's details:\n")
    data = {
        "study_hours_per_week": prompt_float("Study hours per week", 0, 40),
        "attendance_percent": prompt_float("Attendance percent", 0, 100),
        "previous_score": prompt_float("Previous exam score", 0, 100),
        "sleep_hours": prompt_float("Average sleep hours", 3, 10),
        "absences": prompt_float("Number of absences", 0, 40),
    }
    for cat_feature in metadata["categorical_features"]:
        options = metadata["categorical_options"][cat_feature]
        data[cat_feature] = prompt_choice(cat_feature.replace("_", " ").title(), options)
    return data


def demo_input():
    return {
        "study_hours_per_week": 14.0,
        "attendance_percent": 88.0,
        "previous_score": 72.0,
        "sleep_hours": 7.0,
        "absences": 3,
        "parental_education": "Bachelors",
        "internet_access": "Yes",
        "extracurricular_activities": "Yes",
        "part_time_job": "No",
        "study_method": "Group Study",
    }


def predict(data, regressor, classifier, category_clf):
    X_new = pd.DataFrame([data])
    predicted_score = float(regressor.predict(X_new)[0])
    predicted_score = max(0, min(100, predicted_score))
    pass_fail_pred = classifier.predict(X_new)[0]
    pass_proba = classifier.predict_proba(X_new)[0][1]  # probability of "Pass"
    category = category_clf.predict(X_new)[0]
    return predicted_score, ("Pass" if pass_fail_pred == 1 else "Fail"), pass_proba, category


def main():
    parser = argparse.ArgumentParser(description="Predict student performance")
    parser.add_argument("--demo", action="store_true", help="Run with example values, no prompts")
    args = parser.parse_args()

    regressor, classifier, category_clf, metadata = load_artifacts()

    if args.demo:
        data = demo_input()
        print("Running demo prediction with example student:")
        print(data)
    else:
        data = collect_input_interactive(metadata)

    score, result, proba, category = predict(data, regressor, classifier, category_clf)

    print("\n" + "=" * 40)
    print(f"Predicted Final Score : {score:.1f} / 100")
    print(f"Predicted Result      : {result}")
    print(f"Probability of Passing: {proba * 100:.1f}%")
    print(f"Performance Category  : {category}")
    print("=" * 40)


if __name__ == "__main__":
    main()
