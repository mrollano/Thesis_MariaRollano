""" 
Random Forest (Within-Subject Evaluation)

This script trains and evaluates a Random Forest classifier using within-subject evaluation
on pre-split training/testing data. It supports approaches 1–4, corresponding to different
data configurations (one vs. multiple characteristics, slice- vs. trial-wise).

Approaches:
    1. One characteristic - Slice-wise
    2. One characteristic - Trial-wise
    3. Multiple characteristics - Slice-wise
    4. Multiple characteristics - Trial-wise

For each subject:
    - Perform hyperparameter tuning with GridSearchCV.
    - Save the best model per subject.
    - Determine most common hyperparameter configuration across all subjects.
    - Train final models using those common hyperparameters and evaluate them.
    - Save model performance metrics and feature importances (if multiple characteristics).

Outputs:
    - Best model per subject (`*_best_rf.pkl`)
    - Final common model per subject (`*_final_model_common.pkl`)
    - Feature importances (for approaches 3 and 4)
    - CSVs with hyperparameters and final performance metrics
"""

import os
import sys
import joblib
import random
import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

# ================= SETUP =================
ROOT_DIR = "/home/jovyan/Final_TFM"
sys.path.append(os.path.join(ROOT_DIR, "Data", "Scripts", "4. Load"))
from load_within import load_within_subject_data

# Define output directories per approach
approach_output_dirs = {
    "1": "Results/1. with_in_subject/a. one_characteristic/slice_wise/4. forest",
    "2": "Results/1. with_in_subject/a. one_characteristic/trial_wise/4. forest",
    "3": "Results/1. with_in_subject/b. multiple_characteristic/slice_wise/4. forest",
    "4": "Results/1. with_in_subject/b. multiple_characteristic/trial_wise/4. forest",
}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rf_within_model(approach: str):
    if approach not in approach_output_dirs:
        raise ValueError("Only approaches 1, 2, 3, 4 are allowed.")

    seed_everything(42)
    all_data = load_within_subject_data(approach)

    output_dir = os.path.join(ROOT_DIR, approach_output_dirs[approach])
    os.makedirs(output_dir, exist_ok=True)

    # Hyperparameter grid for tuning
    param_grid = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [None, 10, 20],
    }

    all_best_params = []
    subject_params_results = []

    # Hyperparameter tuning per subject
    for subject, (X_train, _, y_train, _) in all_data.items():
        print(f"\nTuning hyperparameters for {subject}...")

        X_train_flat = np.array([x.flatten() for x in X_train])

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(random_state=42))
        ])

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=KFold(n_splits=10, shuffle=True, random_state=42),
            scoring="accuracy",
            n_jobs=-1,
        )

        grid_search.fit(X_train_flat, y_train)
        best_params = grid_search.best_params_

        all_best_params.append(tuple(best_params.items()))
        subject_params_results.append({"subject": subject, **best_params})

        joblib.dump(
            grid_search.best_estimator_,
            os.path.join(output_dir, f"{subject}_best_rf.pkl")
        )

    # Determine the most common hyperparameter combination
    most_common = Counter(all_best_params).most_common(1)[0][0]
    common_params = dict(most_common)
    print(f"\nMost common hyperparameters: {common_params}")

    df_params = pd.DataFrame(subject_params_results)
    df_params.to_csv(os.path.join(output_dir, "rf_hyperparams_by_subject.csv"), index=False)

    # Final evaluation using common hyperparameters
    results = []
    feature_importances = []

    for subject, (X_train, X_test, y_train, y_test) in all_data.items():
        print(f"\nEvaluating {subject} with common model...")

        X_train_flat = np.array([x.flatten() for x in X_train])
        X_test_flat = np.array([x.flatten() for x in X_test])

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=common_params["rf__n_estimators"],
                max_depth=common_params["rf__max_depth"],
                random_state=42
            ))
        ])

        model.fit(X_train_flat, y_train)
        y_pred = model.predict(X_test_flat)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "subject": subject,
            "accuracy": acc,
            "f1_score": f1
        })

        joblib.dump(model, os.path.join(output_dir, f"{subject}_final_model_common.pkl"))

        # Save feature importances only for approaches 3 and 4
        if approach in ["3", "4"]:
            importances = model.named_steps["rf"].feature_importances_
            feature_importances.append(importances)
            np.save(os.path.join(output_dir, f"{subject}_feature_importances.npy"), importances)

    df_results = pd.DataFrame(results)

    # Add AVG ± STD row
    mean_acc = df_results["accuracy"].mean() * 100
    std_acc = df_results["accuracy"].std() * 100
    mean_f1 = df_results["f1_score"].mean() * 100
    std_f1 = df_results["f1_score"].std() * 100

    df_results = pd.concat([
        df_results,
        pd.DataFrame([{
            "subject": "AVG±STD",
            "accuracy": f"{mean_acc:.2f} ± {std_acc:.2f}",
            "f1_score": f"{mean_f1:.2f} ± {std_f1:.2f}"
        }])
    ], ignore_index=True)

    df_results.to_csv(os.path.join(output_dir, "rf_final_metrics_common_params.csv"), index=False)

    # Save average feature importances if applicable
    if approach in ["3", "4"] and feature_importances:
        all_importances = np.vstack(feature_importances)
        mean_importance = np.mean(all_importances, axis=0)
        np.save(os.path.join(output_dir, "mean_feature_importances.npy"), mean_importance)

    print("\nEvaluation completed.")
    print(f"Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"F1 Score: {mean_f1:.2f} ± {std_f1:.2f}")
