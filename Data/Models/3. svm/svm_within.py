"""
Support Vector Machine (Within-Subject Evaluation)

This script trains and evaluates an SVM classifier for within-subject classification 
using approaches 1–4 (slice-wise/trial-wise, one or multiple characteristics). 

For each subject:
- A grid search is performed to find the best hyperparameters using 10-fold CV on training data.
- The most common hyperparameters across all subjects are used to train final models.
- Final models are evaluated on each subject's test set.

Approaches:
    1. One characteristic – Slice-wise – Within-subject
    2. One characteristic – Trial-wise – Within-subject
    3. Multiple characteristics – Slice-wise – Within-subject
    4. Multiple characteristics – Trial-wise – Within-subject

Input:
    - Per-subject split: (X_train, X_test, y_train, y_test)

Output:
    - Final model per subject: *_final_model.pkl
    - CSV with performance metrics: svm_final_metrics_by_subject.csv
"""

import os
import sys
import joblib
import random
import torch
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

# ========== Configuration ==========
ROOT_DIR = "/home/jovyan/Final_TFM"
sys.path.append(os.path.join(ROOT_DIR, "Data", "Scripts", "4. Load"))
from load_within import load_within_subject_data

# Output path for each approach
approach_output_dirs = {
    "1": "Results/1. with_in_subject/a. one_characteristic/slice_wise/3. svm",
    "2": "Results/1. with_in_subject/a. one_characteristic/trial_wise/3. svm",
    "3": "Results/1. with_in_subject/b. multiple_characteristic/slice_wise/3. svm",
    "4": "Results/1. with_in_subject/b. multiple_characteristic/trial_wise/3. svm",
}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========== Main Function ==========
def svm_within_model(approach: str):
    if approach not in approach_output_dirs:
        raise ValueError("Only approaches 1, 2, 3, and 4 are supported.")

    seed_everything()
    all_data = load_within_subject_data(approach)

    output_dir = os.path.join(ROOT_DIR, approach_output_dirs[approach])
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to: {output_dir}")

    # Hyperparameter grid
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__kernel": ["linear", "rbf"],
        "svm__gamma": ["scale", "auto"],
    }

    param_records = []

    # ========== Hyperparameter tuning per subject ==========
    for subject, (X_train, _, y_train, _) in all_data.items():
        print(f"\nTuning hyperparameters for {subject}...")

        X_train_flat = np.array([x.flatten() for x in X_train])
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC())
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
        param_records.append(best_params)
        print(f"{subject} → Best Params: {best_params}")

    # ========== Determine most common hyperparameters ==========
    def get_most_common_params(param_records):
        df = pd.DataFrame(param_records)
        return {col: df[col].mode()[0] for col in df.columns}

    best_common = get_most_common_params(param_records)
    print(f"\nUsing common hyperparameters: {best_common}")

    final_results = []

    # ========== Train and evaluate final model per subject ==========
    for subject, (X_train, X_test, y_train, y_test) in all_data.items():
        print(f"\nEvaluating {subject}...")

        X_train_flat = np.array([x.flatten() for x in X_train])
        X_test_flat = np.array([x.flatten() for x in X_test])

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=best_common["svm__C"],
                kernel=best_common["svm__kernel"],
                gamma=best_common["svm__gamma"]
            )),
        ])
        model.fit(X_train_flat, y_train)
        y_pred = model.predict(X_test_flat)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        joblib.dump(model, os.path.join(output_dir, f"{subject}_final_model.pkl"))

        final_results.append({
            "subject": subject,
            "final_accuracy": acc * 100,
            "final_f1_score": f1 * 100
        })

    # ========== Save final metrics ==========
    df = pd.DataFrame(final_results)
    mean_acc = df["final_accuracy"].mean()
    std_acc = df["final_accuracy"].std()
    mean_f1 = df["final_f1_score"].mean()
    std_f1 = df["final_f1_score"].std()

    df = pd.concat([
        df,
        pd.DataFrame([{
            "subject": "AVG±STD",
            "final_accuracy": f"{mean_acc:.2f} ± {std_acc:.2f}",
            "final_f1_score": f"{mean_f1:.2f} ± {std_f1:.2f}",
        }])
    ], ignore_index=True)

    df.to_csv(os.path.join(output_dir, "svm_final_metrics_by_subject.csv"), index=False)

    print("\nEvaluation completed.")
    print(f"Average Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Average F1 Score: {mean_f1:.2f} ± {std_f1:.2f}")
