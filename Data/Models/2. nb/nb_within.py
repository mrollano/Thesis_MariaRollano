"""
Naive Bayes (Within-Subject Evaluation)

This script trains and evaluates a Gaussian Naive Bayes classifier using 
within-subject data for different feature extraction approaches (1–4).

Each subject is trained individually using their own train/test split. 
The training involves hyperparameter tuning via 10-fold cross-validation 
(`var_smoothing`), followed by final training and evaluation using the 
most common best hyperparameter across subjects.

Approaches:
    1. One characteristic – Slice-wise
    2. One characteristic – Trial-wise
    3. Multiple characteristics – Slice-wise
    4. Multiple characteristics – Trial-wise

Input:
    - train_test_split.pkl or train_test_split_trial_wise.pkl for each subject

Output:
    - Best estimator (per subject): *_best_nb.pkl
    - Final model (common hyperparam): *_final_model_common.pkl
    - Hyperparameter CSV: nb_hyperparams_by_subject.csv
    - Evaluation CSV: nb_final_metrics_common_params.csv
"""

import os
import sys
import joblib
import random
import torch
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

# Root directory and module import
ROOT_DIR = "/home/jovyan/Final_TFM"
sys.path.append(os.path.join(ROOT_DIR, "Data", "Scripts", "4. Load"))
from load_within import load_within_subject_data

# Output directories by approach
approach_output_dirs = {
    "1": "Results/1. with_in_subject/a. one_characteristic/slice_wise/2. nb",
    "2": "Results/1. with_in_subject/a. one_characteristic/trial_wise/2. nb",
    "3": "Results/1. with_in_subject/b. multiple_characteristic/slice_wise/2. nb",
    "4": "Results/1. with_in_subject/b. multiple_characteristic/trial_wise/2. nb",
}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def nb_within_model(approach: str):
    if approach not in approach_output_dirs:
        raise ValueError("Only approaches 1, 2, 3, and 4 are supported.")

    seed_everything()
    all_data = load_within_subject_data(approach)

    output_dir = os.path.join(ROOT_DIR, approach_output_dirs[approach])
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to: {output_dir}")

    # Grid of hyperparameters
    param_grid = {
        "nb__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }

    results = []
    param_counts = []

    # === Grid Search per subject ===
    for subject, (X_train, _, y_train, _) in all_data.items():
        print(f"\nSearching hyperparameters for {subject}...")

        X_train_flat = np.array([x.flatten() for x in X_train])
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("nb", GaussianNB())
        ])

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=KFold(n_splits=10, shuffle=True, random_state=42),
            scoring="accuracy",
            n_jobs=-1
        )
        grid_search.fit(X_train_flat, y_train)
        best_params = grid_search.best_params_
        best_vs = best_params["nb__var_smoothing"]

        joblib.dump(grid_search.best_estimator_, os.path.join(output_dir, f"{subject}_best_nb.pkl"))

        results.append({
            "subject": subject,
            "var_smoothing": best_vs,
            "cv_score": grid_search.best_score_
        })
        param_counts.append(best_vs)

    # Save best hyperparameters
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "nb_hyperparams_by_subject.csv"), index=False)

    # === Final training with most common var_smoothing ===
    most_common = Counter(param_counts).most_common(1)[0][0]
    print(f"\nTraining final models with most common var_smoothing: {most_common}")

    final_results = []

    for subject, (X_train, X_test, y_train, y_test) in all_data.items():
        print(f"\nEvaluating {subject} with final model...")

        X_train_flat = np.array([x.flatten() for x in X_train])
        X_test_flat = np.array([x.flatten() for x in X_test])

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("nb", GaussianNB(var_smoothing=most_common))
        ])
        model.fit(X_train_flat, y_train)
        y_pred = model.predict(X_test_flat)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        joblib.dump(model, os.path.join(output_dir, f"{subject}_final_model_common.pkl"))
        final_results.append({
            "subject": subject,
            "final_accuracy": acc * 100,
            "final_f1_score": f1 * 100
        })

    # Save final metrics
    df_final = pd.DataFrame(final_results)
    mean_acc = df_final["final_accuracy"].mean()
    std_acc = df_final["final_accuracy"].std()
    mean_f1 = df_final["final_f1_score"].mean()
    std_f1 = df_final["final_f1_score"].std()

    df_final = pd.concat([
        df_final,
        pd.DataFrame([{
            "subject": "AVG±STD",
            "final_accuracy": f"{mean_acc:.2f} ± {std_acc:.2f}",
            "final_f1_score": f"{mean_f1:.2f} ± {std_f1:.2f}"
        }])
    ], ignore_index=True)

    df_final.to_csv(os.path.join(output_dir, "nb_final_metrics_common_params.csv"), index=False)

    print("\nFinal evaluation completed.")
    print(f"Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"F1 Score: {mean_f1:.2f} ± {std_f1:.2f}")
