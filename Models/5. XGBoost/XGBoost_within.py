"""     
XGBoost (Within-Subject Evaluation)

This script performs within-subject evaluation using XGBoost classifiers for different feature extraction approaches.
Each subject is split into train/test data.
For each subject:
    - A grid search is performed to find the best hyperparameters using cross-validation.
    - A final model is trained using the most common best parameters across all subjects.
    - Results and feature importances (if applicable) are saved.

Approaches:
    1 - One characteristic, slice-wise
    2 - One characteristic, trial-wise
    3 - Multiple characteristics, slice-wise
    4 - Multiple characteristics, trial-wise

Outputs:
    - Per-subject best models and final common models (*.pkl)
    - Hyperparameter summary and final metrics CSV
    - Feature importances (CSV) for approaches 3 and 4
"""

import os
import sys
import joblib
import random
import torch
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

# ========== SETUP ==========
ROOT_DIR = "/home/jovyan/Final_TFM"
sys.path.append(os.path.join(ROOT_DIR, "Data", "Scripts", "4. Load"))
from load_within import load_within_subject_data

# Define output directories
approach_output_dirs = {
    "1": "Results/1. with_in_subject/a. one_characteristic/slice_wise/5. XGBoost",
    "2": "Results/1. with_in_subject/a. one_characteristic/trial_wise/5. XGBoost",
    "3": "Results/1. with_in_subject/b. multiple_characteristic/slice_wise/5. XGBoost",
    "4": "Results/1. with_in_subject/b. multiple_characteristic/trial_wise/5. XGBoost",
}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========== MAIN FUNCTION ==========
def xgboost_within_model(approach: str):
    if approach not in approach_output_dirs:
        raise ValueError("Only approaches 1, 2, 3, 4 are allowed.")

    seed_everything(42)
    all_data = load_within_subject_data(approach)
    output_dir = os.path.join(ROOT_DIR, approach_output_dirs[approach])
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to: {output_dir}")

    # Hyperparameter grid
    param_grid = {
        "xgb__n_estimators": [100, 200],
        "xgb__max_depth": [3, 5],
        "xgb__learning_rate": [0.01, 0.1],
        "xgb__subsample": [0.8, 1.0],
    }

    results = []
    param_counts = []

    # ========== HYPERPARAMETER TUNING ==========
    for subject, (X_train, _, y_train, _) in all_data.items():
        print(f"\nSearching hyperparameters for {subject}...")

        X_train_flat = np.array([x.flatten() for x in X_train])
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", XGBClassifier(random_state=42, eval_metric="logloss"))
        ])

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
            scoring="accuracy",
            n_jobs=-1
        )
        grid_search.fit(X_train_flat, y_train)
        best_params = grid_search.best_params_
        print(f"{subject} best parameters: {best_params}")

        joblib.dump(
            grid_search.best_estimator_,
            os.path.join(output_dir, f"{subject}_best_xgb.pkl")
        )

        results.append({"subject": subject, **best_params, "cv_score": grid_search.best_score_})
        param_counts.append((
            best_params["xgb__n_estimators"],
            best_params["xgb__max_depth"],
            best_params["xgb__learning_rate"]
        ))

    # Save all individual best hyperparameters
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "xgb_hyperparams_by_subject.csv"), index=False)

    # ========== FINAL TRAINING WITH COMMON PARAMS ==========
    most_common = Counter(param_counts).most_common(1)[0][0]
    print("\nTraining final model with most common hyperparameters:")
    print(f"n_estimators={most_common[0]}, max_depth={most_common[1]}, learning_rate={most_common[2]}")

    final_results = []

    for subject, (X_train, X_test, y_train, y_test) in all_data.items():
        print(f"\nEvaluating {subject}...")

        X_train_flat = np.array([x.flatten() for x in X_train])
        X_test_flat = np.array([x.flatten() for x in X_test])

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", XGBClassifier(
                n_estimators=most_common[0],
                max_depth=most_common[1],
                learning_rate=most_common[2],
                random_state=42,
                eval_metric="logloss"
            ))
        ])
        model.fit(X_train_flat, y_train)
        y_pred = model.predict(X_test_flat)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        joblib.dump(model, os.path.join(output_dir, f"{subject}_final_model_common.pkl"))

        final_results.append({
            "subject": subject,
            "final_accuracy": acc,
            "final_f1_score": f1
        })

        print(f"{subject} → Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%")

        # Save feature importances for approaches with multiple characteristics
        if approach in ["3", "4"]:
            xgb_model = model.named_steps["xgb"]
            importances = xgb_model.feature_importances_
            df_imp = pd.DataFrame({
                "feature": np.arange(len(importances)),
                "importance": importances
            })
            df_imp.sort_values("importance", ascending=False).to_csv(
                os.path.join(output_dir, f"{subject}_feature_importance.csv"), index=False
            )

    # ========== SAVE FINAL METRICS ==========
    df_final = pd.DataFrame(final_results)
    mean_acc = df_final["final_accuracy"].mean() * 100
    std_acc = df_final["final_accuracy"].std() * 100
    mean_f1 = df_final["final_f1_score"].mean() * 100
    std_f1 = df_final["final_f1_score"].std() * 100

    df_final = pd.concat([
        df_final,
        pd.DataFrame([{
            "subject": "AVG±STD",
            "final_accuracy": f"{mean_acc:.2f} ± {std_acc:.2f}",
            "final_f1_score": f"{mean_f1:.2f} ± {std_f1:.2f}",
        }])
    ], ignore_index=True)

    df_final.to_csv(os.path.join(output_dir, "xgb_final_metrics_common_params.csv"), index=False)

    print("\n✅ Evaluation completed.")
    print(f"Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"F1 Score: {mean_f1:.2f} ± {std_f1:.2f}")
