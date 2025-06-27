"""
XGBoost (Between-Subject Evaluation)

This script performs between-subject evaluation using XGBoost classifiers. Each subject is used as a test subject (Leave-One-Subject-Out - LOSO),
while the rest are used for training and cross-validation.

For each test subject:
    - 5-fold CV is done on the training set to find the best hyperparameters.
    - A final model is trained on the full training set using those best hyperparameters.
    - The model is tested on the test subject, and results are saved.

Approaches:
    5 - One characteristic (grid format)
    6 - Multiple characteristics (grid format)

Outputs:
    - Per-subject trained models (*.pkl)
    - CSV with metrics (accuracy, F1) and best parameters per subject
    - Feature importances CSV and top-20 plot (only for approach 6)
"""

import os
import sys
import numpy as np
import joblib
import random
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from itertools import product

# ========== Setup ==========
sys.path.insert(0, "/home/jovyan/Final_TFM/Data/Scripts/4. Load")
from load_between import load_between_model

ROOT_DIR = "/home/jovyan/Final_TFM"
subjects = [
    "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
    "sub11", "sub12", "sub13", "sub14", "sub15", "sub16"
]

approach_output_dirs = {
    "5": "Results/2. between_subject/a. one characteristic/5. XGBoost",
    "6": "Results/2. between_subject/b. multiple characteristic/5. XGBoost",
}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def group_subjects(subjects, n_folds):
    shuffled = subjects.copy()
    random.shuffle(shuffled)
    folds = [[] for _ in range(n_folds)]
    for i, subj in enumerate(shuffled):
        folds[i % n_folds].append(subj)
    return folds

def get_kfold_data(folds, all_data):
    fold_data = []
    for i in range(len(folds)):
        val_subjects = folds[i]
        train_subjects = [s for j, fold in enumerate(folds) if j != i for s in fold]

        X_train, y_train = [], []
        for subj in train_subjects:
            X, y = all_data[subj]
            X_train.append(X.reshape(X.shape[0], -1))
            y_train.append(y)

        X_val, y_val = [], []
        for subj in val_subjects:
            X, y = all_data[subj]
            X_val.append(X.reshape(X.shape[0], -1))
            y_val.append(y)

        fold_data.append((np.concatenate(X_train), np.concatenate(y_train),
                          np.concatenate(X_val), np.concatenate(y_val)))
    return fold_data

def xgboost_between_model(approach: str):
    if approach not in approach_output_dirs:
        raise ValueError("Only approaches 5 and 6 are supported.")

    seed_everything(42)
    results = []

    output_dir = os.path.join(ROOT_DIR, approach_output_dirs[approach])
    os.makedirs(output_dir, exist_ok=True)

    all_data = load_between_model(approach)

    param_grid = {
        "xgb__n_estimators": [100, 200],
        "xgb__max_depth": [3, 5],
        "xgb__learning_rate": [0.01, 0.1],
        "xgb__subsample": [0.8, 1.0],
    }

    print("\n XGBoost will use GPU acceleration with 'device=\"cuda\"' if available.")

    for test_subject in subjects:
        print(f"\nTest subject: {test_subject}")
        train_subjects = [s for s in subjects if s != test_subject]

        folds = group_subjects(train_subjects, n_folds=5)
        cv_data = get_kfold_data(folds, all_data)

        best_score = -np.inf
        best_params = None

        for values in product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), values))
            accs = []

            for X_train, y_train, X_val, y_val in cv_data:
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("xgb", XGBClassifier(
                        n_estimators=params["xgb__n_estimators"],
                        max_depth=params["xgb__max_depth"],
                        learning_rate=params["xgb__learning_rate"],
                        subsample=params["xgb__subsample"],
                        tree_method='hist',
                        device='cuda',
                        random_state=42,
                        eval_metric="logloss"
                    ))
                ])
                pipeline.fit(X_train, y_train)
                acc = accuracy_score(y_val, pipeline.predict(X_val))
                accs.append(acc)

            mean_acc = np.mean(accs)
            if mean_acc > best_score:
                best_score = mean_acc
                best_params = params

        # === Train final model with best parameters
        X_train_all, y_train_all = [], []
        for subj in train_subjects:
            X, y = all_data[subj]
            X_train_all.append(X.reshape(X.shape[0], -1))
            y_train_all.append(y)

        X_train_all = np.concatenate(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        best_model = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", XGBClassifier(
                n_estimators=best_params["xgb__n_estimators"],
                max_depth=best_params["xgb__max_depth"],
                learning_rate=best_params["xgb__learning_rate"],
                subsample=best_params["xgb__subsample"],
                tree_method='hist',
                device='cuda',
                random_state=42,
                eval_metric="logloss"
            ))
        ])
        best_model.fit(X_train_all, y_train_all)

        # === Evaluate
        X_test, y_test = all_data[test_subject]
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        y_pred = best_model.predict(X_test_flat)

        acc = accuracy_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred) * 100

        results.append({
            "test_subject": test_subject,
            "accuracy": acc,
            "f1_score": f1,
            **best_params
        })

        joblib.dump(best_model, os.path.join(output_dir, f"{test_subject}_model.pkl"))

        print(f" Finished {test_subject}. Accuracy: {acc:.2f}%, F1: {f1:.2f}%")

        # === Feature importances (only for approach 6)
        if approach == "6":
            xgb = best_model.named_steps["xgb"]
            importances = xgb.feature_importances_
            df_feat = pd.DataFrame({
                "feature_index": np.arange(len(importances)),
                "importance": importances
            }).sort_values("importance", ascending=False)

            df_feat.to_csv(os.path.join(output_dir, f"{test_subject}_feature_importances.csv"), index=False)

            top_feats = df_feat.head(20)
            plt.figure(figsize=(12, 8))
            plt.barh(top_feats["feature_index"][::-1], top_feats["importance"][::-1])
            plt.xlabel("Importance")
            plt.title(f"Top 20 Features - {test_subject}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{test_subject}_top20_features.png"))
            plt.close()

    # === Save final results
    df_results = pd.DataFrame(results)
    acc_mean = df_results["accuracy"].mean()
    acc_std = df_results["accuracy"].std()
    f1_mean = df_results["f1_score"].mean()
    f1_std = df_results["f1_score"].std()

    summary_rows = pd.DataFrame([
        {
            "test_subject": "MEAN",
            "accuracy": acc_mean,
            "f1_score": f1_mean,
            "xgb__n_estimators": "",
            "xgb__max_depth": "",
            "xgb__learning_rate": "",
            "xgb__subsample": ""
        },
        {
            "test_subject": "STD",
            "accuracy": acc_std,
            "f1_score": f1_std,
            "xgb__n_estimators": "",
            "xgb__max_depth": "",
            "xgb__learning_rate": "",
            "xgb__subsample": ""
        }
    ])

    df_results = pd.concat([df_results, summary_rows], ignore_index=True)
    csv_path = os.path.join(output_dir, "xgb_loso_results.csv")
    df_results.to_csv(csv_path, index=False)

    print(f"\n Final Results (Approach {approach})")
    print(f"Mean Accuracy: {acc_mean:.2f}% ± {acc_std:.2f}%")
    print(f"Mean F1 Score: {f1_mean:.2f}% ± {f1_std:.2f}%")
    print(f"Results saved to: {csv_path}")
