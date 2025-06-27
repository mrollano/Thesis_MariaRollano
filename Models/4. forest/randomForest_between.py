"""    
Random Forest (Between-Subject Evaluation)

This script performs leave-one-subject-out (LOSO) evaluation using Random Forest classifiers.
It supports both one-characteristic and multiple-characteristics approaches (5 and 6).
For each test subject:
    - The remaining subjects are used for training.
    - 5-fold cross-validation is performed on training data to select the best hyperparameters.
    - A final model is trained using the best parameters and evaluated on the test subject.
    - Results and feature importances (if applicable) are saved.

Approaches:
    5 - One characteristic
    6 - Multiple characteristics

Outputs:
    - Trained model for each test subject (*.pkl)
    - Final results (accuracy and F1) saved as CSV
    - Feature importances (CSV + PNG) for approach 6
"""

import os
import sys
import numpy as np
import joblib
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from itertools import product

# ================= SETUP =================
sys.path.insert(0, "/home/jovyan/Final_TFM/Data/Scripts/4. Load")
from load_between import load_between_model

ROOT_DIR = "/home/jovyan/Final_TFM"
subjects = [
    "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
    "sub11", "sub12", "sub13", "sub14", "sub15", "sub16"
]

approach_output_dirs = {
    "5": "Results/2. between_subject/a. one characteristic/4. forest",
    "6": "Results/2. between_subject/b. multiple characteristic/4. forest",
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

        fold_data.append((
            (np.concatenate(X_train), np.concatenate(y_train)),
            (np.concatenate(X_val), np.concatenate(y_val))
        ))
    return fold_data

def rf_between_model(approach: str):
    if approach not in approach_output_dirs:
        raise ValueError("Only approaches 5 or 6 are allowed.")

    seed_everything(42)
    results = []

    output_dir = os.path.join(ROOT_DIR, approach_output_dirs[approach])
    os.makedirs(output_dir, exist_ok=True)

    all_data = load_between_model(approach)

    param_grid = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [None, 10, 20],
        "rf__min_samples_split": [2, 5],
    }

    for test_subject in subjects:
        print(f"\nProcessing fold - Test subject: {test_subject}")
        train_subjects = [s for s in subjects if s != test_subject]
        folds = group_subjects(train_subjects, n_folds=5)
        cv_data = get_kfold_data(folds, all_data)

        best_score = -np.inf
        best_params = None

        for values in product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), values))
            accs = []

            for (X_train, y_train), (X_val, y_val) in cv_data:
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("rf", RandomForestClassifier(
                        n_estimators=params["rf__n_estimators"],
                        max_depth=params["rf__max_depth"],
                        min_samples_split=params["rf__min_samples_split"],
                        random_state=42
                    ))
                ])
                pipeline.fit(X_train, y_train)
                acc = accuracy_score(y_val, pipeline.predict(X_val))
                accs.append(acc)

            mean_acc = np.mean(accs)
            if mean_acc > best_score:
                best_score = mean_acc
                best_params = params

        # Final model training with best parameters
        X_train_all, y_train_all = [], []
        for subj in train_subjects:
            X, y = all_data[subj]
            X_train_all.append(X.reshape(X.shape[0], -1))
            y_train_all.append(y)

        X_train_all = np.concatenate(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        best_model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=best_params["rf__n_estimators"],
                max_depth=best_params["rf__max_depth"],
                min_samples_split=best_params["rf__min_samples_split"],
                random_state=42
            ))
        ])
        best_model.fit(X_train_all, y_train_all)

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

        print(f"Fold complete. Accuracy: {acc:.2f}%, F1: {f1:.2f}%")
        joblib.dump(best_model, os.path.join(output_dir, f"{test_subject}_model.pkl"))

        # Save feature importances for multiple-characteristics approach
        if approach == "6":
            rf = best_model.named_steps["rf"]
            importances = rf.feature_importances_
            df_feat = pd.DataFrame({
                "feature_index": np.arange(len(importances)),
                "importance": importances
            }).sort_values("importance", ascending=False)

            df_feat.to_csv(os.path.join(output_dir, f"{test_subject}_feature_importances.csv"), index=False)

            # Save top 20 feature plot
            top_feats = df_feat.head(20)
            plt.figure(figsize=(12, 8))
            plt.barh(top_feats["feature_index"][::-1], top_feats["importance"][::-1])
            plt.xlabel("Importance")
            plt.title(f"Top 20 Features - {test_subject}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{test_subject}_top20_features.png"))
            plt.close()

    df_results = pd.DataFrame(results)

    # Compute overall metrics
    acc_mean = df_results["accuracy"].mean()
    acc_std = df_results["accuracy"].std()
    f1_mean = df_results["f1_score"].mean()
    f1_std = df_results["f1_score"].std()

    summary_rows = pd.DataFrame([
        {
            "test_subject": "MEAN",
            "accuracy": acc_mean,
            "f1_score": f1_mean,
            "rf__n_estimators": "",
            "rf__max_depth": "",
            "rf__min_samples_split": ""
        },
        {
            "test_subject": "STD",
            "accuracy": acc_std,
            "f1_score": f1_std,
            "rf__n_estimators": "",
            "rf__max_depth": "",
            "rf__min_samples_split": ""
        }
    ])

    df_results = pd.concat([df_results, summary_rows], ignore_index=True)
    csv_path = os.path.join(output_dir, "rf_loso_results.csv")
    df_results.to_csv(csv_path, index=False)

    print("\n Final Results")
    print(f"Accuracy: {acc_mean:.2f}% ± {acc_std:.2f}%")
    print(f"F1 Score: {f1_mean:.2f}% ± {f1_std:.2f}%")
    print(f"Results saved to: {csv_path}")
