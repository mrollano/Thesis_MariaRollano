"""
Naive Bayes (Between-Subject Evaluation)

This script trains and evaluates a Gaussian Naive Bayes classifier using 
a leave-one-subject-out (LOSO) cross-validation strategy for between-subject settings.

For each fold, one subject is held out for testing while the remaining subjects are used 
for training. Hyperparameter tuning is performed using 5-fold cross-validation on the 
training subjects to select the best `var_smoothing` value. The final model is then trained 
on all training data and evaluated on the held-out subject.

Approaches:
    5. One characteristic – Between-subject
    6. Multiple characteristics – Between-subject

Input:
    - all_data.pkl per subject (flattened features)

Output:
    - Trained model per subject: *_model.pkl
    - Evaluation results: nb_loso_results.csv
"""

import os
import sys
import joblib
import random
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Add loader path
sys.path.insert(0, "/home/jovyan/Final_TFM/Data/Scripts/4. Load")
from load_between import load_between_model

# ========== Configuration ==========
ROOT_DIR = "/home/jovyan/Final_TFM"

subjects = [
    "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
    "sub11", "sub12", "sub13", "sub14", "sub15", "sub16"
]

approach_output_dirs = {
    "5": "Results/2. between_subject/a. one characteristic/2. nb",
    "6": "Results/2. between_subject/b. multiple characteristic/2. nb",
}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def group_subjects(subjects, n_folds):
    """Randomly shuffle subjects and assign to n_folds"""
    shuffled = subjects.copy()
    random.shuffle(shuffled)
    folds = [[] for _ in range(n_folds)]
    for i, subj in enumerate(shuffled):
        folds[i % n_folds].append(subj)
    return folds

def get_kfold_data(folds, all_data):
    """Split data into k-fold training and validation sets (subject-level split)"""
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

def nb_between_model(approach: str):
    if approach not in approach_output_dirs:
        raise ValueError("Only approaches 5 and 6 are supported.")

    seed_everything()
    results = []

    output_dir = os.path.join(ROOT_DIR, approach_output_dirs[approach])
    os.makedirs(output_dir, exist_ok=True)

    all_data = load_between_model(approach)

    for test_subject in subjects:
        print(f"\n Test subject: {test_subject}")
        train_subjects = [s for s in subjects if s != test_subject]
        folds = group_subjects(train_subjects, n_folds=5)
        cv_data = get_kfold_data(folds, all_data)

        var_smoothing_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        best_score = -np.inf
        best_vs = None

        # Cross-validate to select best var_smoothing
        for vs in var_smoothing_values:
            accs = []
            for (X_train, y_train), (X_val, y_val) in cv_data:
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("nb", GaussianNB(var_smoothing=vs))
                ])
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                accs.append(acc)
            mean_acc = np.mean(accs)
            if mean_acc > best_score:
                best_score = mean_acc
                best_vs = vs

        # Final model training on all training subjects
        X_train_all, y_train_all = [], []
        for subj in train_subjects:
            X, y = all_data[subj]
            X_train_all.append(X.reshape(X.shape[0], -1))
            y_train_all.append(y)

        X_train_all = np.concatenate(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        print(f" Best var_smoothing for {test_subject}: {best_vs}")

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("nb", GaussianNB(var_smoothing=best_vs))
        ])
        model.fit(X_train_all, y_train_all)

        # Evaluate on test subject
        X_test, y_test = all_data[test_subject]
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        y_pred = model.predict(X_test_flat)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "test_subject": test_subject,
            "accuracy": acc * 100,
            "f1_score": f1 * 100,
            "var_smoothing": best_vs
        })

        joblib.dump(model, os.path.join(output_dir, f"{test_subject}_model.pkl"))

    # Save results
    df = pd.DataFrame(results)
    acc_mean = df["accuracy"].mean()
    acc_std = df["accuracy"].std()
    f1_mean = df["f1_score"].mean()
    f1_std = df["f1_score"].std()

    df = pd.concat([
        df,
        pd.DataFrame([{
            "test_subject": "MEAN",
            "accuracy": acc_mean,
            "f1_score": f1_mean,
            "var_smoothing": ""
        }, {
            "test_subject": "STD",
            "accuracy": acc_std,
            "f1_score": f1_std,
            "var_smoothing": ""
        }])
    ], ignore_index=True)

    csv_path = os.path.join(output_dir, "nb_loso_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nFinal results (Approach {approach})")
    print(f" Accuracy: {acc_mean:.2f}% ± {acc_std:.2f}%")
    print(f" F1-score: {f1_mean:.2f} ± {f1_std:.2f}")
    print(f" Results saved to: {csv_path}")
