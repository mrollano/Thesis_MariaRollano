"""
Support Vector Machine (Between-Subject Evaluation)

This script evaluates an SVM classifier using leave-one-subject-out (LOSO) cross-validation 
for between-subject generalization with fixed hyperparameters. It supports approaches 5 and 6.

Approaches:
    5. One characteristic – Between-subject
    6. Multiple characteristics – Between-subject

For each subject:
    - The model is trained using data from all other subjects.
    - The trained model is evaluated on the left-out subject.

Input:
    - Transformed data per subject: (X, y)

Output:
    - Trained model per subject: *_model.pkl
    - CSV with accuracy and F1-score per subject: svm_loso_results_fixed.csv
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Load function for between-subject data
sys.path.insert(0, "/home/jovyan/Final_TFM/Data/Scripts/4. Load")
from load_between import load_between_model

# ========== Configuration ==========
ROOT_DIR = "/home/jovyan/Final_TFM"

subjects = [
    "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
    "sub11", "sub12", "sub13", "sub14", "sub15", "sub16"
]

# Output directory mapping per approach
approach_output_dirs = {
    "5": "Results/2. between_subject/a. one characteristic/3. svm",
    "6": "Results/2. between_subject/b. multiple characteristic/3. svm",
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
def svm_fixed_hyperparams(approach: str):
    if approach not in approach_output_dirs:
        raise ValueError("Only approaches 5 and 6 are supported.")

    seed_everything()
    results = []

    output_dir = os.path.join(ROOT_DIR, approach_output_dirs[approach])
    os.makedirs(output_dir, exist_ok=True)

    all_data = load_between_model(approach)

    # Fixed hyperparameters
    fixed_params = {
        "C": 1,
        "kernel": "rbf",
        "gamma": "scale"
    }

    # LOSO evaluation
    for test_subject in subjects:
        print(f"\nTest subject: {test_subject}")
        train_subjects = [s for s in subjects if s != test_subject]

        X_train_all, y_train_all = [], []
        for subj in train_subjects:
            X, y = all_data[subj]
            X_train_all.append(X.reshape(X.shape[0], -1))
            y_train_all.append(y)

        X_train_all = np.concatenate(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=fixed_params["C"],
                kernel=fixed_params["kernel"],
                gamma=fixed_params["gamma"]
            ))
        ])
        model.fit(X_train_all, y_train_all)

        X_test, y_test = all_data[test_subject]
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        y_pred = model.predict(X_test_flat)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "test_subject": test_subject,
            "accuracy": acc * 100,
            "f1_score": f1 * 100,
            **fixed_params
        })

        joblib.dump(model, os.path.join(output_dir, f"{test_subject}_model.pkl"))

    # ========== Save Results ==========
    df = pd.DataFrame(results)

    acc_mean = df["accuracy"].mean()
    acc_std = df["accuracy"].std()
    f1_mean = df["f1_score"].mean()
    f1_std = df["f1_score"].std()

    summary_rows = pd.DataFrame([
        {
            "test_subject": "MEAN",
            "accuracy": acc_mean,
            "f1_score": f1_mean,
            "C": "",
            "kernel": "",
            "gamma": ""
        },
        {
            "test_subject": "STD",
            "accuracy": acc_std,
            "f1_score": f1_std,
            "C": "",
            "kernel": "",
            "gamma": ""
        }
    ])

    df = pd.concat([df, summary_rows], ignore_index=True)
    csv_path = os.path.join(output_dir, "svm_loso_results_fixed.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n Final results (Approach {approach})")
    print(f"Accuracy: {acc_mean:.2f}% ± {acc_std:.2f}%")
    print(f"F1-score: {f1_mean:.2f} ± {f1_std:.2f}")
    print(f"Results saved to: {csv_path}")
