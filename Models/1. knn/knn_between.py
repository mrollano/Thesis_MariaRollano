"""   
K-Nearest Neighbors (KNN) - Between-Subject Evaluation (LOSO)

This script performs a Leave-One-Subject-Out (LOSO) cross-validation using a KNN classifier.
For each subject, a model is trained on the data from the remaining subjects and evaluated on the left-out subject.
A grid search is conducted on the training subjects via 5-fold cross-validation to find the best hyperparameters.
The final model is trained on all training subjects using the best parameters, and evaluated on the test subject.

Supported approaches:
- 5: One characteristic – between-subject
- 6: Multiple characteristics – between-subject

Input:
- Precomputed data per subject (from load_between_model)
- Shape: (n_samples, n_features) after flattening

Output:
- Trained model per subject (`*_model.pkl`)
- CSV file summarizing accuracy, F1-score and hyperparameters for each test subject
  (`knn_loso_results.csv`), including global mean and std.

Use case: Between-subject generalization with classical ML (KNN)
"""

import os
import sys
import numpy as np
import joblib
import random
import pandas as pd
from itertools import product
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load module
sys.path.insert(0, "/home/jovyan/Final_TFM/Data/Scripts/4. Load")
from load_between import load_between_model

# ========== Configuration ==========
ROOT_DIR = "/home/jovyan/Final_TFM"

subjects = [
    "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
    "sub11", "sub12", "sub13", "sub14", "sub15", "sub16"
]

# Output folders per approach
approach_output_dirs = {
    "5": "Results/2. between_subject/a. one characteristic/1. knn",
    "6": "Results/2. between_subject/b. multiple characteristic/1. knn",
}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ========== Fold grouping ==========
def group_subjects(subjects, n_folds):
    shuffled = subjects.copy()
    random.shuffle(shuffled)
    folds = [[] for _ in range(n_folds)]
    for i, subj in enumerate(shuffled):
        folds[i % n_folds].append(subj)
    return folds

# ========== Prepare training and validation data ==========
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

        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        X_val = np.concatenate(X_val)
        y_val = np.concatenate(y_val)

        fold_data.append(((X_train, y_train), (X_val, y_val)))
    return fold_data

# ========== Main function ==========
def knn_between_model(approach: str):
    if approach not in approach_output_dirs:
        raise ValueError("This script only supports approaches 5 or 6.")

    seed_everything(42)
    results = []

    output_dir = os.path.join(ROOT_DIR, approach_output_dirs[approach])
    os.makedirs(output_dir, exist_ok=True)

    all_data = load_between_model(approach)

    for test_subject in subjects:
        print(f"\n Fold - Test subject: {test_subject}")
        train_subjects = [s for s in subjects if s != test_subject]

        folds = group_subjects(train_subjects, n_folds=5)
        cv_data = get_kfold_data(folds, all_data)

        param_grid = {
            "knn__n_neighbors": [3, 5, 7],
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan"],
        }

        best_score = -np.inf
        best_params = None

        for values in product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), values))
            accs = []

            for (X_train, y_train), (X_val, y_val) in cv_data:
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(
                        n_neighbors=params["knn__n_neighbors"],
                        weights=params["knn__weights"],
                        metric=params["knn__metric"]
                    ))
                ])
                pipeline.fit(X_train, y_train)
                acc = pipeline.score(X_val, y_val)
                accs.append(acc)

            mean_acc = np.mean(accs)
            if mean_acc > best_score:
                best_score = mean_acc
                best_params = params

        # Final model training
        X_train_all, y_train_all = [], []
        for subj in train_subjects:
            X, y = all_data[subj]
            X_train_all.append(X.reshape(X.shape[0], -1))
            y_train_all.append(y)

        X_train_all = np.concatenate(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        best_model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(**{
                k.replace("knn__", ""): v for k, v in best_params.items()
            }))
        ])
        best_model.fit(X_train_all, y_train_all)

        # Evaluate
        X_test, y_test = all_data[test_subject]
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        y_pred = best_model.predict(X_test_flat)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "test_subject": test_subject,
            "accuracy": acc * 100,
            "f1_score": f1 * 100,
            **best_params
        })

        print(f" Fold done. Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%")

        model_path = os.path.join(output_dir, f"{test_subject}_model.pkl")
        joblib.dump(best_model, model_path)

    # Save results
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
            "knn__n_neighbors": "",
            "knn__weights": "",
            "knn__metric": ""
        },
        {
            "test_subject": "STD",
            "accuracy": acc_std,
            "f1_score": f1_std,
            "knn__n_neighbors": "",
            "knn__weights": "",
            "knn__metric": ""
        }
    ])

    df_results = pd.concat([df_results, summary_rows], ignore_index=True)

    csv_path = os.path.join(output_dir, "knn_loso_results.csv")
    df_results.to_csv(csv_path, index=False)

    print(f"\n FINAL RESULTS (Approach {approach})")
    print(f" Accuracy: {acc_mean:.2f}% ± {acc_std:.2f}%")
    print(f" F1-score: {f1_mean:.2f} ± {f1_std:.2f}")
    print(f" Results saved to: {csv_path}")
