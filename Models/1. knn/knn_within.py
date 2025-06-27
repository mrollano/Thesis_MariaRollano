

"""
K-Nearest Neighbors (KNN) - Within-Subject Evaluation

This script trains and evaluates a KNN classifier using a within-subject approach.
Each subject's data is split into training and test sets, and a separate model is
trained for each subject. A grid search with cross-validation is used to find the 
best hyperparameters per subject, and a final model is also trained using the most 
common hyperparameter configuration across all subjects.

Supported approaches (1–4) correspond to:
- 1: One characteristic – slice-wise
- 2: One characteristic – trial-wise
- 3: Multiple characteristics – slice-wise
- 4: Multiple characteristics – trial-wise

Input:
- Precomputed feature splits per subject (from load_within_subject_data)
- Shape: (n_samples, n_features) after flattening

Output (per subject):
- Best model per subject (`*_best_knn.pkl`)
- Final model with common params (`*_final_model_common.pkl`)
- `knn_hyperparams_by_subject.csv` – best params + CV accuracy
- `knn_final_metrics_common_params.csv` – test accuracy and F1 score
"""

import os
import sys
import joblib
import random
import torch
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

# ========== SETUP ==========
ROOT_DIR = "/home/jovyan/Final_TFM"
sys.path.append(os.path.join(ROOT_DIR, "Data", "Scripts", "4. Load"))
from load_within import load_within_subject_data

# Output folders for each within-subject approach
approach_output_dirs = {
    "1": "Results/1. with_in_subject/a. one_characteristic/slice_wise/1. knn",
    "2": "Results/1. with_in_subject/a. one_characteristic/trial_wise/1. knn",
    "3": "Results/1. with_in_subject/b. multiple_characteristic/slice_wise/1. knn",
    "4": "Results/1. with_in_subject/b. multiple_characteristic/trial_wise/1. knn",
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========== MAIN FUNCTION ==========
def knn_within_model(approach: str):
    if approach not in approach_output_dirs:
        raise ValueError("This script only supports approaches 1, 2, 3, and 4.")

    seed_everything(42)

    # Load within-subject data
    all_data = load_within_subject_data(approach)

    output_dir = os.path.join(ROOT_DIR, approach_output_dirs[approach])
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to: {output_dir}")

    # Grid search hyperparameters
    param_grid = {
        "knn__n_neighbors": [3, 5, 7, 9],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }

    results = []
    param_counts = []

    # === Hyperparameter tuning per subject ===
    for subject, (X_train, _, y_train, _) in all_data.items():
        print(f"\nTuning hyperparameters for {subject}...")

        X_train_flat = np.array([x.flatten() for x in X_train])

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier())
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
        print(f"{subject} → Best Params: {best_params}")

        joblib.dump(
            grid_search.best_estimator_,
            os.path.join(output_dir, f"{subject}_best_knn.pkl"),
        )

        results.append({"subject": subject, **best_params, "cv_score": grid_search.best_score_})
        param_counts.append((
            best_params["knn__n_neighbors"],
            best_params["knn__weights"],
            best_params["knn__metric"]
        ))

    # Save best hyperparameters per subject
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "knn_hyperparams_by_subject.csv"), index=False)

    # ========== FINAL MODEL TRAINING ==========
    most_common = Counter(param_counts).most_common(1)[0][0]
    print("\nTraining final model with common hyperparameters...")
    print(f"→ Using: n_neighbors={most_common[0]}, weights={most_common[1]}, metric={most_common[2]}")

    final_results = []

    for subject, (X_train, X_test, y_train, y_test) in all_data.items():
        print(f"\nEvaluating {subject} with common model...")

        X_train_flat = np.array([x.flatten() for x in X_train])
        X_test_flat = np.array([x.flatten() for x in X_test])

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=most_common[0],
                weights=most_common[1],
                metric=most_common[2],
            )),
        ])

        model.fit(X_train_flat, y_train)
        y_pred = model.predict(X_test_flat)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        joblib.dump(
            model, os.path.join(output_dir, f"{subject}_final_model_common.pkl")
        )

        final_results.append({
            "subject": subject,
            "final_accuracy": acc * 100,
            "final_f1_score": f1 * 100
        })

        print(f"{subject} → Accuracy: {acc*100:.2f}%, F1 Score: {f1*100:.2f}")

    # Save evaluation results
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
            "final_f1_score": f"{mean_f1:.2f} ± {std_f1:.2f}",
        }])
    ], ignore_index=True)

    df_final.to_csv(os.path.join(output_dir, "knn_final_metrics_common_params.csv"), index=False)

    print("\nFinal evaluation completed.")
    print(f" Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f" F1 Score: {mean_f1:.2f}% ± {std_f1:.2f}%")
