"""
Trial-wise Train/Test Split for Multi-Feature EEG Vectors (KNN, SVM, NB, RF, XGBoost)

This script performs a trial-wise train/test split on EEG data that has been transformed
into flat feature vectors by extracting multiple characteristics:
- Band Differential Entropy
- Power Spectral Density
- Skewness
- Kurtosis
- Sample Entropy
- Hjorth Parameters

Each of the 6 features is computed over 4 frequency bands, resulting in 24 values per segment
(6 features × 4 bands = 24-dimensional flat vector). Each subject provides 1800 EEG segments,
organized into 30 trials of 60 slices each.

The data is split **at the trial level**, preserving label distribution (stratified). All 60 
segments from the same trial are assigned to the same split (train or test), preventing data 
leakage. Then, the trials are flattened back to slice-wise format.

Use case: Classical machine learning models (e.g., KNN, SVM, Naive Bayes, Random Forest, XGBoost)

Input:
- all_data.pkl → shape: (1800, 24)
- Labels → shape: (1800,)

Output:
- train_test_split_trial_wise.pkl per subject with:
    - X_train, X_test: (n_samples, 24)
    - y_train, y_test: (n_samples,)
"""

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

# Path to transformed multi-feature EEG vectors
input_dir = "Data/Splits/transforms/multiple_characteristics"

# Output path for trial-wise splits
output_dir = "Data/Splits/split_train_test/trial_wise/multiple_characteristic"
os.makedirs(output_dir, exist_ok=True)

# Subjects to process
subjects = [
    "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
    "sub11", "sub12", "sub13", "sub14", "sub15", "sub16"
]

# Each trial consists of 60 1-second EEG segments
slices_per_trial = 60

for subject in subjects:
    print(f"Processing {subject}...")

    input_pkl = os.path.join(input_dir, subject, "all_data.pkl")
    if not os.path.exists(input_pkl):
        print(f"  Skipped: {input_pkl} not found.")
        continue

    # Load transformed EEG data: shape (1800, 24)
    X_all, y_all = joblib.load(input_pkl)
    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # Group into 30 trials of 60 slices each
    n_trials = len(X_all) // slices_per_trial
    X_trials = [X_all[i * slices_per_trial:(i + 1) * slices_per_trial] for i in range(n_trials)]
    y_trials = [y_all[i * slices_per_trial] for i in range(n_trials)]  # One label per trial

    # Stratified trial-wise split
    X_train_blocks, X_test_blocks, y_train_blocks, y_test_blocks = train_test_split(
        X_trials, y_trials, test_size=0.2, stratify=y_trials, random_state=42
    )

    # Flatten trial blocks back to slice-wise
    X_train = np.concatenate(X_train_blocks, axis=0)
    X_test = np.concatenate(X_test_blocks, axis=0)
    y_train = np.concatenate([[label] * slices_per_trial for label in y_train_blocks])
    y_test = np.concatenate([[label] * slices_per_trial for label in y_test_blocks])

    # Save the result
    subject_out_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_out_dir, exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test),
                os.path.join(subject_out_dir, "train_test_split_trial_wise.pkl"))

    print(f"  Saved trial-wise split to {subject_out_dir}/train_test_split_trial_wise.pkl")
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"  Label dist (train): {np.bincount(y_train)}")
    print(f"  Label dist (test): {np.bincount(y_test)}\n")
