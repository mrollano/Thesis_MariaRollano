"""
Trial-wise Train/Test Split for One-Characteristic EEG Features (KNN, SVM, NB, RF, XGBoost)

This script performs a trial-wise split of EEG data previously transformed into flat feature vectors.
Each subject has 1800 1-second segments, grouped into 30 trials of 60 slices each. The data is first 
split at the trial level using sklearn's `train_test_split`, preserving label distribution, and then 
flattened back into individual slices.

This approach ensures that all slices from the same trial remain together in either the training or 
test set, avoiding data leakage.

- Input: all_data.pkl → shape (1800 segments per subject, each as flat vector)
- Output: train_test_split_trial_wise.pkl per subject with (X_train, X_test, y_train, y_test)

Use case: Classical ML models (KNN, SVM, Naive Bayes, Random Forest, XGBoost)
"""

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

# Path to the transformed flat feature data
input_dir = "Data/Splits/transforms/one_characteristic"

# Path where the trial-wise train/test splits will be saved
output_dir = "Data/Splits/split_train_test/trial_wise/one_characteristic"
os.makedirs(output_dir, exist_ok=True)

# List of subjects to process
subjects = [
    "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
    "sub11", "sub12", "sub13", "sub14", "sub15", "sub16"
]

# Each trial consists of 60 consecutive 1-second slices
slices_per_trial = 60

for subject in subjects:
    print(f"Processing {subject}...")

    input_pkl = os.path.join(input_dir, subject, "all_data.pkl")
    if not os.path.exists(input_pkl):
        print(f"  Skipped: {input_pkl} not found.")
        continue

    # Load all EEG segments and corresponding labels
    X_all, y_all = joblib.load(input_pkl)
    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # Group the 1800 slices into 30 trials of 60 slices each
    n_trials = len(X_all) // slices_per_trial
    X_trials = [X_all[i * slices_per_trial:(i + 1) * slices_per_trial] for i in range(n_trials)]
    y_trials = [y_all[i * slices_per_trial] for i in range(n_trials)]  # One label per trial

    # Perform trial-wise train/test split (stratified by trial label)
    X_train_blocks, X_test_blocks, y_train_blocks, y_test_blocks = train_test_split(
        X_trials, y_trials, test_size=0.2, stratify=y_trials, random_state=42
    )

    # Flatten back to individual slices
    X_train = np.concatenate(X_train_blocks, axis=0)
    X_test = np.concatenate(X_test_blocks, axis=0)
    y_train = np.concatenate([[label] * slices_per_trial for label in y_train_blocks])
    y_test = np.concatenate([[label] * slices_per_trial for label in y_test_blocks])

    # Save trial-wise split
    subject_out_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_out_dir, exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test),
                os.path.join(subject_out_dir, "train_test_split_trial_wise.pkl"))

    print(f"  Saved trial-wise split to {subject_out_dir}/train_test_split_trial_wise.pkl\n")
