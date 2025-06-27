"""
Trial-wise Train/Test Split for Grid-Based EEG Features (Single Feature - CNN, ViT)

This script performs a trial-wise train/test split for EEG data that has been spatially
transformed into 2D grids using `ToGrid`. Each subject has 1800 1-second segments, each
transformed into a (4, 9, 9) grid — corresponding to a single feature across 4 frequency bands.
These are grouped into 30 trials of 60 grids each (i.e., 60 time slices per trial).

The split is performed at the **trial level**, ensuring that all slices from a given
trial are assigned to the same set (train or test). This prevents data leakage and 
respects the temporal structure of the trials.

This approach is suitable for deep learning models that operate on spatial representations:
- Convolutional Neural Networks (CNN)
- Vision Transformers (ViT)

Input:
- all_data_grid.pkl → shape (1800, 4, 9, 9)

Output:
- train_test_split_trial_wise_grid.pkl per subject, containing:
    - X_train, X_test: (n_slices, 4, 9, 9)
    - y_train, y_test: (n_slices,)
"""

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

# Input directory: EEG data with grid-based features (single feature)
input_dir = "Data/Splits/toGrid_transforms/one_characteristic"

# Output directory for trial-wise splits
output_dir = "Data/Splits/toGrid_split_train_test/trial_wise/one_characteristic"
os.makedirs(output_dir, exist_ok=True)

# Subjects to process
subjects = [
    "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
    "sub11", "sub12", "sub13", "sub14", "sub15", "sub16"
]

# Each trial consists of 60 consecutive 1-second segments (grids)
slices_per_trial = 60

for subject in subjects:
    print(f"Processing {subject}...")

    input_pkl = os.path.join(input_dir, subject, "all_data_grid.pkl")
    if not os.path.exists(input_pkl):
        print(f"  Skipped: {input_pkl} not found.")
        continue

    # Load data: X (1800, 4, 9, 9), y (1800,)
    X_all, y_all = joblib.load(input_pkl)
    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # Group into 30 trials of 60 grids each
    n_trials = len(X_all) // slices_per_trial
    X_trials = [X_all[i * slices_per_trial:(i + 1) * slices_per_trial] for i in range(n_trials)]
    y_trials = [y_all[i * slices_per_trial] for i in range(n_trials)]  # One label per trial

    # Perform stratified trial-wise split
    X_train_blocks, X_test_blocks, y_train_blocks, y_test_blocks = train_test_split(
        X_trials, y_trials, test_size=0.2, stratify=y_trials, random_state=42
    )

    # Flatten back to individual slices
    X_train = np.concatenate(X_train_blocks, axis=0)
    X_test = np.concatenate(X_test_blocks, axis=0)
    y_train = np.concatenate([[label] * slices_per_trial for label in y_train_blocks])
    y_test = np.concatenate([[label] * slices_per_trial for label in y_test_blocks])

    # Save split
    subject_out_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_out_dir, exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test),
                os.path.join(subject_out_dir, "train_test_split_trial_wise_grid.pkl"))

    print(f"  Saved to {subject_out_dir}/train_test_split_trial_wise_grid.pkl\n")
