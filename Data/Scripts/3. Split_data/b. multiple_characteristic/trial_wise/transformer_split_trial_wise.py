"""
Trial-wise Train/Test Split for Grid-Based Multi-Feature EEG (CNN, ViT)

This script performs a trial-wise train/test split for EEG data that has been spatially
transformed into 2D topographic grids using `ToGrid`, and enriched with multiple features:
- Band Differential Entropy
- Power Spectral Density
- Skewness
- Kurtosis
- Sample Entropy
- Hjorth Parameters

Each feature is computed across 4 frequency bands, resulting in 24 channels per segment 
(6 features × 4 bands = 24). Each channel corresponds to a (9 × 9) grid layout, producing
an input tensor of shape (24, 9, 9) per EEG segment. Each subject contributes 1800 segments 
(30 trials × 60 slices).

The train/test split is performed at the **trial level**, maintaining the integrity of trials
(i.e., all 60 slices from a given trial go to the same split). After the split, trials are 
flattened back into slice-wise format.

Use case: Deep learning models operating on spatial EEG representations (e.g., CNN, ViT).

Input:
- all_data_grid.pkl → shape: (1800, 24, 9, 9)
- Labels → shape: (1800,)

Output:
- train_test_split_trial_wise_grid.pkl per subject with:
    - X_train, X_test: (n_samples, 24, 9, 9)
    - y_train, y_test: (n_samples,)
"""

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

# Path to grid-based multi-feature EEG data
input_dir = "Data/Splits/toGrid_transforms/multiple_characteristics"

# Output path for trial-wise train/test split
output_dir = "Data/Splits/toGrid_split_train_test/trial_wise/multiple_characteristics"
os.makedirs(output_dir, exist_ok=True)

# List of subjects to process
subjects = [
    "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
    "sub11", "sub12", "sub13", "sub14", "sub15", "sub16"
]

# Each trial consists of 60 1-second EEG segments
slices_per_trial = 60

for subject in subjects:
    print(f"Processing {subject}...")

    input_pkl = os.path.join(input_dir, subject, "all_data_grid.pkl")
    if not os.path.exists(input_pkl):
        print(f"  Skipped: {input_pkl} not found.")
        continue

    # Load data: X shape (1800, 24, 9, 9), y shape (1800,)
    X_all, y_all = joblib.load(input_pkl)
    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # Group into 30 trials of 60 segments
    n_trials = len(X_all) // slices_per_trial
    X_trials = [X_all[i * slices_per_trial:(i + 1) * slices_per_trial] for i in range(n_trials)]
    y_trials = [y_all[i * slices_per_trial] for i in range(n_trials)]  # One label per trial

    # Stratified trial-wise split
    X_train_blocks, X_test_blocks, y_train_blocks, y_test_blocks = train_test_split(
        X_trials, y_trials, test_size=0.2, stratify=y_trials, random_state=42
    )

    # Flatten trials back into slices
    X_train = np.concatenate(X_train_blocks, axis=0)
    X_test = np.concatenate(X_test_blocks, axis=0)
    y_train = np.concatenate([[label] * slices_per_trial for label in y_train_blocks])
    y_test = np.concatenate([[label] * slices_per_trial for label in y_test_blocks])

    # Save result
    subject_out_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_out_dir, exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test),
                os.path.join(subject_out_dir, "train_test_split_trial_wise_grid.pkl"))

    print(f"  Saved to {subject_out_dir}/train_test_split_trial_wise_grid.pkl\n")
