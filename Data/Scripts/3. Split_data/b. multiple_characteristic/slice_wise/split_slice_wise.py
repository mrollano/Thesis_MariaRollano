"""
Slice-wise Train/Test Split for Multi-Feature EEG Vectors (KNN, SVM, NB, RF, XGBoost)

This script performs a slice-wise train/test split for EEG segments that have been 
transformed into flat feature vectors combining multiple characteristics:
- Band Differential Entropy
- Power Spectral Density
- Skewness
- Kurtosis
- Sample Entropy
- Hjorth Parameters

Each feature is computed across 4 frequency bands, resulting in 24 values per segment
(6 features × 4 bands = 24-dimensional vector). Each subject provides 1800 1-second segments.

The split is performed at the slice level using `train_test_split`, stratified by label,
without preserving trial grouping.

Use case: Classical machine learning models (e.g., KNN, SVM, Naive Bayes, Random Forest, XGBoost).

Input:
- all_data.pkl → shape: (1800, 24)
- Labels → shape: (1800,)

Output:
- train_test_split.pkl per subject with:
    - X_train, X_test: (n_samples, 24)
    - y_train, y_test: (n_samples,)
"""

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

# Path to transformed multi-feature flat vectors
input_dir = "Data/Splits/transforms/multiple_characteristics"

# Output path for slice-wise split
output_dir = "Data/Splits/split_train_test/slice_wise/multiple_characteristic"
os.makedirs(output_dir, exist_ok=True)

# Subjects to process
subjects = [
    "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
    "sub11", "sub12", "sub13", "sub14", "sub15", "sub16"
]

for subject in subjects:
    print(f"Processing {subject}...")

    input_pkl = os.path.join(input_dir, subject, "all_data.pkl")
    if not os.path.exists(input_pkl):
        print(f"  Skipped: {input_pkl} not found.")
        continue

    # Load (X, y): X is (1800, 24), y is (1800,)
    X_all, y_all = joblib.load(input_pkl)

    # Perform stratified slice-wise split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    # Save split
    subject_out_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_out_dir, exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test),
                os.path.join(subject_out_dir, "train_test_split.pkl"))

    print(f"  Saved split to {subject_out_dir}/train_test_split.pkl")
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"  Label dist (train): {np.bincount(y_train)}")
    print(f"  Label dist (test): {np.bincount(y_test)}")
