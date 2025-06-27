"""
Slice-wise Train/Test Split for One-Characteristic EEG Features

This script performs a slice-wise data split: each 1-second EEG segment (previously transformed) 
is considered independently. For each subject, segments are shuffled and split into train/test 
sets using sklearn's `train_test_split`, preserving label distribution (stratification).

- Input: all_data.pkl (1800 segments per subject)
- Output: train_test_split.pkl per subject with (X_train, X_test, y_train, y_test)

This strategy ignores the temporal grouping of slices into trials and assumes each segment is independent.
"""

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

# Path to transformed data
input_dir = "Data/Splits/transforms/one_characteristic"

# Output path for slice-wise splits
output_dir = "Data/Splits/split_train_test/slice_wise/one_characteristic"
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

    # Load all slices and labels
    X_all, y_all = joblib.load(input_pkl)

    # Slice-wise random split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    # Save split
    subject_out_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_out_dir, exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test),
                os.path.join(subject_out_dir, "train_test_split.pkl"))

    print(f"  Saved split to {subject_out_dir}/train_test_split.pkl")
