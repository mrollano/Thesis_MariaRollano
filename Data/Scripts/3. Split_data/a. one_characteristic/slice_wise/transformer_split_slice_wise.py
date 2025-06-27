"""
Slice-wise Train/Test Split for Grid-Transformed EEG Data (CNN/ViT)

This script performs a slice-wise split of EEG segments that have been transformed into 
spatial grids using `ToGrid`. Each grid corresponds to a 1-second EEG segment with 
multiple feature channels. The data is randomly split using `train_test_split`, 
preserving label distribution.

Use case: CNNs and Transformers that require 2D input.

- Input: all_data_grid.pkl → shape (1800, 24, 9, 9)
- Output: train_test_split_grid.pkl per subject with (X_train, X_test, y_train, y_test)
"""

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

# Input: Grid-transformed data
input_dir = "Data/Splits/toGrid_transforms/one_characteristic"

# Output: Split data for spatial models
output_dir = "Data/Splits/toGrid_split_train_test/slice_wise/one_characteristic"
os.makedirs(output_dir, exist_ok=True)

# Subjects
subjects = [
    "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
    "sub11", "sub12", "sub13", "sub14", "sub15", "sub16"
]

for subject in subjects:
    print(f"Processing {subject}...")

    input_pkl = os.path.join(input_dir, subject, "all_data_grid.pkl")
    if not os.path.exists(input_pkl):
        print(f"  Skipped: {input_pkl} not found.")
        continue

    # Load (X, y) where X: (1800, 24, 9, 9), y: (1800,)
    X_all, y_all = joblib.load(input_pkl)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    # Save split
    subject_out_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_out_dir, exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test),
                os.path.join(subject_out_dir, "train_test_split_grid.pkl"))

    print(f"  Saved split to {subject_out_dir}/train_test_split_grid.pkl")
