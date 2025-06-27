"""
Slice-wise Train/Test Split for Grid-Based Multi-Feature EEG (CNN, ViT)

This script performs a slice-wise train/test split on EEG data that has been 
transformed into 2D spatial grids using `ToGrid`, and enriched with multiple features:
- Band Differential Entropy
- Power Spectral Density
- Skewness
- Kurtosis
- Sample Entropy
- Hjorth Parameters

Each feature is computed across 4 frequency bands. The resulting grid for each segment 
is a stacked tensor of shape (24, 9, 9), where:
- 6 features × 4 bands = 24 channels
- Each channel is mapped to a 2D (9 × 9) topographic layout

This split is performed at the individual slice level (1-second EEG segment), using
stratified random sampling. Temporal relationships between segments are ignored.

Use case: Deep learning models based on spatial representations:
- Convolutional Neural Networks (CNN)
- Vision Transformers (ViT)

Input:
- all_data_grid.pkl → shape: (1800, 24, 9, 9)
- Labels → shape: (1800,)

Output:
- train_test_split_grid.pkl per subject with:
    - X_train, X_test: (n_samples, 24, 9, 9)
    - y_train, y_test: (n_samples,)
"""

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

# Path to transformed grid-based multi-feature EEG data
input_dir = "Data/Splits/toGrid_transforms/multiple_characteristics"

# Output directory for slice-wise split
output_dir = "Data/Splits/toGrid_split_train_test/slice_wise/multiple_characteristic"
os.makedirs(output_dir, exist_ok=True)

# Subjects to process
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

    # Load transformed data: X shape (1800, 24, 9, 9), y shape (1800,)
    X_all, y_all = joblib.load(input_pkl)

    # Slice-wise stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    # Save the split data
    subject_out_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_out_dir, exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test),
                os.path.join(subject_out_dir, "train_test_split_grid.pkl"))

    print(f"  Saved split to {subject_out_dir}/train_test_split_grid.pkl")
