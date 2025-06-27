"""
Multiple Feature Extraction for Classical ML Models (Flat Vectors)

This script processes segmented EEG data per subject and extracts multiple features per segment
using the `torcheeg` library. Each feature is computed per frequency band and flattened
into a single vector. These flat vectors are suitable for classical ML models such as
K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Naive Bayes, Random Forest, and XGBoost.

Features extracted per band:
- Band Differential Entropy (BDE)
- Power Spectral Density (PSD)
- Skewness
- Kurtosis
- Sample Entropy
- Hjorth Parameters

Output:
- Flat feature vectors and binary labels saved as all_data.pkl (one per subject)
"""

import joblib
import shutil
import numpy as np
import sys
import os

# Add project path
sys.path.insert(0, "/home/jovyan/Final_TFM") 

from torcheeg.datasets.module.numpy_dataset import NumpyDataset
from torcheeg import transforms
from torcheeg.transforms import (
    BandDifferentialEntropy,
    BandPowerSpectralDensity,
    BandSkewness,
    BandKurtosis,
    BandSampleEntropy,
    BandHjorth,
    Concatenate,
    Select,
    Flatten,
)

# Load EEG data function
sys.path.insert(0, "/home/jovyan/Final_TFM/Data/Scripts/1. Load_data")
from load_data import get_subject_data

# Subjects to process
subjects = [
    "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
    "sub11", "sub12", "sub13", "sub14", "sub15", "sub16",
]

# Output path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
output_dir = os.path.join(ROOT_DIR, "Final_TFM", "Data", "Splits", "transforms", "multiple_characteristics")
os.makedirs(output_dir, exist_ok=True)

# Loop through subjects
for subject in subjects:
    print(f"Processing subject {subject}...")

    X, y = get_subject_data(subject)

    subject_dir = os.path.join(output_dir, subject)
    io_path = os.path.join(subject_dir, "offline_transformations")
    os.makedirs(subject_dir, exist_ok=True)

    # Remove previous transformations if they exist
    if os.path.exists(io_path):
        shutil.rmtree(io_path)

    try:
        # Apply multiple offline transforms and flatten each
        dataset = NumpyDataset(
            X=X,
            y=y,
            io_path=io_path,
            offline_transform=Concatenate([
                transforms.Compose([BandDifferentialEntropy(sampling_rate=250), Flatten()]),
                transforms.Compose([BandPowerSpectralDensity(sampling_rate=250), Flatten()]),
                transforms.Compose([BandSkewness(sampling_rate=250), Flatten()]),
                transforms.Compose([BandKurtosis(sampling_rate=250), Flatten()]),
                transforms.Compose([BandSampleEntropy(sampling_rate=250), Flatten()]),
                transforms.Compose([BandHjorth(sampling_rate=250), Flatten()]),
            ]),
            label_transform=transforms.Compose([Select("trust")]),
            num_worker=1,
            num_samples_per_worker=50,
        )
    except Exception as e:
        print(f"Error processing {subject}: {e}")
        continue

    # Save processed data
    data = [(x, int(label)) for x, label in dataset]
    X_all, y_all = zip(*data)
    X_all = np.array(X_all)
    y_all = np.array(y_all)

    joblib.dump((X_all, y_all), os.path.join(subject_dir, "all_data.pkl"))

    print(f"{subject} processed and saved as all_data.pkl.")
    print(f"Final dimensions: X = {X_all.shape}, y = {y_all.shape}\n")
