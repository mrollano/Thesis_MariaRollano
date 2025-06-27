"""
BDE Feature Extraction from EEG Segments (Subjects 04–16)

This script loads segmented EEG data for selected subjects using the `get_subject_data()` function.
For each 1-second EEG segment (shape: 64 channels × 250 timepoints), it computes 4 Band Differential Entropy (BDE)
features per segment using the `torcheeg` library. The features are saved as NumPy-based datasets 
in .pkl files, one per subject.

Steps:
- Load EEG segments and trust labels.
- Apply BandDifferentialEntropy transform (offline) to each segment.
- Store transformed features and labels using joblib.

Output:
- Final feature-label pairs saved to: Data/Splits/transforms/one_characteristic/{subject}/all_data.pkl
"""

import joblib
import shutil
import numpy as np
import sys
import os

# Add project path to sys.path
sys.path.insert(0, "/home/jovyan/Final_TFM")
from torcheeg.datasets.module.numpy_dataset import NumpyDataset
from torcheeg import transforms
from torcheeg.transforms import BandDifferentialEntropy, Select

# Import load function
sys.path.insert(0, "/home/jovyan/Final_TFM/Data/Scripts/1. Load_data")
from load_data import get_subject_data

# List of subjects to process
subjects = [
    "sub04", "sub05", "sub07", "sub08", "sub09",
    "sub10", "sub11", "sub12", "sub13", "sub14",
    "sub15", "sub16",
]

# Define output directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
output_dir = os.path.join(ROOT_DIR, "Final_TFM", "Data", "Splits", "transforms", "one_characteristic")
os.makedirs(output_dir, exist_ok=True)

# Process each subject
for subject in subjects:
    print(f"Processing subject {subject}...")

    # Load raw EEG segments and labels
    X, y = get_subject_data(subject)

    # Create subject-specific output path
    subject_dir = os.path.join(output_dir, subject)
    io_path = os.path.join(subject_dir, "offline_transformations")
    os.makedirs(subject_dir, exist_ok=True)

    # Remove existing transformation data if any
    if os.path.exists(io_path):
        shutil.rmtree(io_path)

    try:
        # Create torcheeg dataset with Band Differential Entropy transform
        dataset = NumpyDataset(
            X=X,
            y=y,
            io_path=io_path,
            offline_transform=transforms.Compose([
                BandDifferentialEntropy(sampling_rate=250)
            ]),
            label_transform=transforms.Compose([Select("trust")]),
            num_worker=1,
            num_samples_per_worker=50,
        )
    except Exception as e:
        print(f"Error processing {subject}: {e}")
        continue

    # Extract and save transformed features and labels
    data = [(x, int(label)) for x, label in dataset]
    X_all, y_all = zip(*data)

    joblib.dump((X_all, y_all), os.path.join(subject_dir, "all_data.pkl"))

    print(f"{subject} processed and saved as all_data.pkl.\n")
