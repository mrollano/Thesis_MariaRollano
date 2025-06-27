"""
ToGrid Transformation for CNN/ViT with Multiple EEG Features

This script extracts multiple features per 1-second EEG segment using torcheeg, 
transforms each feature into a spatial grid representation with `ToGrid`, and concatenates
them to form a comprehensive input for CNNs or Vision Transformers.

Features applied per band:
- Band Differential Entropy
- Power Spectral Density
- Skewness
- Kurtosis
- Sample Entropy
- Hjorth Parameters

Each feature produces 4 values (one per band), resulting in 24 channels in total.
Each channel is mapped to a 9x9 grid using a custom EEG layout.

Output:
- For each subject: a .pkl file containing (X, y)
  - X shape: (1800, 24, 9, 9) → 1800 segments × 24 feature channels × 9×9 spatial grid
  - y shape: (1800,) → binary trust labels per segment
"""

import os
import sys
import joblib
import numpy as np
from tqdm import tqdm

# Add root path
sys.path.insert(0, "/home/jovyan/Final_TFM")

from torcheeg import transforms
from torcheeg.transforms import (
    BandDifferentialEntropy,
    BandPowerSpectralDensity,
    BandSkewness,
    BandKurtosis,
    BandSampleEntropy,
    BandHjorth,
    ToGrid,
)
from torcheeg.datasets.constants.emotion_recognition.my_custom_layout import MY_CHANNEL_LOCATION_DICT

# Import subject data
sys.path.insert(0, "/home/jovyan/Final_TFM/Data/Scripts/1. Load_data")
from load_data import get_subject_data

# Subjects to process
subjects = [
    "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
    "sub11", "sub12", "sub13", "sub14", "sub15", "sub16",
]

# Output directory
output_dir = "/home/jovyan/Final_TFM/Data/Splits/toGrid_transforms/multiple_characteristics"
os.makedirs(output_dir, exist_ok=True)

# Define transforms
transform_list = [
    BandDifferentialEntropy(sampling_rate=250),
    BandPowerSpectralDensity(sampling_rate=250),
    BandSkewness(sampling_rate=250),
    BandKurtosis(sampling_rate=250),
    BandSampleEntropy(sampling_rate=250),
    BandHjorth(sampling_rate=250),
]
to_grid = ToGrid(MY_CHANNEL_LOCATION_DICT)

# Process each subject
for subject in subjects:
    print(f"\nProcessing {subject}...")

    X, y = get_subject_data(subject)
    labels = y["trust"] 

    X_out = []
    y_out = []

    for i in tqdm(range(len(X)), desc=f"{subject}"):
        features = []

        # Apply all transforms and map each to a grid
        for transform in transform_list:
            out = transform(eeg=X[i])
            grid = to_grid(eeg=out["eeg"])["eeg"]
            features.append(grid)

        # Concatenate 6 grids (each with 4 channels) into one tensor of shape (24, 9, 9)
        full_tensor = np.concatenate(features, axis=0)  # Shape: (24, 9, 9)
        X_out.append(full_tensor)
        y_out.append(int(labels[i]))

    # Final arrays: (1800, 24, 9, 9) and (1800,)
    X_out = np.array(X_out, dtype=np.float32)
    y_out = np.array(y_out)

    # Save
    save_path = os.path.join(output_dir, subject)
    os.makedirs(save_path, exist_ok=True)
    joblib.dump((X_out, y_out), os.path.join(save_path, "all_data_grid.pkl"))

    print(f"Saved {subject}: X = {X_out.shape}, y = {y_out.shape}")
