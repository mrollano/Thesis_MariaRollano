"""
ToGrid Transformation for CNN/ViT Models (One Characteristic - BDE)

This script loads previously extracted Band Differential Entropy (BDE) features per EEG segment
(from each subject) and applies a spatial transformation using `ToGrid()` from `torcheeg`.

This prepares the data for models that require spatial input, such as CNNs or Vision Transformers (ViTs).

- Input: all_data.pkl (1D feature vectors per segment)
- Output: all_data_grid.pkl (2D grid-mapped features per segment)
"""

import sys
import os
import joblib
from torcheeg.transforms import ToGrid
from torcheeg.datasets.constants.emotion_recognition.my_custom_layout import MY_CHANNEL_LOCATION_DICT

# Add root project path 
sys.path.insert(0, "/home/jovyan/Final_TFM") 
print("sys.path[0]:", sys.path[0])

# Define base paths
ROOT_DIR = "/home/jovyan/Final_TFM"
INPUT_PATH = os.path.join(ROOT_DIR, "Data", "Splits", "transforms", "one_characteristic")
OUTPUT_PATH = os.path.join(ROOT_DIR, "Data", "Splits", "toGrid_transforms", "one_characteristic")

# Initialize ToGrid transformation with custom layout
togrid = ToGrid(MY_CHANNEL_LOCATION_DICT)

# Get list of subjects
subjects = os.listdir(INPUT_PATH)

# Process each subject's data
for subject in subjects:
    print(f"\nProcessing subject: {subject}")
    
    subject_input_path = os.path.join(INPUT_PATH, subject, "all_data.pkl")
    subject_output_dir = os.path.join(OUTPUT_PATH, subject)
    os.makedirs(subject_output_dir, exist_ok=True)

    # Skip if original data doesn't exist
    if not os.path.exists(subject_input_path):
        print(f"File does not exist: {subject_input_path}")
        continue

    # Load BDE-transformed data
    X_all, y_all = joblib.load(subject_input_path)

    # Apply ToGrid transformation
    X_all_grid = [togrid(eeg=x) for x in X_all]

    # Save new transformed dataset
    output_path = os.path.join(subject_output_dir, "all_data_grid.pkl")
    joblib.dump((X_all_grid, y_all), output_path)

    print(f"Saved: {output_path}")
