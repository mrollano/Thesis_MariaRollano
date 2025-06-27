"""
EEG Data Loader for Subjects 04–16

This script loads preprocessed EEG data from multiple subjects stored in EEGLAB `.set` files.
Each subject has 30 epochs (each 60 seconds), and each epoch is segmented into 60 cropped 
segments of 1 second (250 samples). This results in a total of 1800 segments per subject, each 
with shape (64 channels, 250 timepoints). The script also assigns binary trust labels 
(1 = trust, 0 = no trust) to each 1-second segment. 

Functions:
- load_eeg_data(): loads and reshapes the EEG epochs.
- get_subject_data(): returns the EEG data and labels for a specific subject.
"""

import numpy as np
import mne
import os

def load_eeg_data(file_path):
    epochs = mne.io.read_epochs_eeglab(file_path)
    X = []

    for epoch in epochs:
        data = epoch[:, 750:]  # Remove baseline
        data = data.reshape((64, 60, 250)).transpose(1, 0, 2)  # Convert to shape (60, 64, 250)
        X.append(data)

    X = np.concatenate(X, axis=0)  # Final shape: (n_total_trials, 64, 250)
    return X


label1 = np.ones(60, dtype=int)
label0 = np.zeros(60, dtype=int)

# Base directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW_DATA_DIR = os.path.join(ROOT_DIR, "Raw Data")

# =============================== SUBJECT DATA LOADING ===============================

# Use the same pattern as shown here for each subject
X4 = load_eeg_data(os.path.join(RAW_DATA_DIR, "sub04", "baselineremoval_signal04.set"))
y4 = {"trust": np.concatenate([...])}  # See above for full definition

# Subjects 5–16
# (Already loaded correctly in your code — I’ll keep that part untouched to save space here)

# =============================== SUBJECT SELECTOR FUNCTION ===============================

def get_subject_data(subject):
    if subject == "sub04":
        return X4, y4
    elif subject == "sub05":
        return X5, y5
    elif subject == "sub07":
        return X7, y7
    elif subject == "sub08":
        return X8, y8
    elif subject == "sub09":
        return X9, y9
    elif subject == "sub10":
        return X10, y10
    elif subject == "sub11":
        return X11, y11
    elif subject == "sub12":
        return X12, y12
    elif subject == "sub13":
        return X13, y13
    elif subject == "sub14":
        return X14, y14
    elif subject == "sub15":
        return X15, y15
    elif subject == "sub16":
        return X16, y16
    else:
        raise ValueError(f"No subject: {subject}")
