import os
import joblib
import numpy as np

def load_between_data_grid(approach: str):
    """
    Loads between-subject EEG data for grid-based features (CNN/ViT models).

    This function is used for approaches:
        - 5: One characteristic (grid format)
        - 6: Multiple characteristics (grid format)

    Each subject's file contains:
        - X: NumPy array of shape (1800, n_channels, 9, 9)
        - y: NumPy array of shape (1800,)

    Returns:
        dict: {
            subject_id: (X, y)
        }

    Raises:
        ValueError: If an invalid approach is passed.
    """

    ROOT_DIR = "/home/jovyan/Final_TFM"

    base_paths = {
        "5": "Data/Splits/toGrid_transforms/one_characteristic",
        "6": "Data/Splits/toGrid_transforms/multiple_characteristics",
    }

    if approach not in base_paths:
        raise ValueError(f"Invalid approach '{approach}'. Use '5' or '6'.")

    base_path = os.path.join(ROOT_DIR, base_paths[approach])

    subjects = [
        "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
        "sub11", "sub12", "sub13", "sub14", "sub15", "sub16",
    ]

    all_data = {}

    for subject in subjects:
        pkl_path = os.path.join(base_path, subject, "all_data_grid.pkl")

        if not os.path.exists(pkl_path):
            print(f" File not found: {pkl_path}")
            continue

        try:
            X, y = joblib.load(pkl_path)
            X = np.array(X)
            y = np.array(y)
            all_data[subject] = (X, y)
            print(f" Loaded {subject}: X shape = {X.shape}, y shape = {y.shape}")
        except Exception as e:
            print(f" Error loading {subject}: {e}")

    return all_data
