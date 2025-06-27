import os
import joblib
import numpy as np

def load_between_model(approach: str):
    """
    Loads between-subject EEG data for approaches 5 and 6.

    These approaches correspond to training models across subjects:
    - Approach 5: One characteristic (flat features)
    - Approach 6: Multiple characteristics (flat features)

    Each subject's data is loaded from a single .pkl file containing:
    - X: NumPy array of shape (1800, n_features)
    - y: NumPy array of shape (1800,)

    Returns:
        dict: {
            subject_id: (X, y)
        }

    Raises:
        ValueError: if approach is not '5' or '6'.
    """

    ROOT_DIR = "/home/jovyan/Final_TFM"

    base_paths = {
        "5": "Data/Splits/transforms/one_characteristic",
        "6": "Data/Splits/transforms/multiple_characteristics",
    }

    if approach not in base_paths:
        raise ValueError(f"Invalid approach '{approach}'. Choose '5' or '6'.")

    base_path = os.path.join(ROOT_DIR, base_paths[approach])

    subjects = [
        "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
        "sub11", "sub12", "sub13", "sub14", "sub15", "sub16",
    ]

    all_data = {}

    for subject in subjects:
        pkl_path = os.path.join(base_path, subject, "all_data.pkl")
        print(f"\n→ Loading from: {pkl_path}")

        if not os.path.exists(pkl_path):
            print(f"   Not found: {pkl_path}")
            continue

        try:
            X, y = joblib.load(pkl_path)
            X = np.array(X)
            y = np.array(y)
            all_data[subject] = (X, y)

            uniques, counts = np.unique(y, return_counts=True)
            print(f"   Loaded {subject} — X: {X.shape}, y: {y.shape}")
            print(f"    Labels: {dict(zip(uniques, counts))}")

        except Exception as e:
            print(f"  Error loading {subject}: {e}")

    return all_data
