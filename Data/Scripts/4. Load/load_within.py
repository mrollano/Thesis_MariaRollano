import os
import joblib

def load_within_subject_data(approach: str):
    """
    Loads within-subject train/test splits for approaches 1–4.

    These approaches differ in terms of:
    - Number of features: one or multiple
    - Splitting strategy: slice-wise or trial-wise

    Returns:
        dict: {
            subject_id: (X_train, X_test, y_train, y_test)
        }

    Raises:
        ValueError: if an invalid approach number is provided.
    """

    ROOT_DIR = "/home/jovyan/Final_TFM"

    # Path and filename settings for each within-subject approach
    approach_config = {
        "1": {
            "path": "Data/Splits/split_train_test/slice_wise/one_characteristic",
            "filename": "train_test_split.pkl",
            "desc": "One characteristic — Within-subject — Slice-wise",
        },
        "2": {
            "path": "Data/Splits/split_train_test/trial_wise/one_characteristic",
            "filename": "train_test_split_trial_wise.pkl",
            "desc": "One characteristic — Within-subject — Trial-wise",
        },
        "3": {
            "path": "Data/Splits/split_train_test/slice_wise/multiple_characteristic",
            "filename": "train_test_split.pkl",
            "desc": "Multiple characteristics — Within-subject — Slice-wise",
        },
        "4": {
            "path": "Data/Splits/split_train_test/trial_wise/multiple_characteristic",
            "filename": "train_test_split_trial_wise.pkl",
            "desc": "Multiple characteristics — Within-subject — Trial-wise",
        },
    }

    if approach not in approach_config:
        raise ValueError(
            f"Invalid approach '{approach}'. Choose from: {list(approach_config.keys())}"
        )

    config = approach_config[approach]
    base_path = os.path.join(ROOT_DIR, config["path"])
    filename = config["filename"]

    print(f"\n>>> Approach {approach}: {config['desc']}")
    print(f"Looking in: {base_path}\n")

    subjects = [
        "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
        "sub11", "sub12", "sub13", "sub14", "sub15", "sub16",
    ]

    all_data = {}

    for subject in subjects:
        split_path = os.path.join(base_path, subject, filename)

        if os.path.exists(split_path):
            print(f"  ✓ Loaded: {filename} for {subject}")
            X_train, X_test, y_train, y_test = joblib.load(split_path)
            all_data[subject] = (X_train, X_test, y_train, y_test)
        else:
            print(f"  ✗ File not found: {filename} for {subject}")

    return all_data
