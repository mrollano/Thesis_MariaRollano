import os
import joblib

def load_within_data_grid(approach: str):
    """
    Loads within-subject train/test splits for CNN and ViT models 
    using ToGrid-transformed EEG features.

    These approaches correspond to:
        1: One characteristic — Slice-wise — ToGrid
        2: One characteristic — Trial-wise — ToGrid
        3: Multiple characteristics — Slice-wise — ToGrid
        4: Multiple characteristics — Trial-wise — ToGrid

    Returns:
        dict: {
            subject_id: (X_train, X_test, y_train, y_test)
        }

    Raises:
        ValueError: if an invalid approach is given.
    """

    ROOT_DIR = "/home/jovyan/Final_TFM"

    # Configuration: path, filename, description
    approach_config = {
        "1": {
            "path": "Data/Splits/toGrid_split_train_test/slice_wise/one_characteristic",
            "filename": "train_test_split_transformer.pkl",
            "desc": "One characteristic — Within-subject — Slice-wise — ToGrid",
        },
        "2": {
            "path": "Data/Splits/toGrid_split_train_test/trial_wise/one_characteristic",
            "filename": "train_test_split_trial_wise_transformer.pkl",
            "desc": "One characteristic — Within-subject — Trial-wise — ToGrid",
        },
        "3": {
            "path": "Data/Splits/toGrid_split_train_test/slice_wise/multiple_characteristic",
            "filename": "train_test_split_grid.pkl",
            "desc": "Multiple characteristics — Within-subject — Slice-wise — ToGrid",
        },
        "4": {
            "path": "Data/Splits/toGrid_split_train_test/trial_wise/multiple_characteristics",
            "filename": "train_test_split_trial_wise_grid.pkl",
            "desc": "Multiple characteristics — Within-subject — Trial-wise — ToGrid",
        },
    }

    if approach not in approach_config:
        raise ValueError(
            f"Invalid approach '{approach}'. Valid options are: {list(approach_config.keys())}"
        )

    config = approach_config[approach]
    base_path = os.path.join(ROOT_DIR, config["path"])
    filename = config["filename"]

    print(f"\n>>> Approach {approach}: {config['desc']}")
    print(f"Searching in: {base_path}\n")

    subjects = [
        "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
        "sub11", "sub12", "sub13", "sub14", "sub15", "sub16",
    ]

    all_data = {}

    for subject in subjects:
        split_path = os.path.join(base_path, subject, filename)

        if os.path.exists(split_path):
            print(f"  ✓ Loaded {filename} for {subject}")
            X_train, X_test, y_train, y_test = joblib.load(split_path)
            all_data[subject] = (X_train, X_test, y_train, y_test)
        else:
            print(f"  ✗ File not found for {subject}: {filename}")

    return all_data
