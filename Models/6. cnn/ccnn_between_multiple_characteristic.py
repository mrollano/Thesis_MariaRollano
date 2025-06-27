"""   
This script trains a Compact Convolutional Neural Network (CCNN) using a between-subject
strategy with multiple EEG characteristics per electrode (24-channel input).

Each subject is used as the test set once (LOSO cross-validation).
Training and validation are done using the remaining subjects, with a small portion used
for validation to select the best model checkpoint.

Input shape per sample: (24, 9, 9) → 24 EEG features mapped to a 9x9 topographic grid.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import BinaryF1Score
import pandas as pd
from torcheeg.models import CCNN

# Setup paths
sys.path.insert(0, "/home/jovyan/Final_TFM/Data/Scripts/4. Load")
from toGrid_load_between import load_between_data_grid

ROOT_DIR = "/home/jovyan/Final_TFM"
approach_output_dirs = {
    "6": "Results/2. between_subject/b. multiple characteristic/6. cnn"
}

subjects = [
    "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
    "sub11", "sub12", "sub13", "sub14", "sub15", "sub16"
]


def seed_everything(seed: int = 42):
    """Ensure full reproducibility by fixing all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(dataloader, model, loss_fn, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
    return loss.item()


def validate(dataloader, model, loss_fn, device):
    """Validate model accuracy on validation set."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
    return correct / len(dataloader.dataset)


def ccnn_between_model_multiple(approach: str):
    """Train CCNN in a between-subject setting using multiple EEG features per electrode."""
    if approach not in approach_output_dirs:
        raise ValueError(f"Invalid approach '{approach}'. Options: {list(approach_output_dirs.keys())}")

    print(f"\nTraining CCNN – Between-subject (Approach {approach})")
    seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = os.path.join(ROOT_DIR, approach_output_dirs[approach])
    os.makedirs(output_dir, exist_ok=True)

    all_data = load_between_data_grid(approach)
    results = []
    f1_metric = BinaryF1Score().to(device)

    for test_subject in subjects:
        print(f"\n→ Test Subject: {test_subject}")

        train_subjects = [s for s in subjects if s != test_subject]
        random.shuffle(train_subjects)
        n_val = max(1, len(train_subjects) // 10)
        val_subjects = train_subjects[:n_val]
        real_train_subjects = train_subjects[n_val:]

        def make_tensor_dataset(X_list, y_list):
            X_t = torch.stack([torch.tensor(x, dtype=torch.float32) for x in X_list])
            y_t = torch.tensor(y_list, dtype=torch.long)
            return TensorDataset(X_t, y_t)

        # Aggregate training data
        X_train, y_train = [], []
        for subj in real_train_subjects:
            X, y = all_data[subj]
            X_train.extend(X)
            y_train.extend(y)

        # Aggregate validation data
        X_val, y_val = [], []
        for subj in val_subjects:
            X, y = all_data[subj]
            X_val.extend(X)
            y_val.extend(y)

        # Test subject
        X_test, y_test = all_data[test_subject]

        train_loader = DataLoader(make_tensor_dataset(X_train, y_train), batch_size=128, shuffle=True)
        val_loader = DataLoader(make_tensor_dataset(X_val, y_val), batch_size=128)
        test_loader = DataLoader(make_tensor_dataset(X_test, y_test), batch_size=128)

        # Model setup
        model = CCNN(num_classes=2, in_channels=24, grid_size=(9, 9)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        best_val_acc = 0.0

        # -------------------------------
        # Training loop (500 epochs)
        # -------------------------------
        for epoch in range(500):
            train_loss = train(train_loader, model, loss_fn, optimizer, device)
            val_acc = validate(val_loader, model, loss_fn, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(output_dir, f"{test_subject}_best_model.pt"))

        # -------------------------------
        # Final evaluation on test set
        # -------------------------------
        model.load_state_dict(torch.load(os.path.join(output_dir, f"{test_subject}_best_model.pt")))
        model.eval()

        preds_total, targets_total = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds_total.extend(out.argmax(1).cpu().numpy())
                targets_total.extend(yb.cpu().numpy())

        acc = (np.array(preds_total) == np.array(targets_total)).mean() * 100
        f1 = f1_metric(
            torch.tensor(preds_total).to(device),
            torch.tensor(targets_total).to(device)
        ).item() * 100

        print(f"   → Accuracy: {acc:.2f} %, F1 Score: {f1:.2f} %")
        results.append({"subject": test_subject, "accuracy": acc, "f1_score": f1})

    # -------------------------------
    # Save global results to CSV
    # -------------------------------
    df = pd.DataFrame(results)
    df = pd.concat([
        df,
        pd.DataFrame([{
            "subject": "MEAN",
            "accuracy": df["accuracy"].mean(),
            "f1_score": df["f1_score"].mean()
        }, {
            "subject": "STD",
            "accuracy": df["accuracy"].std(),
            "f1_score": df["f1_score"].std()
        }])
    ], ignore_index=True)

    csv_path = os.path.join(output_dir, "ccnn_loso_metrics.csv")
    df.to_csv(csv_path, index=False)

    print("\n--------------------------------------------------")
    print(f"Mean Accuracy : {df.loc[df.subject == 'MEAN', 'accuracy'].values[0]:.2f} %")
    print(f"Mean F1 Score : {df.loc[df.subject == 'MEAN', 'f1_score'].values[0]:.2f} %")
    print(f"Results saved to: {csv_path}")
    print("--------------------------------------------------\n")
