"""  
CCNN Model - Within-Subject Evaluation (One Characteristic)

This script trains a Compact Convolutional Neural Network (CCNN) from the TorchEEG library for within-subject classification.
It uses EEG signals structured in a 2D grid format (e.g., (4, 9, 9)) corresponding to 4 channels and a 9x9 grid.

Approaches:
    1 - Slice-wise (one characteristic)
    2 - Trial-wise (one characteristic)

For each subject:
    - 10-fold cross-validation is performed on the training set to select the best model per fold.
    - Each fold's best model is evaluated on the test set.
    - Accuracy and F1-score are averaged across folds.
    - Final results are saved per subject, with global statistics.

Requirements:
    - TorchEEG
    - TorchMetrics
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score
from sklearn.model_selection import KFold
from torcheeg.models import CCNN

sys.path.insert(0, "/home/jovyan/Final_TFM/Data/Scripts/4. Load")
from toGrid_load_within import load_within_data_grid

# ========== Configuration ==========
ROOT_DIR = "/home/jovyan/Final_TFM"
approach_output_dirs = {
    "1": "Results/1. with_in_subject/a. one_characteristic/slice_wise/6. cnn_new_parameters_128",
    "2": "Results/1. with_in_subject/a. one_characteristic/trial_wise/6. cnn_new_parameters_128",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========== Training Loop ==========
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for xb, yb in dataloader:
        xb = xb['eeg'].float().to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()

# ========== Validation Loop ==========
def valid(dataloader, model, loss_fn):
    model.eval()
    preds, targets = [], []
    total_loss = 0.0

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb['eeg'].float().to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_loss += loss.item()

            preds.extend(pred.argmax(1).cpu().numpy())
            targets.extend(yb.cpu().numpy())

    preds = np.array(preds)
    targets = np.array(targets)
    acc = np.mean(preds == targets)
    f1 = BinaryF1Score().to(device)(
        torch.tensor(preds).to(device),
        torch.tensor(targets).to(device)
    ).item()
    return acc, total_loss / len(dataloader), f1

# ========== Main Training Function ==========
def ccnn_within_model(approach: str):
    if approach not in approach_output_dirs:
        raise ValueError(f"Invalid approach '{approach}'.")

    seed_everything()
    all_data = load_within_data_grid(approach)
    output_dir = os.path.join(ROOT_DIR, approach_output_dirs[approach])
    os.makedirs(output_dir, exist_ok=True)

    final_results = []

    for subject, (X_train, X_test, y_train, y_test) in all_data.items():
        print(f"\nTraining CCNN model for {subject}")
        subject_output = os.path.join(output_dir, subject)
        os.makedirs(subject_output, exist_ok=True)

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)

        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        test_accuracies, test_f1s = [], []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            x_tr, x_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            train_loader = DataLoader(list(zip(x_tr, y_tr)), batch_size=128, shuffle=True)
            val_loader = DataLoader(list(zip(x_val, y_val)), batch_size=128)
            test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=128)

            model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
            loss_fn = nn.CrossEntropyLoss()

            best_acc = 0.0

            for epoch in range(500):
                train_loss = train(train_loader, model, loss_fn, optimizer)
                val_acc, val_loss, val_f1 = valid(val_loader, model, loss_fn)

                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), os.path.join(subject_output, f"best_model_fold{fold}.pt"))

            # === Test phase
            model.load_state_dict(torch.load(os.path.join(subject_output, f"best_model_fold{fold}.pt")))
            model.eval()
            preds, targets = [], []

            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb['eeg'].float().to(device)
                    yb = yb.to(device)
                    out = model(xb)
                    preds.extend(out.argmax(1).cpu().numpy())
                    targets.extend(yb.cpu().numpy())

            acc = np.mean(np.array(preds) == np.array(targets))
            f1 = BinaryF1Score().to(device)(
                torch.tensor(preds).to(device),
                torch.tensor(targets).to(device)
            ).item()

            test_accuracies.append(acc)
            test_f1s.append(f1)

        final_results.append({
            "subject": subject,
            "accuracy": np.mean(test_accuracies),
            "f1_score": np.mean(test_f1s),
        })

        print(f"{subject} → Test Accuracy: {np.mean(test_accuracies)*100:.2f}%, Test F1: {np.mean(test_f1s)*100:.2f}%")

    # ========== Save Final Results ==========
    df = pd.DataFrame(final_results)
    mean_acc = df["accuracy"].mean() * 100
    std_acc = df["accuracy"].std() * 100
    mean_f1 = df["f1_score"].mean() * 100
    std_f1 = df["f1_score"].std() * 100

    df = pd.concat([
        df,
        pd.DataFrame([{
            "subject": "AVG±STD",
            "accuracy": f"{mean_acc:.2f}% ± {std_acc:.2f}%",
            "f1_score": f"{mean_f1:.2f} ± {std_f1:.2f}"
        }])
    ], ignore_index=True)

    df.to_csv(os.path.join(output_dir, "ccnn_test_metrics_by_subject.csv"), index=False)

    print("\n Training complete.")
    print(f"Average Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Average Test F1 Score: {mean_f1:.2f} ± {std_f1:.2f}")
