""" 
ViT Between-Subject Classification – Multiple Characteristics
------------------------------------------------------------
This script trains and evaluates a Vision Transformer (ViT) model
using a leave-one-subject-out (LOSO) cross-validation strategy.
It is designed for between-subject classification with multiple EEG characteristics.

Data input:      Grid-based EEG tensors (chunk_size=24)
Model:           ViT (depth=6, heads=9, hid_channels=128, dropout=0.2)
Evaluation:      Accuracy and F1 score
Approach key:    "6" – between, multiple characteristics
Output:          Trained models + metrics per subject in /Results/2. between_subject/b. multiple characteristic/7. transformer
"""

import os
import sys
import random
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import joblib
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score
from torcheeg.models import ViT

# Load custom loader
sys.path.insert(0, "/home/jovyan/Final_TFM/Data/Scripts/4. Load")
from toGrid_load_between import load_between_data_grid

# ========== Configuration ==========
ROOT_DIR = "/home/jovyan/Final_TFM"
approach_output_dirs = {
    "6": "Results/2. between_subject/b. multiple characteristic/7. transformer",
}
subjects = [
    "sub04", "sub05", "sub07", "sub08", "sub09", "sub10",
    "sub11", "sub12", "sub13", "sub14", "sub15", "sub16",
]

# ========== Seed for Reproducibility ==========
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========== Main Function ==========
def vit_between_model(approach: str):
    print(f"\nTraining ViT – Between-subject (Approach {approach})")
    if approach not in approach_output_dirs:
        raise ValueError(f"Invalid approach '{approach}'. Allowed: {list(approach_output_dirs.keys())}")

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = os.path.join(ROOT_DIR, approach_output_dirs[approach])
    os.makedirs(output_dir, exist_ok=True)

    all_data = load_between_data_grid(approach)
    results = []

    for test_subject in subjects:
        print(f"\nLOSO Fold – Test subject: {test_subject}")

        # Split data
        train_subjects = [s for s in subjects if s != test_subject]
        random.shuffle(train_subjects)
        n_val = max(1, len(train_subjects) // 10)
        val_subjects = train_subjects[:n_val]
        real_train_subjects = train_subjects[n_val:]

        X_train, y_train = [], []
        for subj in real_train_subjects:
            X, y = all_data[subj]
            X_train.extend(X)
            y_train.extend(y)

        X_val, y_val = [], []
        for subj in val_subjects:
            X, y = all_data[subj]
            X_val.extend(X)
            y_val.extend(y)

        X_test, y_test = all_data[test_subject]

        # === Model Definition ===
        model = ViT(
            chunk_size=24,
            grid_size=(9, 9),
            t_patch_size=1,
            num_classes=2,
            depth=6,
            heads=9,
            hid_channels=128,
            dropout=0.2,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        loss_fn = nn.CrossEntropyLoss()

        # === DataLoaders ===
        train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=128, shuffle=True)
        val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=128)
        test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=128)

        best_val_acc = 0.0

        # === Training loop ===
        for epoch in range(500):
            start_time = time.time()
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.float().to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # === Validation ===
            model.eval()
            correct = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.float().to(device), yb.to(device)
                    pred = model(xb)
                    correct += (pred.argmax(1) == yb).sum().item()

            val_acc = correct / len(val_loader.dataset)
            epoch_time = time.time() - start_time
            print(f" Epoch {epoch+1}: val_acc = {val_acc*100:.2f}%, time = {epoch_time:.2f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(output_dir, f"{test_subject}_best_model.pt"))

        # === Test evaluation ===
        model.load_state_dict(torch.load(os.path.join(output_dir, f"{test_subject}_best_model.pt")))
        model.eval()
        preds, y_true = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.float().to(device), yb.to(device)
                out = model(xb)
                preds.extend(out.argmax(1).cpu().numpy())
                y_true.extend(yb.cpu().numpy())

        acc = (np.array(preds) == np.array(y_true)).mean()
        f1 = BinaryF1Score().to(device)(
            torch.tensor(preds).to(device),
            torch.tensor(y_true).to(device)
        ).item()

        results.append({
            "subject": test_subject,
            "accuracy": acc * 100,
            "f1_score": f1 * 100
        })

        print(f"  {test_subject}: Accuracy = {acc*100:.2f}%, F1 Score = {f1*100:.2f}%")

    # === Save final metrics ===
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

    df.to_csv(os.path.join(output_dir, "vit_loso_metrics.csv"), index=False)

    print("\n Evaluation completed.")
    print(f"Mean Accuracy : {df.loc[df['subject'] == 'MEAN', 'accuracy'].values[0]:.2f}%")
    print(f"Mean F1 Score : {df.loc[df['subject'] == 'MEAN', 'f1_score'].values[0]:.2f}%")
