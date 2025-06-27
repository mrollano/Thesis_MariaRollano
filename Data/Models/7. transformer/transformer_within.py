import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import joblib
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score
from sklearn.model_selection import train_test_split
from torcheeg.models import ViT

# Add custom script path to import the data loader
sys.path.insert(0, "/home/jovyan/Final_TFM/Data/Scripts/4. Load")
from toGrid_load_within import load_within_data_grid

# Set the root directory
ROOT_DIR = "/home/jovyan/Final_TFM"

# Output directories for each within-subject approach
approach_output_dirs = {
    "1": "Results/1. with_in_subject/a. one_characteristic/slice_wise/7. transformer",
    "2": "Results/1. with_in_subject/a. one_characteristic/trial_wise/7. transformer",
}

# Set seed for reproducibility
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Main function for training and evaluating the ViT model
def vit_within_model(approach: str):
    print(f"\nTraining Transformer Model (Approach {approach})")
    if approach not in approach_output_dirs:
        raise ValueError(f"Invalid approach '{approach}'. Must be one of: {list(approach_output_dirs.keys())}")

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the pre-split data for within-subject evaluation
    all_data = load_within_data_grid(approach)
    output_dir = os.path.join(ROOT_DIR, approach_output_dirs[approach])
    os.makedirs(output_dir, exist_ok=True)

    final_results = []

    # Train and evaluate model for each subject
    for subject, (X_train, X_test, y_train, y_test) in all_data.items():
        print(f"\nTraining model for subject {subject}...")

        subject_output = os.path.join(output_dir, subject)
        os.makedirs(subject_output, exist_ok=True)

        # Split training data into training and validation
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        train_loader = DataLoader(list(zip([x["eeg"] for x in X_tr], y_tr)), batch_size=128, shuffle=True)
        val_loader = DataLoader(list(zip([x["eeg"] for x in X_val], y_val)), batch_size=128)
        test_loader = DataLoader(list(zip([x["eeg"] for x in X_test], y_test)), batch_size=128)

        # Define the ViT model (Vision Transformer)
        model = ViT(
            chunk_size=4,
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
        best_val_acc = 0.0

        # Train for 500 epochs
        for epoch in range(500):
            model.train()
            for X, y in train_loader:
                X = X.float().to(device)
                y = y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation step
            model.eval()
            correct, val_loss = 0, 0
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.float().to(device)
                    y = y.to(device)
                    pred = model(X)
                    val_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).sum().item()

            val_acc = correct / len(val_loader.dataset)
            print(f" Epoch {epoch+1}: val_acc = {val_acc*100:.2f}%")

            # Save best model by validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(subject_output, "best_model.pt"))

        # Final test evaluation
        model.load_state_dict(torch.load(os.path.join(subject_output, "best_model.pt")))
        model.eval()
        preds, y_true = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.float().to(device)
                y = y.to(device)
                out = model(X)
                preds.extend(out.argmax(1).cpu().numpy())
                y_true.extend(y.cpu().numpy())

        acc = (np.array(preds) == np.array(y_true)).mean()
        f1 = (
            BinaryF1Score()
            .to(device)(torch.tensor(preds).to(device), torch.tensor(y_true).to(device))
            .item()
        )

        # Save model and metrics
        joblib.dump(model.state_dict(), os.path.join(subject_output, "vit_final_model.pt"))
        final_results.append({
            "subject": subject,
            "accuracy": acc,
            "f1_score": f1
        })

        print(f" {subject}: Accuracy = {acc*100:.2f}%, F1 Score = {f1*100:.2f}")

    # Aggregate results across all subjects
    df = pd.DataFrame(final_results)
    mean_acc = df["accuracy"].mean()
    std_acc = df["accuracy"].std()
    mean_f1 = df["f1_score"].mean()
    std_f1 = df["f1_score"].std()

    # Add AVG±STD summary row
    df = pd.concat([
        df,
        pd.DataFrame([{
            "subject": "AVG±STD",
            "accuracy": f"{mean_acc*100:.2f}% ± {std_acc*100:.2f}%",
            "f1_score": f"{mean_f1*100:.2f} ± {std_f1*100:.2f}",
        }])
    ])

    # Save final CSV with metrics
    df.to_csv(os.path.join(output_dir, "vit_metrics_by_subject.csv"), index=False)

    print("\nEvaluation completed.")
    print(f"Average Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Average F1 Score: {mean_f1*100:.2f} ± {std_f1*100:.2f}")
