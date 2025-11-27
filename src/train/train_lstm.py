# src/train/train_lstm.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import LeaveOneGroupOut
import random

from src.models.lstm import LSTMClassifier
from src.data.dataset import SequenceDataset
from src.utils.metrics import aggregate_window_predictions, compute_metrics_from_trials, save_fold_json
from src.utils.preprocessing import standardize_train_test, reduce_dimensionality

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------
# TRAIN ONE EPOCH (FIXED: config + class_weights passed)
# -----------------------------------------------------
def train_one_epoch(model, optim, loader, device, config, class_weights=None):
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)

        logits = model(x)

        # Balanced loss
        if config.get("use_class_weights", False) and class_weights is not None:
            loss = F.cross_entropy(logits, y, weight=class_weights)
        else:
            loss = F.cross_entropy(logits, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

    return total_loss / max(n, 1)

# -----------------------------------------------------
# WINDOW EVALUATION
# -----------------------------------------------------
def evaluate_windows(model, loader, device):
    model.eval()
    logits_all = []
    y_all = []
    metas_all = []

    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            out = model(x)
            logits_all.append(out.cpu().numpy())
            y_all.append(batch['y'].numpy())
            metas_all.extend(batch["meta"])

    if len(logits_all) == 0:
        return np.zeros((0,2)), np.zeros((0,), dtype=int), []

    logits_all = np.vstack(logits_all)
    y_all = np.concatenate(y_all)

    return logits_all, y_all, metas_all

# -----------------------------------------------------
# FULL PIPELINE
# -----------------------------------------------------
def lstm_train_pipeline(X, y, groups, metas, subject_files, config):
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("use_cuda", True) else "cpu")
    logo = LeaveOneGroupOut()
    results = []

    set_seed(config.get("seed", 42))
    os.makedirs(config["out_dir"], exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):

        left_out_sub = int(groups[test_idx[0]])
        print(f"\n=== Fold {fold+1}, leaving out subject {left_out_sub} ===")

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        metas_train = [metas[i] for i in train_idx]
        metas_test  = [metas[i] for i in test_idx]

        # PCA (optional)
        if config.get("pca_components", None):
            X_train, X_test = reduce_dimensionality(X_train, X_test, n_components=config["pca_components"])

        # Normalization
        X_train, X_test = standardize_train_test(X_train, X_test)

        # Datasets & loaders
        train_ds = SequenceDataset(X_train, y_train, meta=metas_train)
        test_ds  = SequenceDataset(X_test, y_test, meta=metas_test)

        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        test_loader  = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

        # Model
        model = LSTMClassifier(
            input_dim=X_train.shape[-1],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_classes=config["num_classes"],
            bidirectional=config["bidirectional"],
            dropout=config["dropout"],
        ).to(device)

        optim = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0.0))

        # -----------------------------------------------------
        # Compute class weights ONCE per fold (fixed)
        # -----------------------------------------------------
        if config.get("use_class_weights", False):
            unique, counts = np.unique(y_train, return_counts=True)
            freq = counts / counts.sum()
            inv = 1.0 / (freq + 1e-8)
            inv = inv / inv.sum()
            class_weights = torch.tensor(inv, dtype=torch.float32).to(device)

            print(f"Class weights for this fold: {inv}")
        else:
            class_weights = None

        best_val = -np.inf
        best_state = None
        patience = config.get("patience", 6)
        patience_ctr = 0

        # -----------------------------------------------------
        # TRAINING LOOP
        # -----------------------------------------------------
        for epoch in range(1, config["epochs"] + 1):
            tr_loss = train_one_epoch(model, optim, train_loader, device, config, class_weights)
            logits_test, y_windows_test, metas_pred = evaluate_windows(model, test_loader, device)

            # Build true label map
            true_map = {}
            for idx, m in enumerate(metas_test):
                key = (m["subject"], m["start"])
                true_map[key] = int(y_windows_test[idx])

            # Aggregate window predictions
            trial_preds = aggregate_window_predictions(logits_test, metas_test, agg_mode="prob_mean")
            metrics = compute_metrics_from_trials(trial_preds, true_map)

            val_acc = metrics.get("trial_accuracy", 0.0)
            print(f"Fold {fold+1} Epoch {epoch} — tr_loss={tr_loss:.4f}  val_acc={val_acc:.4f}")

            if val_acc > best_val:
                best_val = val_acc
                best_state = model.state_dict()
                patience_ctr = 0
            else:
                patience_ctr += 1

            if patience_ctr >= patience:
                print("Early stopping triggered.")
                break

        # Save best model
        model_path = os.path.join(config["out_dir"], f"best_lstm_sub-{left_out_sub:02d}.pt")
        if best_state is not None:
            torch.save(best_state, model_path)
            model.load_state_dict(best_state)

        # Final evaluation
        logits_test, y_windows_test, metas_pred = evaluate_windows(model, test_loader, device)

        true_map = {}
        for idx, m in enumerate(metas_test):
            key = (m["subject"], m["start"])
            true_map[key] = int(y_windows_test[idx])

        trial_preds = aggregate_window_predictions(logits_test, metas_test, agg_mode="prob_mean")
        metrics = compute_metrics_from_trials(trial_preds, true_map)

        out_json = save_fold_json(config["out_dir"], left_out_sub, metrics, trial_preds, config)
        print(f"Saved fold results to {out_json}")

        results.append({"fold": fold, "left_out": left_out_sub, "metrics": metrics, "json": out_json})

    return results
