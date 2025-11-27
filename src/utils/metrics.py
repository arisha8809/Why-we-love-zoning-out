# src/utils/metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import json
import os
from collections import defaultdict, Counter

def aggregate_window_predictions(logits: np.ndarray, metas, agg_mode="prob_mean"):
    """
    logits: (n_windows, n_classes) raw logits or probabilities
    metas: list of dicts with at least 'subject' and optionally 'trial' or 'start'/'end'
    agg_mode: "prob_mean" or "majority"
    Returns:
      trial_preds: list of dicts {subject, trial_id, true_label, pred_label, probs}
    """
    # if logits are raw, convert to probs
    if logits.ndim == 2:
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / exps.sum(axis=1, keepdims=True)
    else:
        raise ValueError("logits must be 2D: (n_windows, n_classes)")
    # group windows by trial (we use subject + start/end as a trial key)
    groups = defaultdict(list)
    for i, m in enumerate(metas):
        # meta may contain keys: subject, start, end ; use subject+start as trial id
        subj = m.get('subject', None)
        start = m.get('start', i)
        key = (subj, start)
        groups[key].append(i)
    trial_preds = []
    for (subj, start), idxs in groups.items():
        block_probs = probs[idxs]  # (n_win_in_trial, n_classes)
        mean_prob = block_probs.mean(axis=0)
        pred = int(mean_prob.argmax())
        trial_preds.append({
            "subject": int(subj) if subj is not None else None,
            "trial_key_start": int(start),
            "pred_label": int(pred),
            "probs": mean_prob.tolist(),
            "n_windows": len(idxs),
            "window_idxs": [int(i) for i in idxs]
        })
    return trial_preds

def compute_metrics_from_trials(trial_preds, true_labels_map):
    """
    trial_preds: list as returned by aggregate_window_predictions
    true_labels_map: dict mapping (subject, trial_key_start) -> true_label
    Returns dict with accuracy and optionally AUC (if binary)
    """
    y_true = []
    y_pred = []
    probs = []
    for t in trial_preds:
        key = (t["subject"], t["trial_key_start"])
        if key not in true_labels_map:
            continue
        y_true.append(int(true_labels_map[key]))
        y_pred.append(int(t["pred_label"]))
        probs.append(np.array(t["probs"]))
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else None
    auc = None
    if probs and probs[0].shape[0] == 2:
        try:
            probs_pos = np.array(probs)[:,1]
            auc = float(roc_auc_score(y_true, probs_pos))
        except Exception:
            auc = None
    return {"trial_accuracy": acc, "trial_auc": auc, "n_trials": len(y_true)}

def save_fold_json(out_dir: str, subject_left_out, fold_metrics: dict, trial_preds, config: dict):
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"lstm_fold_sub-{subject_left_out:02d}.json")
    payload = {
        "subject_left_out": int(subject_left_out),
        "metrics": fold_metrics,
        "trial_predictions": trial_preds,
        "config": config
    }
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    return filename
