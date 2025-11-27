# src/utils/preprocessing.py
import numpy as np
import os
import json
from typing import Tuple, List, Dict
from sklearn.decomposition import PCA
import warnings


def _coerce_events_to_1d(events: np.ndarray):
    """
    Ensure events is a 1-D integer array aligned to TRs.
    If events is (T,1) -> squeeze.
    If events is (T,k) with k>1 -> pick last column (assumed to be label column),
      but warn so the user can check.
    """
    ev = np.asarray(events)
    if ev.ndim == 1:
        return ev
    if ev.ndim == 2:
        # squeeze single-column
        if ev.shape[1] == 1:
            return ev[:, 0]
        # multiple columns: assume last column is the label (common pattern),
        # but warn the user so they can verify.
        warnings.warn(
            f"Events array has shape {ev.shape}. "
            "Selecting the last column as labels. If this is wrong, adapt the loader."
        )
        return ev[:, -1]
    # higher dims: try to squeeze
    evs = ev.squeeze()
    if evs.ndim == 1:
        return evs
    raise ValueError(f"Unable to coerce events into 1D labels. shape={events.shape}")

def make_windows(timeseries: np.ndarray,
                 events: np.ndarray,
                 seq_len: int,
                 stride: int,
                 label_mode: str = "last"):
    """
    Convert timeseries into sliding windows.

    timeseries: (T, n_features)
    events: (T,) or (T,1) or (T,k) labels aligned per TR (int labels, -1 for no-event)
    seq_len: window length (timesteps)
    stride: step between windows
    label_mode: "last" (label at last timestep) or "majority" (most frequent label in window)

    Returns:
      X: (n_windows, seq_len, n_features)
      y: (n_windows,)
      meta: list of dicts {'start': idx, 'end': idx+seq_len-1}
    """
    # Basic checks
    if timeseries is None or events is None:
        raise ValueError("timeseries and events must be provided.")
    ts = np.asarray(timeseries)
    ev = _coerce_events_to_1d(events)

    if ts.shape[0] != ev.shape[0]:
        raise ValueError(f"timeseries length ({ts.shape[0]}) and events length ({ev.shape[0]}) differ.")

    T, nfeat = ts.shape
    windows = []
    labels = []
    meta = []
    for start in range(0, T - seq_len + 1, stride):
        end = start + seq_len
        w = ts[start:end, :]
        win_events = ev[start:end]
        if label_mode == "last":
            label = int(win_events[-1])
        elif label_mode == "majority":
            vals, counts = np.unique(win_events, return_counts=True)
            label = int(vals[np.argmax(counts)])
        else:
            raise ValueError("label_mode must be 'last' or 'majority'")
        # skip windows with no label (-1)
        if label == -1:
            continue
        windows.append(w)
        labels.append(label)
        meta.append({"start": int(start), "end": int(end-1)})
    if len(windows) == 0:
        return np.zeros((0, seq_len, nfeat), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
    X = np.stack(windows).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y, meta


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray):
    """
    Standardize per-feature across time within training set (fit on train) and apply to test.
    X_train: (n_windows_train, seq_len, nfeat)
    X_test: (n_windows_test, seq_len, nfeat)
    Returns standardized arrays.
    """
    # Collapse windows/time to compute mean/std per feature
    tr_flat = X_train.reshape(-1, X_train.shape[-1])
    mean = tr_flat.mean(axis=0)
    std = tr_flat.std(axis=0)
    std[std == 0] = 1.0
    def apply(X):
        return ((X - mean.reshape(1,1,-1)) / std.reshape(1,1,-1)).astype(np.float32)
    return apply(X_train), apply(X_test)

def reduce_dimensionality(X_train: np.ndarray, X_test: np.ndarray, n_components: int = None):
    """
    Optionally apply PCA fit on training data and transform both train and test.
    X_train: (n_windows_train, seq_len, nfeat)
    X_test: (n_windows_test, seq_len, nfeat)
    Returns arrays with last dim = n_components (or original if n_components is None)
    """
    if n_components is None:
        return X_train, X_test
    tr_flat = X_train.reshape(-1, X_train.shape[-1])
    pca = PCA(n_components=n_components)
    pca.fit(tr_flat)
    def transform(X):
        n_windows, seq_len, nfeat = X.shape
        flat = X.reshape(-1, nfeat)
        flat_t = pca.transform(flat)
        return flat_t.reshape(n_windows, seq_len, n_components).astype(np.float32)
    return transform(X_train), transform(X_test)

def load_subject_preprocessed(path: str):
    """
    Load a single subject's preprocessed timeseries and per-TR labels.
    Expects a npz or json with arrays:
      - 'timeseries' : (T, nfeat) numpy array or list
      - 'events' : (T,) labels (integers) aligned to TRs, -1 for unlabeled frames
    Adapt this loader if your preprocessed files use different keys or formats.
    """
    if path.endswith('.npz'):
        d = np.load(path, allow_pickle=True)
        ts = d['timeseries']
        events = d['events']
    elif path.endswith('.npy'):
        ts = np.load(path)
        # user must provide events separately. we'll error.
        raise ValueError("npy path provided: must supply events separately.")
    elif path.endswith('.json'):
        with open(path, 'r') as f:
            j = json.load(f)
        ts = np.array(j['timeseries'], dtype=np.float32)
        events = np.array(j['events'], dtype=np.int64)
    else:
        raise ValueError("Unsupported file type for subject preload: " + path)
    return ts.astype(np.float32), events.astype(np.int64)

def build_dataset_from_subjects(subject_files: List[str],
                                seq_len: int,
                                stride: int,
                                label_mode: str = "last",
                                pca_components: int = None):
    """
    subject_files: list of file paths for each subject's preprocessed npz/json
    Returns:
      X_all: (N_windows, seq_len, nfeat)
      y_all: (N_windows,)
      groups: (N_windows,) subject index per window (0..n_subjects-1)
      metas: list of meta dicts with subject index + start/end
    """
    X_list = []
    y_list = []
    groups = []
    metas = []
    for si, path in enumerate(subject_files):
        ts, events = load_subject_preprocessed(path)   # uses your existing loader
        # coerce events if necessary (in case loader returned 2D labels)
        events = _coerce_events_to_1d(events)
        X, y, meta = make_windows(ts, events, seq_len=seq_len, stride=stride, label_mode=label_mode)
        for m in meta:
            m['subject'] = si
        X_list.append(X)
        y_list.append(y)
        groups.extend([si] * len(y))
        metas.extend(meta)
    if len(X_list) == 0:
        return np.zeros((0, seq_len, 0)), np.zeros((0,), dtype=np.int64), np.array([], dtype=np.int64), []
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    groups = np.array(groups, dtype=np.int64)
    return X_all, y_all, groups, metas