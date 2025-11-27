from pathlib import Path
import glob
import numpy as np
import subprocess

# ==== Original imports (SVM + PCA pipeline) ====
from preprocess_subject import preprocess_subject
from extract_timeseries import extract_timeseries
from generate_tr_labels import generate_tr_labels
from compute_pca import compute_pca
from decode_subject import decode_subject

# ==== Group-level SVM analyses ====
from group_loso import run_loso
from group_event_locked import group_event_locked
from group_isc import group_isc

from config import RAW_DATA_DIR, FMRI_PATTERN

# ----------------------------------------------------
# 1. Find subjects
# ----------------------------------------------------
fmri_paths = sorted(glob.glob(str(RAW_DATA_DIR / FMRI_PATTERN), recursive=True))
subjects = [Path(p).stem.split("_")[0] for p in fmri_paths]
print("Found subjects:", subjects)

# ----------------------------------------------------
# 2. Per-subject preprocessing
#    → produce inputs for BOTH SVM and LSTM
# ----------------------------------------------------
for bold_path in fmri_paths:

    subj = Path(bold_path).stem.split("_")[0]

    # Preprocess
    img, mask, yeo_masked, TR = preprocess_subject(bold_path)

    # Extract Yeo-7 signals
    ts = extract_timeseries(img, yeo_masked, TR, subj)

    # TR labels
    tr_labels = generate_tr_labels(subj, TR, img.shape[-1])

    # --- SVM INPUTS (unchanged) ---
    X_pca = compute_pca(img, mask, subj)
    decode_subject(X_pca, tr_labels, subj)

    # --- LSTM INPUTS ---
    np.savez_compressed(f"data_processed/preproc/{subj}_preproc.npz",
                        timeseries=ts, events=tr_labels)
    print(f"Saved LSTM npz for {subj}")

# ----------------------------------------------------
# 3. GROUP-LEVEL SVM ANALYSIS (unchanged)
# ----------------------------------------------------
print("\n=== Running SVM LOSO ===")
run_loso(subjects)

print("\n=== Running Event-Locked Analysis ===")
group_event_locked(subjects)

print("\n=== Running ISC Analysis ===")
group_isc(subjects)

# ----------------------------------------------------
# 4. GROUP-LEVEL LSTM ANALYSIS
#    (parallel decoding pipeline)
# ----------------------------------------------------
print("\n=== Running LSTM LOSO ===")
subprocess.run(["python", "src/run_lstm.py", "--config", "configs/lstm_config.yaml"])

print("\nPipeline complete! SVM + LSTM results ready.")
