# scripts/run_one_subject.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Import pipeline modules
from scripts.preprocess_subject import preprocess_subject
from scripts.extract_timeseries import extract_timeseries
from scripts.generate_tr_labels import generate_tr_labels
from scripts.compute_pca import compute_pca
from scripts.decode_subject import decode_subject

# ----------------------------
# FIND RAW SUBJECT FILE
# ----------------------------

def find_subject_bold(subj_id):
    raw_dir = Path("data_raw")
    candidates = list(raw_dir.rglob(f"{subj_id}*.nii*"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No preprocessed bold file found for {subj_id} in data_raw/")
    return candidates[0]


# ----------------------------
# RUN FULL PIPELINE
# ----------------------------

def run_pipeline(subj_id):

    print(f"\n==============================")
    print(f" Running full pipeline for {subj_id}")
    print(f"==============================\n")

    # 1) ------------------- Find raw file
    bold_path = find_subject_bold(subj_id)
    print(f"Found raw file: {bold_path}")

    # 2) ------------------- Preprocess / mask / atlas
    print("\n[1] Preprocessing subject...")
    img, mask, yeo_masked, TR = preprocess_subject(str(bold_path))
    n_trs = img.shape[-1]
    print(f"✓ Preprocessing complete. TR={TR}, TRs={n_trs}")

    # 3) ------------------- Extract timeseries
    print("\n[2] Extracting Yeo-7 timeseries...")
    ts = extract_timeseries(img, yeo_masked, TR, subj_id)
    print(f"✓ Timeseries saved. Shape={ts.shape}")

    # 4) ------------------- Generate TR labels
    print("\n[3] Generating TR labels from narrative...")
    y = generate_tr_labels(subj_id, TR, n_trs)
    print(f"✓ TR labels saved. Length={len(y)}")

    # 5) ------------------- Compute PCA
    print("\n[4] Running PCA...")
    X_pca = compute_pca(img, mask, subj_id)
    print(f"✓ PCA complete. PCA shape={X_pca.shape}")

    # 6) ------------------- Match lengths (if needed)
    min_len = min(X_pca.shape[0], len(y))
    X_pca = X_pca[:min_len]
    y = y[:min_len]

    # 7) ------------------- Decode
    print("\n[5] Running SVM decoding (within-subject)...")
    results = decode_subject(X_pca, y, subj_id)
    print(f"✓ Decoding complete.")

    # 8) ------------------- Print summary
    print("\n--------------- SUMMARY ---------------")
    print(f"Subject:            {subj_id}")
    print(f"Final TRs used:     {min_len}")
    print(f"Decoding accuracy:  {results['accuracy']:.3f}")
    print(f"Permutation p-val:  {results['perm_p']:.4f}")
    print("----------------------------------------\n")

    return results


# ----------------------------
# MAIN
# ----------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_one_subject.py sub-01")
        sys.exit(1)

    subj_id = sys.argv[1]
    run_pipeline(subj_id)
