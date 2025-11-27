import numpy as np
import pandas as pd
import os
import glob
import re

TS_DIR = "data_processed/timeseries"
LBL_DIR = "data_processed/tr_labels"
OUT_DIR = "data_processed/preproc"

os.makedirs(OUT_DIR, exist_ok=True)

ts_files = sorted(glob.glob(os.path.join(TS_DIR, "*.csv")))

def extract_subject_id(filename):
    m = re.search(r"(sub-[0-9]+)", filename)
    return m.group(1) if m else None

def load_numeric_csv(path):
    df = pd.read_csv(path, header=None)
    # Remove any rows containing non-numeric values
    df_clean = df[
        df.apply(lambda row: all(str(x).replace('.', '').replace('-', '').isdigit() for x in row), axis=1)
    ]
    return df_clean.values

print("\nFound timeseries files:", len(ts_files))

for ts_path in ts_files:
    filename = os.path.basename(ts_path)
    subj = extract_subject_id(filename)

    if subj is None:
        print(f"❌ Cannot extract subject ID from: {filename}")
        continue

    label_candidates = glob.glob(os.path.join(LBL_DIR, f"{subj}*.csv"))
    if len(label_candidates) == 0:
        print(f"⚠️  No label CSV found for {subj}, skipping…")
        continue

    lbl_path = label_candidates[0]

    print(f"\nProcessing {subj}:")
    print("  TS  =", filename)
    print("  LBL =", os.path.basename(lbl_path))

    timeseries = load_numeric_csv(ts_path).astype(np.float32)
    events = load_numeric_csv(lbl_path).squeeze().astype(np.int64)

    ts_len = len(timeseries)
    lbl_len = len(events)

    # ---- FIX LENGTH MISMATCH ----
    if lbl_len > ts_len:
        print(f"  ⚠️  Label file longer ({lbl_len}) than fMRI ({ts_len}). Trimming labels.")
        events = events[:ts_len]
    elif lbl_len < ts_len:
        print(f"  ⚠️  Timeseries longer ({ts_len}) than labels ({lbl_len}). Trimming fMRI.")
        timeseries = timeseries[:lbl_len]

    # Final sanity check
    if len(timeseries) != len(events):
        print(f"❌ Still mismatched after trimming for {subj}, skipping.")
        continue

    out_path = os.path.join(OUT_DIR, f"{subj}_preproc.npz")
    np.savez_compressed(out_path, timeseries=timeseries, events=events)

    print("  ✔ Saved:", out_path)

print("\nDone!")
