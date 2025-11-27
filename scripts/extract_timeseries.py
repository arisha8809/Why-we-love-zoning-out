import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from nilearn.signal import clean
from config import PROCESSED_DIR, HP_FILTER

def extract_timeseries(img, yeo_masked, TR, subj):
    print(f"[{subj}] Extracting Yeo-7 timeseries…")
    masker = NiftiLabelsMasker(labels_img=yeo_masked, standardize=False, detrend=True)
    ts = masker.fit_transform(img)  # shape (T, 7)

    # Clean time-series
    ts_clean = clean(ts, t_r=TR, high_pass=HP_FILTER, standardize=True)

    out_csv = PROCESSED_DIR / "timeseries" / f"{subj}_yeo7_timeseries_clean.csv"
    pd.DataFrame(ts_clean, columns=[f"Net{i}" for i in range(1, 8)]).to_csv(out_csv, index_label="TR")
    print(f"[{subj}] Saved timeseries -> {out_csv}")

    return ts_clean
