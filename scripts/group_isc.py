import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

from config import PROCESSED_DIR, RESULTS_DIR

# High-contrast Yeo-7 colors
NETWORK_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
]

FIGSIZE = (8,5)
DPI = 300
FONT = 14


# ------------------------------------------------------
# ISC Computation
# ------------------------------------------------------

def compute_pairwise_ISC(stack):
    """
    stack shape = S × T × N
    returns ISC per network (length N)
    """
    S, T, N = stack.shape
    isc = np.zeros(N, dtype=float)
    pairs = 0

    for i in range(S):
        for j in range(i + 1, S):
            for n in range(N):
                isc[n] += np.corrcoef(stack[i, :, n], stack[j, :, n])[0, 1]
            pairs += 1

    return isc / pairs if pairs > 0 else isc


# ------------------------------------------------------
# MAIN GROUP ISC FUNCTION
# ------------------------------------------------------

def group_isc(subjects):
    print("[GROUP] ISC started…")

    nets = [f"Net{i}" for i in range(1, 8)]
    ts_list = []

    for subj in subjects:
        ts_path = PROCESSED_DIR / "timeseries" / f"{subj}_yeo7_timeseries_clean.csv"

        if not ts_path.exists():
            print(f"[WARN] Missing file for {subj}, skipping.")
            continue

        # Load without header → then drop TR column (first column)
        df = pd.read_csv(ts_path, header=0)

        if df.shape[1] != 8:
            print(f"[ERROR] File {ts_path} has {df.shape[1]} columns, expected 8 (TR + 7 nets). Skipping.")
            continue

        # Drop TR column
        df = df.iloc[:, 1:]

        if df.shape[1] != 7:
            print(f"[ERROR] Unexpected number of network columns after dropping TR for {subj}. Skipping.")
            continue

        ts_list.append(df.values.astype(float))

    if len(ts_list) == 0:
        print("[ERROR] No valid timeseries found.")
        return None

    # Align lengths (shortest TRs)
    minTR = min(len(ts) for ts in ts_list)
    stack = np.stack([ts[:minTR] for ts in ts_list], axis=0)

    print("[DEBUG] Stack shape S×T×N =", stack.shape)

    # Compute ISC
    isc_vals = compute_pairwise_ISC(stack)

    df_isc = pd.DataFrame({
        "network": nets,
        "isc": isc_vals
    })

    # Save CSV
    out_csv = PROCESSED_DIR / "group_level" / "isc_clean.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_isc.to_csv(out_csv, index=False)

    # Plot
    plt.figure(figsize=FIGSIZE)
    sns.barplot(
        data=df_isc, x="network", y="isc",
        palette=NETWORK_COLORS, edgecolor="black"
    )

    plt.title("Inter-Subject Correlation (ISC) — Yeo-7 Networks", fontsize=FONT+2)
    plt.xlabel("Yeo Network", fontsize=FONT)
    plt.ylabel("ISC", fontsize=FONT)
    plt.tight_layout()

    out_png = RESULTS_DIR / "figures" / "isc_clean.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=DPI, facecolor="white", edgecolor="white")
    plt.close()

    print(f"[GROUP] Saved clean ISC → {out_png}")
    return df_isc


# ------------------------------------------------------
# CLI
# ------------------------------------------------------

def _parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="*", help="Specify subjects manually")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Auto-detect subjects
    if args.subjects:
        subjects = args.subjects
    else:
        files = glob.glob(str(PROCESSED_DIR / "timeseries" / "sub-*_yeo7_timeseries_clean.csv"))
        subjects = [Path(f).stem.split("_")[0] for f in files]

    print("[INFO] Running ISC for subjects:", subjects)
    df = group_isc(subjects)
    print("Done.")
