"""
results.py
----------

1. Event-locked Yeo-7 responses
2. ISC (Inter-subject correlation) barplot
3. Sliding-window ISC timecourse
4. Pre-event limbic anticipation curves
5. LSTM LOSO decoding accuracy
6. Shock–Calm brain map (Yeo atlas)
7. Shock–Calm barplot

All figures saved to: results/figures/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import sem
from scipy.signal import savgol_filter
import glob
import json

from nilearn import datasets, plotting
import nibabel as nib

from config import PROCESSED_DIR, RESULTS_DIR


# -----------------------------------------------------------
# PATHS
# -----------------------------------------------------------

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

DATA_PREPROC = Path("data_processed/preproc")
TIMESERIES_DIR = Path("data_processed/timeseries")
EVENT_LOCKED_DIR = Path("data_processed/group_event_locked")
DECODE_DIR = Path("data_processed/decoding/lstm_outputs")

EMOTIONS = {0:"Calm",1:"Neutral",2:"Shocking"}
NETWORKS = [f"Net{i}" for i in range(1,8)]

COLORS = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728",
    "#9467bd","#8c564b","#e377c2"
]


# -----------------------------------------------------------
# 1. EVENT-LOCKED FIGURES
# -----------------------------------------------------------

def plot_event_locked():
    print("[1] Event-locked plots…")

    summary_path = EVENT_LOCKED_DIR / "event_locked_summary.json"
    if not summary_path.exists():
        print("Missing event-locked summary.")
        return
    
    summary = json.load(open(summary_path))

    for emotion, data in summary.items():
        mean = np.array(data["mean"])   
        err  = np.array(data["sem"])
        t_axis = np.arange(-5, 15)

        plt.figure(figsize=(12,6))
        for i in range(7):
            smooth = savgol_filter(mean[:,i],7,2)
            plt.plot(t_axis, smooth, color=COLORS[i], linewidth=2.2, label=f"Net{i+1}")
            plt.fill_between(t_axis, smooth-err[:,i], smooth+err[:,i], color=COLORS[i], alpha=0.18)

        plt.axvline(0, linestyle="--", color="black", linewidth=1.5)
        plt.title(f"{emotion} — Event-Locked Yeo Networks", fontsize=16)
        plt.xlabel("TR relative to event")
        plt.ylabel("Activation")
        plt.legend(loc="upper left", bbox_to_anchor=(1.02,1))
        plt.tight_layout()
        
        out = FIGURES / f"event_locked_{emotion}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


# -----------------------------------------------------------
# 2. ISC BARPLOT
# -----------------------------------------------------------

def plot_isc():
    print("[2] ISC plot…")

    isc_path = Path("data_processed/group_level/isc_clean.csv")
    if not isc_path.exists():
        print("Missing ISC CSV.")
        return

    df = pd.read_csv(isc_path)
    
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x="network", y="isc", palette=COLORS, edgecolor="black")
    plt.title("Inter-Subject Correlation (ISC) — Yeo-7 Networks", fontsize=16)
    plt.xlabel("Network")
    plt.ylabel("ISC")
    plt.tight_layout()

    out = FIGURES / "isc_barplot.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


# -----------------------------------------------------------
# 3. SLIDING-WINDOW ISC TIMECOURSE
# -----------------------------------------------------------

def plot_sliding_isc():
    print("[3] Sliding-window ISC…")

    files = sorted(glob.glob(str(TIMESERIES_DIR / "sub-*_yeo7_timeseries_clean.csv")))
    mats = []
    for f in files:
        df = pd.read_csv(f)
        df = df.iloc[:,1:]
        mats.append(df.values)

    T = min([m.shape[0] for m in mats])
    mats = [m[:T] for m in mats]
    S = len(mats)

    WIN = 20
    isc_time = []

    for t in range(0, T-WIN):
        W = np.array([m[t:t+WIN] for m in mats])
        net_isc = []

        for n in range(7):
            corr_sum = 0
            pairs = 0
            for i in range(S):
                for j in range(i+1, S):
                    corr_sum += np.corrcoef(W[i,:,n], W[j,:,n])[0,1]
                    pairs+=1
            net_isc.append(corr_sum / pairs)
        isc_time.append(np.mean(net_isc))

    isc_time = np.array(isc_time)

    plt.figure(figsize=(12,5))
    plt.plot(isc_time, color="purple")
    plt.title("Sliding-Window ISC Timecourse", fontsize=16)
    plt.xlabel("Time (TR)")
    plt.ylabel("ISC")
    plt.tight_layout()

    out = FIGURES / "sliding_isc_timecourse.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


# -----------------------------------------------------------
# 4. PRE-EVENT ANTICIPATION CURVE
# -----------------------------------------------------------

def plot_pre_event():
    print("[4] Pre-event limbic anticipation…")

    PRE = 5
    NET = 5  # Limbic = Net6 (index 5)

    files = sorted(glob.glob(str(DATA_PREPROC / "sub-*_preproc.npz")))
    curves = {0:[],1:[],2:[]}

    for f in files:
        d = np.load(f)
        ts = d["timeseries"]
        labels = d["events"].squeeze()

        for t in range(PRE, len(ts)):
            lab = int(labels[t])
            pre_win = ts[t-PRE:t, NET]
            curves[lab].append(pre_win)

    t_axis = np.arange(-PRE, 0)

    plt.figure(figsize=(10,6))
    for lab, arr in curves.items():
        if len(arr)==0: continue
        mean = np.vstack(arr).mean(axis=0)
        plt.plot(t_axis, mean, linewidth=2, label=EMOTIONS[lab])

    plt.axvline(0, linestyle="--", color="black")
    plt.title("Pre-Event Anticipation (Limbic Network)", fontsize=16)
    plt.xlabel("TR before event")
    plt.ylabel("Activation")
    plt.legend()
    plt.tight_layout()

    out = FIGURES / "pre_event_anticipation.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


# -----------------------------------------------------------
# 5. LSTM LOSO ACCURACY
# -----------------------------------------------------------

def plot_lstm_loso():
    print("[5] LSTM LOSO accuracy…")

    pattern = PROCESSED_DIR / "decoding" / "lstm_outputs" / "lstm_fold_sub-*.json"
    files = sorted(glob.glob(str(pattern)))

    if len(files)==0:
        print("No LSTM folds found.")
        return

    subj_ids = []
    accs = []

    for f in files:
        with open(f,"r") as fp:
            dat = json.load(fp)
        left_out = dat.get("subject_left_out")
        acc = dat.get("metrics",{}).get("trial_accuracy")
        if left_out is None or acc is None:
            continue
        subj_ids.append(left_out)
        accs.append(acc)

    subj_ids = np.array(subj_ids)
    accs = np.array(accs)
    order = np.argsort(subj_ids)
    subj_ids = subj_ids[order]
    accs = accs[order]

    mean_acc = accs.mean()

    plt.figure(figsize=(8,4))
    x = np.arange(len(subj_ids))
    plt.bar(x, accs)
    plt.axhline(1/3, color="red", linestyle="--", label="Chance")
    plt.axhline(mean_acc, color="black", linestyle="-.", label=f"Mean={mean_acc:.2f}")

    plt.xticks(x, [f"S{sub:02d}" for sub in subj_ids], rotation=45)
    plt.ylabel("Accuracy")
    plt.title("LSTM Decoding (LOSO)")
    plt.legend()
    plt.tight_layout()

    out = FIGURES / "lstm_loso_accuracy.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


# -----------------------------------------------------------
# 6. Shock–Calm BRAIN MAP
# -----------------------------------------------------------

def load_event_summary():
    summary_path = EVENT_LOCKED_DIR / "event_locked_summary.json"
    return json.load(open(summary_path))


def compute_shock_calm_effect(summary, post_start=5, post_end=11):
    calm = np.array(summary["Calm"]["mean"])
    shock = np.array(summary["Shocking"]["mean"])
    calm_avg = calm[post_start:post_end].mean(axis=0)
    shock_avg = shock[post_start:post_end].mean(axis=0)
    return shock_avg - calm_avg


def plot_brain_shock_minus_calm(shock_effect):
    print("[6] Shock–Calm brain map…")

    yeo = datasets.fetch_atlas_yeo_2011(n_networks=7)
    atlas_path = yeo["maps"]

    atlas_img = nib.load(atlas_path)
    atlas = atlas_img.get_fdata()

    data_vol = np.zeros_like(atlas)
    for k in range(7):
        data_vol[atlas == (k+1)] = shock_effect[k]

    out_img = nib.Nifti1Image(data_vol, atlas_img.affine, atlas_img.header)
    vmax = max(abs(shock_effect).max(), 1e-6)

    display = plotting.plot_glass_brain(
        out_img,
        display_mode="lyrz",
        cmap="coolwarm",
        vmax=vmax,
        vmin=-vmax,
        title="Shock > Calm (Yeo-7)"
    )

    out = FIGURES / "brain_shock_minus_calm.png"
    display.savefig(out, dpi=300)
    display.close()
    print(f"Saved: {out}")


# -----------------------------------------------------------
# 7. Shock–Calm BARPLOT
# -----------------------------------------------------------

def plot_shock_calm_bar(shock_effect):
    print("[7] Shock–Calm barplot…")

    plt.figure(figsize=(6,4))
    sns.barplot(x=NETWORKS, y=shock_effect, edgecolor="black")
    plt.axhline(0, color="black")
    plt.ylabel("Shock − Calm")
    plt.title("Shock–Calm Effect per Network")
    plt.tight_layout()

    out = FIGURES / "shock_calm_barplot.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


# -----------------------------------------------------------
# RUN EVERYTHING
# -----------------------------------------------------------

if __name__ == "__main__":
    print("=== Generating ALL RESULTS ===")
    plot_event_locked()
    plot_isc()
    plot_sliding_isc()
    plot_pre_event()
    plot_lstm_loso()

    summary = load_event_summary()
    effect = compute_shock_calm_effect(summary)
    plot_brain_shock_minus_calm(effect)
    plot_shock_calm_bar(effect)

    print("\nAll results saved to results/figures/")
