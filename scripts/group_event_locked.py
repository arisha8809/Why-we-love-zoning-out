import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import csv
from scipy.stats import sem
from scipy.signal import savgol_filter
import glob
import argparse

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

DATA_DIR = Path("data_processed/preproc")
OUT_DIR  = Path("data_processed/group_event_locked")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMOTION_LABELS = {0: "Calm", 1: "Neutral", 2: "Shocking"}

PRE  = 5
POST = 15

FIGSIZE = (12, 6)
DPI = 300
LINE_WIDTH = 2.8
ALPHA_SHADE = 0.25
FONT = 8
EVENT_COLOR = "black"
EVENT_LINESTYLE = "--"
USE_SMOOTHING = False    # <-- turn OFF smoothing to get sharp curves

# High-contrast colors (matplotlib default palette)
NETWORK_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
]


# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------

def extract_event_windows(ts, labels, pre=PRE, post=POST):
    T = ts.shape[0]
    events = {lab: [] for lab in EMOTION_LABELS.keys()}
    win_len = pre + post

    for t in range(pre, T - post):
        lab = int(labels[t])
        if lab not in events:
            continue

        win = ts[t-pre : t+post]
        if win.shape[0] == win_len:
            events[lab].append(win)

    return events


def aggregate_group_windows(subj_windows):
    if len(subj_windows) == 0:
        return None
    return np.concatenate(subj_windows, axis=0)


# ----------------------------------------------------------
# Main Function
# ----------------------------------------------------------

def group_event_locked(subjects=None, data_dir=DATA_DIR, out_dir=OUT_DIR, pre=PRE, post=POST):
    print("[GROUP] Event-locked averaging…")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect subjects
    if subjects is None:
        files = sorted(glob.glob(str(data_dir / "sub-*_preproc.npz")))
        subjects = [Path(f).stem.split("_")[0] for f in files]
    print(f"[INFO] Subjects found: {subjects}")

    group_store = {lab: [] for lab in EMOTION_LABELS.keys()}

    # Load per-subject windows
    for subj in subjects:
        path = data_dir / f"{subj}_preproc.npz"
        if not path.exists():
            print(f"[WARN] Missing {subj}, skipping.")
            continue

        dat = np.load(path)
        ts = dat["timeseries"]
        labels = dat["events"].squeeze()

        events = extract_event_windows(ts, labels, pre=pre, post=post)

        for lab in EMOTION_LABELS.keys():
            if len(events[lab]) > 0:
                group_store[lab].append(np.stack(events[lab]))

    summary = {}
    t_axis = np.arange(-pre, post)

    plt.rcParams.update({
        "font.size": FONT,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # ------------------------------------------------------
    # Plot Per Emotion
    # ------------------------------------------------------

    for lab, subj_wins in group_store.items():
        label_name = EMOTION_LABELS[lab]

        if len(subj_wins) == 0:
            print(f"[WARN] No windows for {label_name}, skipping.")
            continue

        all_wins = aggregate_group_windows(subj_wins)  # (n_events, win_len, nfeat)
        mean = all_wins.mean(axis=0)
        err  = sem(all_wins, axis=0)

        summary[label_name] = {
            "mean": mean.tolist(),
            "sem": err.tolist(),
            "n_events": int(all_wins.shape[0]),
            "n_subjects": len(subj_wins),
        }

        # ------------------ Combined Plot ------------------
        plt.figure(figsize=FIGSIZE)

        for nf in range(mean.shape[1]):
            curve = mean[:, nf]
            if USE_SMOOTHING:
                curve = savgol_filter(curve, 7, 2)

            plt.plot(
                t_axis, curve,
                color=NETWORK_COLORS[nf],
                linewidth=LINE_WIDTH,
                label=f"Network {nf+1}"
            )

            plt.fill_between(
                t_axis,
                curve - err[:, nf],
                curve + err[:, nf],
                color=NETWORK_COLORS[nf],
                alpha=ALPHA_SHADE,
            )

        # Event marker
        plt.axvline(0, color=EVENT_COLOR, linestyle=EVENT_LINESTYLE, linewidth=1.5)
        plt.text(0.3, plt.ylim()[1]*0.95, "Event", fontsize=FONT, color=EVENT_COLOR)

        plt.title(
            f"{label_name} — Event-Locked Yeo Network Responses\n"
            f"(n_events={all_wins.shape[0]}, n_subjects={len(subj_wins)})",
            fontsize=FONT+3
        )

        plt.xlabel("TR relative to event", fontsize=FONT)
        plt.ylabel("Network activation (a.u.)", fontsize=FONT)
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=FONT-2)

        plt.tight_layout()

        save_path = out_dir / f"event_locked_{label_name}_clean.png"
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="white")
        plt.close()
        print(f"[OK] Saved clean plot: {save_path}")

    # Save summary JSON
    json_path = out_dir / "event_locked_summary.json"
    json.dump(summary, open(json_path, "w"), indent=2)
    print(f"[OK] Saved JSON summary: {json_path}")

    return summary


# ----------------------------------------------------------
# CLI Entrypoint
# ----------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="*", help="Explicit subject list (optional)")
    p.add_argument("--pre", type=int, default=PRE, help="TRs before event")
    p.add_argument("--post", type=int, default=POST, help="TRs after event")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    subs = args.subjects if args.subjects else None
    summary = group_event_locked(subjects=subs, pre=args.pre, post=args.post)
    print("Done.")
