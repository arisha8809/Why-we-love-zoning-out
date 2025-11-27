import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from config import *
import json

# Load precomputed sentence-level tension
df_sent = pd.read_csv(PROCESSED_DIR / "events" / "alice_sentence_tension.csv")
sentences = df_sent["sentence"].tolist()
labels_sent = df_sent["label"].values

def generate_tr_labels(subj, TR, n_trs):
    print(f"[{subj}] Generating TR labels…")

    total_duration = TR * n_trs
    num_sent = len(sentences)
    sent_dur = total_duration / num_sent

    onsets = np.arange(num_sent) * sent_dur
    offsets = onsets + sent_dur

    TR_labels = np.zeros(n_trs, dtype=int)

    for i in range(num_sent):
        st = onsets[i] + HRF_SHIFT_SEC
        en = offsets[i] + HRF_SHIFT_SEC

        tr_start = max(0, int(st // TR))
        tr_end   = min(n_trs - 1, int(np.ceil(en / TR)))
        TR_labels[tr_start:tr_end+1] = labels_sent[i]

    out_csv = PROCESSED_DIR / "tr_labels" / f"{subj}_TR_labels.csv"
    pd.DataFrame({"TR_label": TR_labels}).to_csv(out_csv, index_label="TR")
    print(f"[{subj}] Saved TR labels -> {out_csv}")

    return TR_labels
