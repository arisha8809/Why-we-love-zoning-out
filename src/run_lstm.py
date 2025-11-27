# src/run_lstm.py
import argparse
import yaml
import os
import glob
from src.utils.preprocessing import build_dataset_from_subjects
from src.train.train_lstm import lstm_train_pipeline

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Collect subject preprocessed files:
    # The config expects 'subject_glob' like "data_processed/preproc/sub-*_preproc.npz"
    subj_glob = cfg.get('subject_glob')
    if not subj_glob:
        raise ValueError("Please set 'subject_glob' in the config YAML to point to preprocessed subject files.")
    subject_files = sorted(glob.glob(subj_glob))
    if len(subject_files) == 0:
        raise ValueError(f"No subject files found for glob: {subj_glob}")

    print(f"Found {len(subject_files)} subject files.")
    X, y, groups, metas = build_dataset_from_subjects(subject_files,
                                                     seq_len=cfg['seq_len'],
                                                     stride=cfg['stride'],
                                                     label_mode=cfg.get('label_mode', 'last'),
                                                     pca_components=None)
    print("Built windows:", X.shape, y.shape, groups.shape)

    results = lstm_train_pipeline(X, y, groups, metas, subject_files, cfg)
    print("All folds done. Summary:")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
