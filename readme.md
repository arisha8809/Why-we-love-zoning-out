# Why We Love Zoning Out — fMRI + Deep Learning Analysis  

A complete analysis pipeline exploring how the human brain responds to calm, neutral, and shocking moments using naturalistic fMRI data.

---

## Overview  

This project analyzes a publicly released naturalistic fMRI dataset where participants listened to *Alice in Wonderland*.  
The goal is to understand why fictional events can feel real by examining how large-scale brain networks react to different narrative events.

The repository provides:

- Full preprocessing pipeline for raw fMRI  
- Extraction of Yeo-7 network timeseries  
- Event-locked responses for calm, neutral, and shocking events  
- Limbic anticipation curves  
- Inter-subject correlation (ISC) and sliding-window ISC  
- LSTM decoder with leave-one-subject-out (LOSO) evaluation  
- The complete article: **“Why We Love Zoning Out”**  

All analyses are fully reproducible.

---

## Dataset Download  

This project uses the **ALICE fMRI dataset**, a naturalistic story-listening dataset.

Download from:

🔗 **OpenNeuro – ALICE fMRI dataset**  
https://openneuro.org/datasets/ds002322/versions/1.0.0

After downloading, place the data inside:

```
data_raw/
```

This folder is **ignored by Git**.

---

## Project Structure  

```
ALICE_fMRI/
│
├── configs/
├── data_processed/
├── decoding/
├── events/
├── group_event_locked/
├── group_level/
├── pca_features/
├── preproc/
├── quality_checks/
├── timeseries/
├── tr_labels/
│
├── data_raw/                # (Ignored) raw fMRI data
│
├── results/
│
├── scripts/
│   ├── compute_pca.py
│   ├── config.py
│   ├── decode_subject.py
│   ├── extract_timeseries.py
│   ├── generate_preproc_npz.py
│   ├── generate_sentence_tension.py
│   ├── generate_tr_labels.py
│   ├── group_event_locked.py
│   ├── group_isc.py
│   ├── group_loso.py
│   ├── preprocess_subject.py
│   ├── run_all.py
│   ├── run_one_subject.py
│   └── alice-text.txt
│
├── src/
├── train/
└── utils/
```

---

## Running the Pipeline  

Run the entire pipeline using:

```bash
python run_all.py
```

This automatically:
- Preprocesses raw data  
- Extracts Yeo-7 network signals  
- Generates TR labels  
- Computes ISC and sliding-window ISC  
- Trains/validates the LSTM decoder  
- Saves all final figures and metrics  

---

## Citation  

If you use this repository, please cite the ALICE dataset and this analysis pipeline.

---

## Contact  

For questions or collaboration:  
📧 arishagupta98@gmail.com
