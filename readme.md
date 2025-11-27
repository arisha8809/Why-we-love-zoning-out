```markdown
# Why We Love Zoning Out — fMRI + Deep Learning Analysis
A complete analysis pipeline exploring how the human brain responds to calm, neutral and shocking moments in a naturalistic story (Alice in Wonderland). This repository includes preprocessing, Yeo-7 network extraction, event-locked responses, ISC, decoding models, and the full written article.

---

## Overview

This project analyzes a publicly released naturalistic fMRI dataset where participants listened to *Alice in Wonderland*.  
The goal is to understand why fictional events can feel real by examining how large-scale brain networks react to different types of narrative moments.

The repository provides:

- Full preprocessing pipeline for raw fMRI  
- Extraction of Yeo-7 network timecourses  
- Event-locked responses for calm, neutral and shocking events  
- Limbic anticipation curves  
- Inter-subject correlation (ISC) and sliding-window ISC  
- LSTM decoder with leave-one-subject-out (LOSO) evaluation  
- The complete article: **"Why We Love Zoning Out"**

All analyses are fully reproducible.

---

## Project Structure

```
ALICE fMRI/
│
├── configs/
│
├── data_processed/
│   ├── decoding/
│   ├── events/
│   ├── group_event_locked/
│   ├── group_level/
│   ├── pca_features/
│   ├── preproc/
│   ├── quality_checks/
│   ├── timeseries/
│   └── tr_labels/
│
├── data_raw/
│
├── results/
│   └── figures/
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
│   └── results.py
│
├── src/
│   ├── data/
│   ├── models/
│   ├── train/
│   └── utils/
│
├── run_lstm.py
├── alice-text.txt
├── requirements.txt
└── why we love zoning out.pdf
```

---

## Dataset Source

This project uses the **Alice in Wonderland naturalistic fMRI dataset**, collected by the **Princeton Neuroscience Institute (PNI), Princeton University**.

Dataset reference:

> *Alice in Wonderland naturalistic fMRI dataset, Princeton Neuroscience Institute.*  
> Publicly available through PNI data portals and mirrors such as OpenNeuro.

This dataset is widely used for naturalistic neuroscience research.

---

## Key Analyses Included

### Event-Locked Responses  
Network activity before, during and after calm, neutral and shocking events.

### Limbic Anticipation  
Shows emotional build-up before shocking moments.

### Inter-Subject Correlation (ISC)  
Measures how similarly different participants' brains respond.

### Sliding-Window ISC  
Tracks shared attention over time.

### LSTM Decoding (LOSO)  
Uses only seven networks to predict whether a moment is calm, neutral or shocking.

---

## How to Run Everything

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess fMRI data  
Place raw data in `data_raw/` and run:
```bash
python scripts/run_all.py
```

### 3. Train the LSTM decoder
```bash
python run_lstm.py --config configs/lstm_config.yaml
```

### 4. Generate all publication figures
```bash
python scripts/results.py
```

Outputs are saved in:
```
results/figures/
```

---

## Included Article

The repository includes the complete written piece:

**`why we love zoning out.pdf`**

---

## License

This project is open-source. You may use, modify or extend the code with attribution.
```