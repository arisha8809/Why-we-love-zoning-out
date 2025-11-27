from pathlib import Path

# Root directory
PROJECT_ROOT = Path.cwd()

RAW_DATA_DIR = PROJECT_ROOT / "data_raw"
PROCESSED_DIR = PROJECT_ROOT / "data_processed"
RESULTS_DIR = PROJECT_ROOT / "results"

TEXT_PATH = PROJECT_ROOT / "alice_text.txt"

# fMRI read pattern (recursive search)
FMRI_PATTERN = "**/sub-*.nii*"

# Analysis params
TR_FALLBACK = 2.0
REMOVE_INITIAL_TR = 4
HRF_SHIFT_SEC = 4

YEON_NETWORKS = 7
HP_FILTER = 0.008
SMOOTH_FWHM = 6
N_PCA = 100

N_FOLDS = 5
C_SVM = 1.0
N_PERMUTATIONS = 500

EVENT_WINDOW = (-6, 16)
PEAK_PERCENTILE = 85
