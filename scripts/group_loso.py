import numpy as np
import json
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import PROCESSED_DIR, RESULTS_DIR, C_SVM

def run_loso(subjects):
    print("[GROUP] LOSO decoding…")
    all_X = []
    all_y = []

    for subj in subjects:
        X = np.load(PROCESSED_DIR / "pca_features" / f"{subj}_PCA100.npy")
        y = np.loadtxt(PROCESSED_DIR / "tr_labels" / f"{subj}_TR_labels.csv", delimiter=",", skiprows=1, usecols=1)
        minlen = min(len(X), len(y))
        all_X.append(X[:minlen])
        all_y.append(y[:minlen])

    loso = []
    for i, subj in enumerate(subjects):
        X_train = np.vstack([all_X[j] for j in range(len(subjects)) if j != i])
        y_train = np.hstack([all_y[j] for j in range(len(subjects)) if j != i])
        X_test = all_X[i]
        y_test = all_y[i]

        clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=C_SVM))
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        loso.append(acc)
        print(f"  {subj} → {acc:.3f}")

    out = {
        "subjects": subjects,
        "loso_acc": loso,
        "mean_loso": float(np.mean(loso))
    }
    json.dump(out, open(PROCESSED_DIR / "group_level" / "LOSO.json", "w"), indent=2)
    return out
