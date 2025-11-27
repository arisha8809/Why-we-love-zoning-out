import numpy as np
import json
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from config import PROCESSED_DIR, N_FOLDS, C_SVM, N_PERMUTATIONS

def decode_subject(X, y, subj):
    print(f"[{subj}] Decoding…")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=C_SVM))

    scores = cross_val_score(clf, X, y, cv=cv)
    acc = float(scores.mean())

    # permutation test
    perms = []
    rng = np.random.default_rng(42)
    for _ in range(N_PERMUTATIONS):
        yp = rng.permutation(y)
        perms.append(cross_val_score(clf, X, yp, cv=cv).mean())
    perms = np.array(perms)
    p = (np.sum(perms >= acc) + 1) / (N_PERMUTATIONS + 1)

    # confusion matrix
    clf.fit(X, y)
    ypred = clf.predict(X)
    cm = confusion_matrix(y, ypred).tolist()

    out = {
        "subject": subj,
        "accuracy": acc,
        "perm_p": p,
        "cv_scores": scores.tolist(),
        "confusion": cm,
        "report": classification_report(y, ypred, output_dict=True)
    }

    out_path = PROCESSED_DIR / "decoding" / f"{subj}_decoding.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[{subj}] Saved decoding -> {out_path}")

    return out
