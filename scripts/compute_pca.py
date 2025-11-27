import numpy as np
from nilearn.image import smooth_img
from nilearn.masking import apply_mask
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from config import PROCESSED_DIR, SMOOTH_FWHM, N_PCA
import json

def compute_pca(img, mask, subj):
    print(f"[{subj}] PCA…")

    smoothed = smooth_img(img, fwhm=SMOOTH_FWHM)
    X = apply_mask(smoothed, mask)

    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=N_PCA, svd_solver="randomized", random_state=42)
    X_pca = pca.fit_transform(Xs)

    out_npy = PROCESSED_DIR / "pca_features" / f"{subj}_PCA{N_PCA}.npy"
    np.save(out_npy, X_pca)

    json.dump({
        "explained_var": pca.explained_variance_ratio_.tolist()
    }, open(out_npy.with_suffix(".json"), "w"), indent=2)

    print(f"[{subj}] Saved PCA -> {out_npy}")
    return X_pca
