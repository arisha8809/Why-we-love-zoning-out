import json
from nilearn import image, plotting, datasets
from nilearn.masking import compute_epi_mask
from config import *
import matplotlib.pyplot as plt

# Load Yeo atlas globally so repeated runs are faster
yeo = datasets.fetch_atlas_yeo_2011(n_networks=YEON_NETWORKS, thickness="thick")
yeo_img = image.index_img(image.load_img(yeo["maps"]), 0)

def preprocess_subject(bold_path):
    bold_path = Path(bold_path)
    subj = bold_path.stem.split("_")[0]
    out_dir = PROCESSED_DIR / "quality_checks"
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"[{subj}] Loading fMRI…")
    img = image.load_img(bold_path)

    # Determine TR
    hdr = img.header.get_zooms()
    TR = hdr[3] if len(hdr) >= 4 and hdr[3] > 0 else TR_FALLBACK
    print(f"[{subj}] TR = {TR:.2f}s")

    # Remove first TRs
    if REMOVE_INITIAL_TR:
        img = image.index_img(img, slice(REMOVE_INITIAL_TR, None))

    # Brain mask
    print(f"[{subj}] Computing mask…")
    mask = compute_epi_mask(img)

    # Resample Yeo atlas
    print(f"[{subj}] Resampling Yeo atlas…")
    yeo_res = image.resample_to_img(yeo_img, img, interpolation="nearest")

    # Apply mask
    yeo_masked = image.math_img("atlas * (mask>0)", atlas=yeo_res, mask=mask)

    # Save visualization
    fig_path = out_dir / f"{subj}_yeo_mask.png"
    plotting.plot_roi(yeo_masked, title=f"{subj} Yeo masked")
    plt.savefig(fig_path); plt.close()

    # Save basic metadata
    meta = {
        "subject": subj,
        "TR": TR,
        "n_trs": img.shape[-1],
        "mask_path": str(fig_path)
    }
    json.dump(meta, open(PROCESSED_DIR / "quality_checks" / f"{subj}_meta.json", "w"), indent=2)

    return img, mask, yeo_masked, TR
