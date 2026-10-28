"""ACC + MAE-AFD evaluation for predicted FODs vs full-data MSMT-CSD GT.

Metrics:
- **Angular Correlation Coefficient (ACC)**: cosine similarity of two FODs
  in SH space. With orthonormal SH basis, the inner product of SH
  coefficients equals the L2 inner product of the spherical functions, so
  ACC is just the normalized SH-coefficient dot product per voxel.
  Aggregated: mean ACC over the WM mask (FA > 0.25 on full-data DTI).
- **MAE-AFD**: Apparent Fibre Density = the l=0 SH coefficient (= integral
  of FOD over the sphere up to a constant). Reported as mean absolute
  error between predicted and GT AFD over the same WM mask.

Inputs per subject:
- GT FODs at $D/gt_fod/{wm,gm,csf}_fod.nii.gz (gen_gt_fods.sh)
- Predicted WM FOD as a nii.gz with shape (X, Y, Z, 45) — caller provides
  the path on the command line.

Usage:
  python evaluate_fod_acc_afd.py \
      --subject 107422 --pred_wm_fod /path/to/coeffs_test.nii.gz \
      --label nesh_e1 --csv_out runs/fod_metrics.csv
"""
import argparse
import csv
from pathlib import Path

import os
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel

HCP_ROOT = Path(os.environ.get("BINR_HCP_ROOT", "/data/HCP"))
PARKMR_ROOT = Path(os.environ.get("BINR_PARKMR_ROOT", "/data/ParkMR"))
FA_WM_THRESHOLD = 0.25


def dataset_paths(dataset, subject):
    """Return (diff_root, dwi_path, bvec_path, bval_path, brain_mask_path)."""
    if dataset == "hcp":
        d = HCP_ROOT / subject / "Diffusion"
        return d, d / "data.nii.gz", d / "bvecs", d / "bvals", d / "nodif_brain_mask.nii.gz"
    if dataset == "parkmr":
        d = PARKMR_ROOT / subject
        return (d, d / "linear_corrected.nii.gz", d / "linear_corrected.bvec",
                d / "linear_corrected.bval", d / "linear_corrected_ss_mask.nii.gz")
    raise ValueError(f"unknown dataset: {dataset}")


def parse_bvals(path):
    with open(path) as f:
        return np.array([float(x) for x in f.read().split()])


def parse_bvecs(path):
    with open(path) as f:
        x, y, z = f.readlines()
    return np.stack([
        np.array([float(v) for v in x.split()]),
        np.array([float(v) for v in y.split()]),
        np.array([float(v) for v in z.split()]),
    ], axis=1)


def get_wm_mask(dataset, subject):
    """WM mask = FA > FA_WM_THRESHOLD on full-data DTI fit, intersected
    with the brain mask. Cached at $D/dti_wm_mask_fa{thresh}.nii.gz."""
    diff, dwi_p, bvec_p, bval_p, brain_p = dataset_paths(dataset, subject)
    cache = diff / f"dti_wm_mask_fa{int(FA_WM_THRESHOLD*100):02d}.nii.gz"
    if cache.is_file():
        return nib.load(cache).get_fdata().astype(bool)
    print(f"[{subject}] computing WM mask via DTI fit (cache miss)...", flush=True)
    dwi = nib.load(dwi_p).get_fdata().astype(np.float64)
    bvals = parse_bvals(bval_p)
    bvecs = parse_bvecs(bvec_p)
    brain_mask = nib.load(brain_p).get_fdata().astype(bool)
    gtab = gradient_table(bvals, bvecs=bvecs)
    fit = TensorModel(gtab).fit(dwi, mask=brain_mask)
    fa = fit.fa
    wm_mask = (fa > FA_WM_THRESHOLD) & brain_mask
    nii = nib.load(brain_p)
    out = nib.Nifti1Image(wm_mask.astype(np.uint8), affine=nii.affine, header=nii.header)
    nib.save(out, cache)
    return wm_mask


def acc_per_voxel(pred, gt, eps=1e-12):
    """Cosine similarity of SH-coef vectors, per voxel.

    pred, gt: (X, Y, Z, n_coeffs).
    Returns: (X, Y, Z) array of ACC in [-1, 1].
    """
    num = np.sum(pred * gt, axis=-1)
    den = np.sqrt(np.sum(pred * pred, axis=-1) * np.sum(gt * gt, axis=-1)) + eps
    return num / den


def afd(fod):
    """Apparent Fibre Density: l=0 coefficient (first channel)."""
    return fod[..., 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--pred_wm_fod", required=True,
                    help="Predicted WM FOD nii.gz (X, Y, Z, n_sh)")
    ap.add_argument("--label", required=True,
                    help="Method label for the row (e.g. nesh_e1, shore_pinv_e1)")
    ap.add_argument("--csv_out", required=True)
    ap.add_argument("--dataset", choices=["hcp", "parkmr"], default="hcp")
    args = ap.parse_args()

    diff, _, _, _, brain_p = dataset_paths(args.dataset, args.subject)
    gt_path = diff / "gt_fod" / "wm_fod.nii.gz"
    if not gt_path.is_file():
        raise FileNotFoundError(f"GT FOD missing: {gt_path}")
    pred_path = Path(args.pred_wm_fod)
    if not pred_path.is_file():
        raise FileNotFoundError(f"Pred FOD missing: {pred_path}")

    print(f"[{args.subject} / {args.label}] loading FODs...", flush=True)
    gt = nib.load(gt_path).get_fdata().astype(np.float64)
    pred = nib.load(pred_path).get_fdata().astype(np.float64)
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt={gt.shape}, pred={pred.shape}")

    wm_mask = get_wm_mask(args.dataset, args.subject)
    n_wm = int(wm_mask.sum())

    print(f"[{args.subject} / {args.label}] computing ACC + AFD...", flush=True)
    acc_vol = acc_per_voxel(pred, gt)
    afd_pred = afd(pred)
    afd_gt = afd(gt)

    acc_wm = float(np.mean(acc_vol[wm_mask]))
    acc_wm_std = float(np.std(acc_vol[wm_mask]))
    afd_mae_wm = float(np.mean(np.abs(afd_pred[wm_mask] - afd_gt[wm_mask])))
    afd_gt_mean_wm = float(np.mean(afd_gt[wm_mask]))
    afd_relmae_wm = afd_mae_wm / max(afd_gt_mean_wm, 1e-9)

    # Brain-wide aggregate (excludes WM threshold for context).
    brain_mask = nib.load(brain_p).get_fdata().astype(bool)
    acc_brain = float(np.mean(acc_vol[brain_mask]))
    afd_mae_brain = float(np.mean(np.abs(afd_pred[brain_mask] - afd_gt[brain_mask])))

    row = {
        "label": args.label,
        "subject": args.subject,
        "n_wm_voxels": n_wm,
        "acc_wm_mean": acc_wm,
        "acc_wm_std": acc_wm_std,
        "afd_mae_wm": afd_mae_wm,
        "afd_relmae_wm": afd_relmae_wm,
        "afd_gt_mean_wm": afd_gt_mean_wm,
        "acc_brain_mean": acc_brain,
        "afd_mae_brain": afd_mae_brain,
        "pred_path": str(pred_path),
    }

    fields = list(row.keys())
    out = Path(args.csv_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out.is_file() and out.stat().st_size > 0
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    print(f"[{args.subject} / {args.label}]   ACC_WM={acc_wm:+.4f}  "
          f"AFD_MAE_WM={afd_mae_wm:.4f} ({100*afd_relmae_wm:.1f}% of GT mean)  "
          f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
