"""Single-shell DTI fitting + tractography-relevant angular error.

Used by the ParkMR evaluator as a downstream-task proxy: fit a diffusion
tensor per voxel from a single shell, extract principal eigenvector e₁,
and compare predicted vs ground-truth e₁ as a mean angular error in
degrees over white-matter (FA-thresholded) voxels.

Mathematically: at a fixed b-value with N gradient directions,
    log(S_i / S_0) = −b · q_i^T · D · q_i      for i = 1..N
Linear in 6 unique components of the symmetric tensor D. Solved per
voxel by a single least-squares fit; closed-form, fast.
"""
from __future__ import annotations

import numpy as np


def _design_matrix(bvecs: np.ndarray, b_value: float) -> np.ndarray:
    """Single-shell DTI design matrix X such that log(S/S0) = X · d_vec.

    bvecs: [N, 3] unit gradient directions for the shell.
    b_value: scalar shell b-value.

    Returns: X [N, 6] where d_vec = (D_xx, D_yy, D_zz, D_xy, D_xz, D_yz).
    Off-diagonal columns include the factor of 2 from q^T D q expansion.
    """
    qx, qy, qz = bvecs[:, 0], bvecs[:, 1], bvecs[:, 2]
    X = -float(b_value) * np.stack(
        [qx * qx, qy * qy, qz * qz, 2 * qx * qy, 2 * qx * qz, 2 * qy * qz],
        axis=1,
    )
    return X.astype(np.float32)


def fit_dti_single_shell(signal: np.ndarray, b_value: float,
                         bvecs: np.ndarray, S0: np.ndarray,
                         clip_min: float = 1.0) -> np.ndarray:
    """Fit symmetric DTI tensor per voxel from single-shell signal.

    signal: [V, N] DWI signal at the N directions of the shell
    b_value: scalar shell b-value
    bvecs:   [N, 3] unit gradient directions
    S0:      [V] estimated unattenuated signal per voxel
    clip_min: clip signal to this floor before log to avoid −inf

    Returns: D [V, 3, 3] symmetric tensors. Voxels with degenerate signal
    (S0 ≤ 0 or signal ≤ 0) get a small isotropic tensor as fallback.
    """
    V = signal.shape[0]
    S = np.maximum(signal.astype(np.float32), clip_min)
    S0_safe = np.maximum(S0.astype(np.float32), 1.0)
    Y = np.log(S / S0_safe[:, None])                 # [V, N]
    X = _design_matrix(bvecs, b_value)               # [N, 6]
    # Solve X · d = Y^T  →  d [6, V]
    d, *_ = np.linalg.lstsq(X, Y.T, rcond=None)
    d = d.T                                          # [V, 6]
    D = np.zeros((V, 3, 3), dtype=np.float32)
    D[:, 0, 0] = d[:, 0]
    D[:, 1, 1] = d[:, 1]
    D[:, 2, 2] = d[:, 2]
    D[:, 0, 1] = D[:, 1, 0] = d[:, 3]
    D[:, 0, 2] = D[:, 2, 0] = d[:, 4]
    D[:, 1, 2] = D[:, 2, 1] = d[:, 5]
    return D


def dti_eigen(D: np.ndarray):
    """Eigendecomp + scalar maps.

    D: [V, 3, 3] symmetric.
    Returns dict with:
      e1   [V, 3]  principal eigenvector (largest eigenvalue)
      eig  [V, 3]  eigenvalues sorted descending
      MD   [V]
      FA   [V]
    """
    evals, evecs = np.linalg.eigh(D.astype(np.float64))   # ascending
    evals = evals[:, ::-1].astype(np.float32)             # descending
    evecs = evecs[:, :, ::-1].astype(np.float32)
    e1 = evecs[:, :, 0]
    MD = evals.mean(axis=1)
    diff_sq = (evals - MD[:, None]) ** 2
    denom = np.maximum((evals ** 2).sum(axis=1), 1e-12)
    FA = np.sqrt(1.5 * diff_sq.sum(axis=1) / denom)
    return {"e1": e1, "eig": evals, "MD": MD, "FA": FA}


def angular_error_deg(v_pred: np.ndarray, v_gt: np.ndarray) -> np.ndarray:
    """Per-voxel antipodal-aware angular error in degrees."""
    v_pred = v_pred / np.maximum(np.linalg.norm(v_pred, axis=-1, keepdims=True), 1e-8)
    v_gt = v_gt / np.maximum(np.linalg.norm(v_gt, axis=-1, keepdims=True), 1e-8)
    cos = np.abs((v_pred * v_gt).sum(axis=-1))
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))


def _cache_path(cache_dir, subject: str, b_value: float, fa_threshold: float):
    """Stable on-disk path for the cached GT DTI fit."""
    from pathlib import Path
    cd = Path(cache_dir)
    cd.mkdir(parents=True, exist_ok=True)
    fname = (f"gt_dti_subject={subject}_b={int(round(b_value))}"
             f"_fa{fa_threshold:.2f}.npz")
    return cd / fname


def load_gt_reference(cache_dir, subject: str, b_value: float,
                      fa_threshold: float):
    """Return cached GT DTI dict if it exists, else None."""
    p = _cache_path(cache_dir, subject, b_value, fa_threshold)
    if not p.exists():
        return None
    z = np.load(str(p))
    return {
        "e1_gt": z["e1_gt"],
        "FA_gt": z["FA_gt"],
        "MD_gt": z["MD_gt"],
        "fa_mask": z["fa_mask"].astype(bool),
        "b_value": float(z["b_value"]),
        "fa_threshold": float(z["fa_threshold"]),
        "cached_path": str(p),
    }


def save_gt_reference(gt_ref: dict, cache_dir, subject: str):
    """Persist GT DTI fit to disk."""
    p = _cache_path(cache_dir, subject, gt_ref["b_value"], gt_ref["fa_threshold"])
    np.savez(
        str(p),
        e1_gt=gt_ref["e1_gt"],
        FA_gt=gt_ref["FA_gt"],
        MD_gt=gt_ref["MD_gt"],
        fa_mask=gt_ref["fa_mask"],
        b_value=gt_ref["b_value"],
        fa_threshold=gt_ref["fa_threshold"],
    )
    return p


def fit_gt_reference(gt_signal: np.ndarray, b_value: float,
                     bvecs: np.ndarray, S0: np.ndarray,
                     fa_threshold: float = 0.2):
    """Fit DTI on GT signal once; return e1_GT, FA mask, and scalar maps.

    Used as the cached ground-truth reference frame. Per program.md §8.1
    this is a one-time fit on GT data, not a per-voxel model parameter
    cache — purely a baseline for angular-error comparison.
    """
    D_gt = fit_dti_single_shell(gt_signal, b_value, bvecs, S0)
    eg = dti_eigen(D_gt)
    fa_mask = eg["FA"] >= float(fa_threshold)
    return {
        "e1_gt": eg["e1"],
        "FA_gt": eg["FA"],
        "MD_gt": eg["MD"],
        "fa_mask": fa_mask,
        "b_value": float(b_value),
        "fa_threshold": float(fa_threshold),
    }


def angular_error_summary(e1_pred: np.ndarray, gt_ref: dict,
                          fa_pred: np.ndarray = None,
                          md_pred: np.ndarray = None) -> dict:
    """Compute mean/median angular error and optional FA/MD agreement.

    e1_pred: [V, 3] predicted principal directions (any source: derived
             from model output, or from a DTI fit on predicted signal).
    gt_ref:  dict from fit_gt_reference(...).
    fa_pred / md_pred: optional [V] predicted scalars to compare.
    """
    e1_gt = gt_ref["e1_gt"]
    fa_mask = gt_ref["fa_mask"]
    ang = angular_error_deg(e1_pred, e1_gt)
    n_total = int(ang.shape[0])
    n_wm = int(fa_mask.sum())
    out = {
        "shell_b_value": gt_ref["b_value"],
        "fa_threshold": gt_ref["fa_threshold"],
        "n_voxels_total": n_total,
        "n_voxels_fa_passed": n_wm,
        "mean_ang_err_deg": float(ang.mean()),
        "median_ang_err_deg": float(np.median(ang)),
        "mean_ang_err_deg_wm": float(ang[fa_mask].mean()) if n_wm > 0 else float("nan"),
        "median_ang_err_deg_wm": float(np.median(ang[fa_mask])) if n_wm > 0 else float("nan"),
        "fa_gt_mean_wm": float(gt_ref["FA_gt"][fa_mask].mean()) if n_wm > 0 else float("nan"),
        "md_gt_mean_wm": float(gt_ref["MD_gt"][fa_mask].mean()) if n_wm > 0 else float("nan"),
    }
    if fa_pred is not None and n_wm > 0:
        out["fa_pred_mean_wm"] = float(fa_pred[fa_mask].mean())
    if md_pred is not None and n_wm > 0:
        out["md_pred_mean_wm"] = float(md_pred[fa_mask].mean())
    return out


def extract_e1_pred_dti_fit(pred_signal: np.ndarray, b_value: float,
                            bvecs: np.ndarray, S0: np.ndarray):
    """Fallback path: fit DTI on predicted signal and return its e1.

    Use this when the model doesn't expose a native orientation (SHORE,
    Bessel-basis, anything that's just signal-coefficient projection).
    """
    D_pred = fit_dti_single_shell(pred_signal, b_value, bvecs, S0)
    ep = dti_eigen(D_pred)
    return ep["e1"], ep["FA"], ep["MD"]
