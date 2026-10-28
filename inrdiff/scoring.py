"""Scorers and non-learned baselines used alongside the INR results.

Four entry points, exposed as subcommands of evaluate.py:

  upsample    classical tri-cubic interpolation, the no-ML super-resolution baseline
  dti-extrap  mono-exponential DTI extrapolation to an unacquired shell
  frame       brute-force search over signed axis permutations of an fODF export
  fodf        ACC and AFD against an MSMT-CSD reference (needs the scilpy environment)

Each keeps the argument handling it was written with; evaluate.py only dispatches.
"""
from __future__ import annotations

from __future__ import annotations
from datetime import UTC, datetime
from datetime import datetime, timezone
from pathlib import Path
from scipy import ndimage
from inrdiff.utils import load_data
import argparse
import argparse, json, os, sys
import importlib.util
import itertools
import json
import math
import nibabel as nib
import numpy as np
import os

HCP_ROOT = os.environ.get("BINR_HCP_ROOT", os.environ.get("BINR_HCP_ROOT", "/data/HCP"))
B0_THRESHOLD = 10.0
ORDERS = {0: "nearest", 1: "trilinear", 3: "cubic B-spline", 5: "quintic spline"}
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DATASET_CFG = {
    "hcp": dict(
        data_paths=lambda s: {
            "data": fos.environ.get("BINR_HCP_ROOT", "/data/HCP") + "/{s}/Diffusion/data.nii.gz",
            "mask": fos.environ.get("BINR_HCP_ROOT", "/data/HCP") + "/{s}/Diffusion/nodif_brain_mask.nii.gz",
            "bvec": fos.environ.get("BINR_HCP_ROOT", "/data/HCP") + "/{s}/Diffusion/bvecs",
            "bval": fos.environ.get("BINR_HCP_ROOT", "/data/HCP") + "/{s}/Diffusion/bvals"},
        train_shells=(1000, 3000), heldout_shells=(2000,), b0_threshold=50, bval_tol=100),
    "parkmr": dict(
        data_paths=lambda s: {
            "data": fos.environ.get("BINR_PARKMR_ROOT", "/data/ParkMR") + "/{s}/linear_corrected.nii.gz",
            "mask": fos.environ.get("BINR_PARKMR_ROOT", "/data/ParkMR") + "/{s}/linear_corrected_ss_mask.nii.gz",
            "bvec": fos.environ.get("BINR_PARKMR_ROOT", "/data/ParkMR") + "/{s}/linear_corrected.bvec",
            "bval": fos.environ.get("BINR_PARKMR_ROOT", "/data/ParkMR") + "/{s}/linear_corrected.bval"},
        train_shells=(1400, 2000), heldout_shells=(100, 700), b0_threshold=10, bval_tol=50),
}
UTC = timezone.utc
ABS_THR_DEFAULT = 0.07
REL_THR_DEFAULT = 0.1
WM_C00_THR_DEFAULT = 0.05


# --- from e4_classical_upsample.py ----------------------------------
def psnr_held(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-volume PSNR with the volume's own held-out range; mean over volumes.

    Mirrors experiment_hcp_superres.py:_psnr_per_volume.
    """
    out = []
    for j in range(gt.shape[1]):
        g, p = gt[:, j], pred[:, j]
        rmse = float(np.sqrt(np.mean((p - g) ** 2)))
        rng = float(g.max() - g.min())
        if rmse > 1e-12 and rng > 1e-12:
            out.append(20.0 * math.log10(rng / rmse))
    return float(np.mean(out)) if out else float("nan")


# --- from e4_classical_upsample.py ----------------------------------
def load_subject(subject: str):
    """Load once per subject: the 1.3 GB read dominates on NFS, and both factors
    use the same volume."""
    d = Path(HCP_ROOT) / subject / "Diffusion"
    bvals = np.loadtxt(str(d / "bvals")).ravel()
    keep_vol = bvals > B0_THRESHOLD                    # DWI only, as the driver does
    mask = np.asarray(nib.load(str(d / "nodif_brain_mask.nii.gz")).dataobj) > 0
    dwi = np.asarray(nib.load(str(d / "data.nii.gz")).dataobj,
                     dtype=np.float32)[..., keep_vol]
    return dwi, mask


# --- from e4_classical_upsample.py ----------------------------------
def run(subject: str, factor: int, orders=(0, 1, 3, 5), loaded=None):
    dwi, mask = loaded if loaded is not None else load_subject(subject)

    lattice = np.zeros_like(mask)
    lattice[::factor, ::factor, ::factor] = True
    train = mask & lattice
    held = mask & ~train
    hx, hy, hz = np.nonzero(held)
    # held-out full-res indices mapped into low-resolution lattice coordinates
    coords = np.stack([hx / factor, hy / factor, hz / factor])

    gt = dwi[hx, hy, hz, :]                            # [n_held, n_dwi]
    res = {}
    for o in orders:
        pred = np.empty_like(gt)
        for v in range(dwi.shape[-1]):
            low = dwi[::factor, ::factor, ::factor, v]
            pred[:, v] = ndimage.map_coordinates(low, coords, order=o, mode="nearest",
                                                 prefilter=(o > 1))
        res[o] = psnr_held(pred, gt)
        print(f"  order {o} ({ORDERS.get(o, o):<14s}) held-out PSNR {res[o]:.4f}", flush=True)
    return dict(subject=subject, factor=factor,
                n_train_voxels=int(train.sum()), n_heldout_voxels=int(held.sum()),
                n_dwi=int(dwi.shape[-1]),
                psnr_by_order={str(k): v for k, v in res.items()},
                best_order=int(max(res, key=res.get)), best_psnr=max(res.values()),
                ts=datetime.now(UTC).isoformat())


# --- from e4_classical_upsample.py ----------------------------------
def main_upsample():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--factors", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    print(f"[classical] subject={a.subject} factors={a.factors} loading...", flush=True)
    import time
    t0 = time.perf_counter()
    loaded = load_subject(a.subject)
    print(f"  loaded in {time.perf_counter()-t0:.0f}s "
          f"(shape {loaded[0].shape})", flush=True)
    for fac in a.factors:
        print(f"[classical] subject={a.subject} factor={fac}", flush=True)
        r = run(a.subject, fac, loaded=loaded)
        print(f"  train {r['n_train_voxels']} / held {r['n_heldout_voxels']} voxels, "
              f"{r['n_dwi']} DWI volumes")
        print(f"  BEST: order {r['best_order']} ({ORDERS.get(r['best_order'])}) "
              f"{r['best_psnr']:.4f} dB | CUBIC(3) {r['psnr_by_order']['3']:.4f} dB",
              flush=True)
        if a.out_dir:
            o = Path(a.out_dir) / f"{a.subject}_f{fac}.json"
            o.parent.mkdir(parents=True, exist_ok=True)
            o.write_text(json.dumps(r, indent=2))
            print(f"  wrote {o}", flush=True)


# --- from eval_dti_extrapolation_baseline.py ------------------------
def _shell_mask(bvals, shells, tol):
    out = np.zeros(bvals.shape[0], dtype=bool)
    for s in shells:
        out |= np.abs(bvals - s) <= tol
    return out


# --- from eval_dti_extrapolation_baseline.py ------------------------
def _rel_mae(pred, gt):
    mae = float(np.abs(pred - gt).mean())
    mean_sig = float(np.abs(gt).mean())
    return 100.0 * mae / max(mean_sig, 1e-12), mean_sig


# --- from eval_dti_extrapolation_baseline.py ------------------------
def run_subject(dataset, subject):
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    cfg = DATASET_CFG[dataset]
    tol = cfg["bval_tol"]
    dwi, mask, bvec, bval = load_data(cfg["data_paths"](subject))
    bval = bval.numpy().astype(np.float64)
    bvec = bvec.numpy().astype(np.float64)
    # memory-lean: brain voxels only -> [N_brain, V]
    sig = dwi[mask.bool()].float().numpy().astype(np.float64)
    del dwi

    b0 = bval <= cfg["b0_threshold"]
    train = _shell_mask(bval, cfg["train_shells"], tol)
    held = _shell_mask(bval, cfg["heldout_shells"], tol)
    S0 = sig[:, b0].mean(axis=1)
    S0 = np.maximum(S0, 1e-6)

    print(f"=== {dataset}/{subject} ===  brain={sig.shape[0]}  "
          f"b0={int(b0.sum())} train={int(train.sum())} held={int(held.sum())}")

    # Fit DTI on B0 + training shells
    fit_sel = b0 | train
    gtab_fit = gradient_table(bvals=bval[fit_sel], bvecs=bvec[fit_sel])
    tm = TensorModel(gtab_fit)
    fit = tm.fit(sig[:, fit_sel])

    # Predict each held-out shell
    out = {"per_shell": {}}
    held_pred_all, held_gt_all = [], []
    for sh in cfg["heldout_shells"]:
        idx = _shell_mask(bval, (sh,), tol)
        gtab_h = gradient_table(bvals=bval[idx], bvecs=bvec[idx])
        pred = fit.predict(gtab_h, S0=S0)        # [N_brain, n_dir_sh]
        gt = sig[:, idx]
        rm, ms = _rel_mae(pred, gt)
        out["per_shell"][str(int(sh))] = rm
        held_pred_all.append(pred); held_gt_all.append(gt)
        print(f"  held b={sh}: rel_mae_pct={rm:.3f}  mean_sig={ms:.2f}")
    pred_all = np.concatenate(held_pred_all, axis=1)
    gt_all = np.concatenate(held_gt_all, axis=1)
    rm_ov, _ = _rel_mae(pred_all, gt_all)
    out["overall_rel_mae"] = rm_ov
    print(f"  OVERALL held rel_mae_pct={rm_ov:.3f}")
    return out


# --- from eval_dti_extrapolation_baseline.py ------------------------
def main_dti_extrap():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=("hcp", "parkmr"))
    ap.add_argument("--subject", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    res = run_subject(args.dataset, args.subject)
    if args.json_out:
        p = Path(args.json_out)
        store = json.loads(p.read_text()) if p.is_file() else {}
        store.setdefault(args.dataset, {})[args.subject] = res
        p.write_text(json.dumps(store, indent=2))
        print(f"Wrote/updated {p}")


# --- from frame_search.py -------------------------------------------
def _eval_mod():
    from inrdiff import fod_acc_afd
    return fod_acc_afd


# --- from frame_search.py -------------------------------------------
def _octahedral():
    mats = []
    for perm in itertools.permutations(range(3)):
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    M = np.zeros((3, 3))
                    M[0, perm[0]] = sx; M[1, perm[1]] = sy; M[2, perm[2]] = sz
                    mats.append((perm, (sx, sy, sz), M))
    return mats


# --- from frame_search.py -------------------------------------------
def main_frame():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--subject", default="107422")
    ap.add_argument("--dataset", default="hcp")
    ap.add_argument("--max_voxels", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mod = _eval_mod()
    diff, _, _, _, _ = mod.dataset_paths(args.dataset, args.subject)
    gt = nib.load(str(diff / "gt_fod" / "wm_fod.nii.gz")).get_fdata().astype(np.float64)
    pred = nib.load(args.pred).get_fdata().astype(np.float64)
    wm = mod.get_wm_mask(args.dataset, args.subject)

    idx = np.argwhere(wm); rng = np.random.default_rng(args.seed)
    if len(idx) > args.max_voxels:
        idx = idx[rng.choice(len(idx), args.max_voxels, replace=False)]

    from dipy.data import get_sphere
    from dipy.reconst.shm import sh_to_sf
    from dipy.direction.peaks import peak_directions
    sphere = get_sphere("repulsion724")

    def sf(v):
        return sh_to_sf(v[idx[:, 0], idx[:, 1], idx[:, 2]], sphere,
                        sh_order_max=8, basis_type="tournier07", legacy=False)
    sf_gt, sf_pred = sf(gt), sf(pred)

    def top_peak(row):
        d, _, _ = peak_directions(row, sphere, relative_peak_threshold=0.5,
                                  min_separation_angle=25.0)
        return d[0] if len(d) else None

    gtp, prp = [], []
    for i in range(len(idx)):
        g, p = top_peak(sf_gt[i]), top_peak(sf_pred[i])
        if g is not None and p is not None:
            gtp.append(g); prp.append(p)
    gtp = np.array(gtp); prp = np.array(prp)
    print(f"[{args.label}] single+ principal peaks on {len(gtp)} voxels")

    def med_err(P):
        dots = np.abs(np.sum(gtp * P, axis=1))
        return np.degrees(np.arccos(np.clip(dots, 0, 1)))

    results = []
    for perm, signs, M in _octahedral():
        Pt = prp @ M.T
        e = med_err(Pt)
        results.append((np.median(e), perm, signs))
    results.sort()
    print("  best transforms (perm of xyz, sign flips) -> single-fiber median principal error:")
    for med, perm, signs in results[:5]:
        print(f"    perm={perm} signs={signs}  median={med:5.1f} deg")
    print(f"  (identity for reference: median={np.median(med_err(prp)):5.1f} deg)")


# --- from eval_fod_scilpy.py ----------------------------------------
def l_index_per_coef(n_coef: int) -> np.ndarray:
    """SH order l for each coefficient index, even orders only (symmetric fODF).

    n_coef = (L+1)(L+2)/2 for full SH; for even-only it is (L/2+1)(L+1)... we derive
    it by walking even l and counting 2l+1 coefficients each, which matches the
    MRtrix/dipy ordering used throughout this project.
    """
    out, l = [], 0
    while len(out) < n_coef:
        out.extend([l] * (2 * l + 1))
        l += 2
    return np.asarray(out[:n_coef])


# --- from eval_fod_scilpy.py ----------------------------------------
def acc_l_ge1(pred: np.ndarray, gt: np.ndarray, l_idx: np.ndarray, eps=1e-12):
    """Anderson (2005) ACC: normalised inner product over l>=1 only.

    Excluding l=0 is the whole point: it is what makes this a shape metric rather
    than one dominated by overall fibre density (and, here, by a pinned constant).
    """
    keep = l_idx >= 1
    p, g = pred[..., keep], gt[..., keep]
    num = np.sum(p * g, axis=-1)
    den = np.sqrt(np.sum(p * p, axis=-1) * np.sum(g * g, axis=-1)) + eps
    return num / den


# --- from eval_fod_scilpy.py ----------------------------------------
def nufo_afd_scilpy(fod: np.ndarray, mask: np.ndarray, abs_thr: float,
                    rel_thr: float, sh_basis: str = "tournier07",
                    nbr_processes: int = 2):
    """NuFO + AFD max/sum via scilpy's OWN functions, defaults except the threshold.

    Mirrors scilpy/cli/scil_fodf_metrics.py exactly:
        peaks_from_sh(..., relative_peak_threshold=rt, absolute_threshold=at,
                      min_separation_angle=25, normalize_peaks=False, npeaks=5)
        maps_from_sh(...)  ->  nufo_map, afd_max, afd_sum, rgb, _, _
    scilpy's defaults are rt=0.1 and at=0.0; the spec sets the ABSOLUTE threshold to
    0.07 and leaves everything else default, so rt stays 0.1. (An earlier draft of
    this file used rt=0.5 -- dipy's default and what our old scorer used -- which is
    NOT scilpy's default and would have changed every peak count.)

    sh_basis is tournier07 here, not scilpy's descoteaux07 default: both the MRtrix
    MSMT-CSD ground truth and our exported predictions are tournier07, and passing
    the wrong basis silently reinterprets the coefficients.
    """
    from dipy.data import get_sphere
    from scilpy.reconst.sh import maps_from_sh, peaks_from_sh

    sphere = get_sphere(name="repulsion724")        # scilpy default sphere
    peak_dirs, peak_values, peak_indices = peaks_from_sh(
        fod, sphere, mask=mask,
        relative_peak_threshold=rel_thr,
        absolute_threshold=abs_thr,
        min_separation_angle=25,
        normalize_peaks=False,
        npeaks=5,
        sh_basis_type=sh_basis,
        is_legacy=False,
        nbr_processes=nbr_processes,
    )
    nufo_map, afd_max, afd_sum, _rgb, _a, _b = maps_from_sh(
        fod, peak_values, peak_indices, sphere, mask=mask, sh_basis_type=sh_basis,
        nbr_processes=nbr_processes)
    return nufo_map, afd_max, afd_sum


# --- from eval_fod_scilpy.py ----------------------------------------
def main_fodf():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--abs-thr", type=float, default=ABS_THR_DEFAULT)
    ap.add_argument("--rel-thr", type=float, default=REL_THR_DEFAULT,
                    help="scilpy's relative peak threshold; its default is 0.1 and the "
                         "spec changes only the absolute threshold")
    ap.add_argument("--sh-basis", default="tournier07")
    ap.add_argument("--skip-nufo", action="store_true",
                    help="omit NuFO. Dropped on request: at the spec's absolute threshold of "
                         "0.07 it is not interpretable for our parametrisation. Unscaled, our "
                         "pinned c00 puts the fODF baseline at ~1.0 so every ripple counts "
                         "(NuFO 3.257 vs the GT's 2.018); rescaled to the GT's density the "
                         "baseline drops to ~0.043 and almost nothing clears the threshold "
                         "(0.087), because our lobes are weak relative to the isotropic part "
                         "(97%% of coefficient energy sits in c00). Neither number measures "
                         "peak count, and NuFO was also the entire runtime cost -- "
                         "peaks_from_sh over 527k voxels x 724 directions, ~50 min/subject, "
                         "against seconds for the vectorised ACC and AFD.")
    ap.add_argument("--rescale-c00", choices=("none", "global"), default="none",
                    help="rescale the PREDICTION by a single scalar so its mean c00 matches "
                         "the GT's before peak extraction. Needed for our model: our c00 is "
                         "PINNED to sqrt(4*pi)=3.5449 as a gauge fix, giving an fODF baseline "
                         "amplitude of ~1.0, while the CSD GT's c00~0.154 gives ~0.043. The "
                         "spec's ABSOLUTE threshold of 0.07 therefore sits 14x BELOW our "
                         "baseline and 1.6x ABOVE the GT's, so every ripple in our fODF counts "
                         "as a peak -- NuFO came out 3.257 vs the GT's 2.018 with only 20%% "
                         "exact agreement, which is a scale artefact, not a shape error. "
                         "GLOBAL (one scalar) rather than per-voxel: our c00 is constant and "
                         "carries no spatial information, so a per-voxel match would import "
                         "the GT's density map into our prediction. ACC (l>=1) is unaffected "
                         "either way -- it is a normalised inner product and so scale-free.")
    ap.add_argument("--nufo-max-voxels", type=int, default=50000,
                    help="cap voxels used for the NuFO estimate (0 = full mask). "
                         "NuFO agreement is a MEAN over voxels, so a random subsample is "
                         "an unbiased estimator with SE = sd/sqrt(n); at n=50k the SE on a "
                         "proportion is ~0.002, far below any effect of interest. The full "
                         "mask costs ~2.5 h/subject under this container's 4-CPU quota "
                         "(527k voxels x 724 sphere directions, twice), which is not "
                         "viable. ACC and AFD are vectorised and always use the FULL mask.")
    ap.add_argument("--nbr-processes", type=int, default=2,
                    help="scilpy worker count. Its default is ALL cores, but THIS "
                         "CONTAINER HAS A 4-CPU CGROUP QUOTA while nproc reports the "
                         "host's 48 (cpu.max = 400000 100000). Unset, scilpy launched 48 "
                         "workers against 4 cores -- a 12x oversubscription that drove "
                         "load average to 98 and hard-throttled everything. Keep this at "
                         "2-3.")
    ap.add_argument("--signs", default="1,1,1",
                    help="axis signs applied to the PREDICTION before scoring, via "
                         "fodf_frame.reflect_sh. MSMT-CSD-INR predictions need "
                         "--signs=-1,1,1: frame_search measured its single-fibre peak "
                         "error dropping 46.8 -> 6.0 deg under an x-flip, because it reads "
                         "raw FSL bvecs (image frame) while the MRtrix GT is scanner frame. "
                         "Pass with = (--signs=-1,1,1), else argparse reads it as a flag.")
    ap.add_argument("--wm-c00-thr", type=float, default=WM_C00_THR_DEFAULT)
    ap.add_argument("--brain-mask", default=None,
                    help="optional extra brain mask to intersect with the WM criterion")
    ap.add_argument("--frame-note", default="",
                    help="record why ACC is or is not interpretable for this cohort")
    ap.add_argument("--skip-acc", action="store_true",
                    help="set when the export frame is unverified (e.g. ParkMR)")
    a = ap.parse_args()

    pred = np.asarray(nib.load(a.pred).dataobj, dtype=np.float64)
    gt = np.asarray(nib.load(a.gt).dataobj, dtype=np.float64)
    if pred.shape != gt.shape:
        raise SystemExit(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    l_idx = l_index_per_coef(pred.shape[-1])

    signs = tuple(float(x) for x in a.signs.split(","))
    if signs != (1.0, 1.0, 1.0):
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from inrdiff.fodf_frame import reflect_sh
        pred = reflect_sh(pred, int(l_idx.max()), a.sh_basis, signs)
        print(f"[{a.label}] applied frame correction signs={signs} to the prediction",
              flush=True)

    # WM per the spec: GT first coefficient above threshold.
    wm = gt[..., 0] > a.wm_c00_thr
    if a.brain_mask:
        wm &= np.asarray(nib.load(a.brain_mask).dataobj) > 0
    n_wm = int(wm.sum())
    print(f"[{a.label}] WM = GT c00 > {a.wm_c00_thr}: {n_wm} voxels "
          f"({n_wm / max(wm.size, 1):.2%} of volume)", flush=True)

    res = dict(label=a.label, pred=a.pred, gt=a.gt, n_coef=int(pred.shape[-1]),
               l_max=int(l_idx.max()), abs_thr=a.abs_thr, wm_c00_thr=a.wm_c00_thr,
               n_wm_voxels=n_wm, frame_note=a.frame_note, frame_signs=list(signs),
               ts=datetime.now(UTC).isoformat())

    if a.rescale_c00 == "global":
        mp, mg = float(pred[..., 0][wm].mean()), float(gt[..., 0][wm].mean())
        factor = mg / mp if abs(mp) > 1e-12 else 1.0
        pred = pred * factor
        res["c00_rescale_factor"] = factor
        res["c00_rescale_mode"] = "global"
        print(f"  rescaled prediction by {factor:.6f} (mean c00 {mp:.4f} -> {mg:.4f}); "
              f"ACC is scale-free so unchanged", flush=True)
    else:
        res["c00_rescale_factor"] = 1.0
        res["c00_rescale_mode"] = "none"

    # AFD total -- rotation invariant, so always reported
    afd_p, afd_g = pred[..., 0][wm], gt[..., 0][wm]
    res["afd_total_pred_mean"] = float(afd_p.mean())
    res["afd_total_gt_mean"] = float(afd_g.mean())
    res["afd_total_mae"] = float(np.abs(afd_p - afd_g).mean())
    res["afd_total_relmae"] = float(res["afd_total_mae"] / max(abs(res["afd_total_gt_mean"]), 1e-9))
    res["afd_frame_dependent"] = False
    print(f"  AFD total: pred {res['afd_total_pred_mean']:.4f} gt "
          f"{res['afd_total_gt_mean']:.4f} relMAE {res['afd_total_relmae']:.4f}", flush=True)

    # ACC -- frame dependent
    if a.skip_acc:
        res["acc_l_ge1_mean"] = None
        res["acc_frame_dependent"] = True
        res["acc_skipped_reason"] = a.frame_note or "export frame unverified"
        print(f"  ACC: SKIPPED ({res['acc_skipped_reason']})", flush=True)
    else:
        acc = acc_l_ge1(pred, gt, l_idx)[wm]
        res["acc_l_ge1_mean"] = float(np.mean(acc))
        res["acc_l_ge1_median"] = float(np.median(acc))
        res["acc_frame_dependent"] = True
        print(f"  ACC (l>=1, Anderson 2005): mean {res['acc_l_ge1_mean']:.4f} "
              f"median {res['acc_l_ge1_median']:.4f}", flush=True)

    if a.skip_nufo:
        res["nufo_skipped"] = True
        print("  NuFO: SKIPPED (not interpretable at a fixed absolute threshold for a "
              "pinned-c00 fODF; see --skip-nufo)", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(f"  wrote {a.out}\nFOD_SCILPY_DONE", flush=True)
        return

    # NuFO -- rotation invariant. Estimated on a fixed random subsample of the WM
    # mask (see --nufo-max-voxels); ACC and AFD above used the full mask.
    wm_nufo = wm
    if a.nufo_max_voxels and n_wm > a.nufo_max_voxels:
        rng = np.random.default_rng(0)
        idx = np.argwhere(wm)
        sel = idx[rng.choice(len(idx), a.nufo_max_voxels, replace=False)]
        wm_nufo = np.zeros_like(wm)
        wm_nufo[sel[:, 0], sel[:, 1], sel[:, 2]] = True
        print(f"  NuFO on a {a.nufo_max_voxels} voxel subsample of {n_wm} "
              f"(seed 0; SE on a proportion ~{(0.25/a.nufo_max_voxels)**0.5:.4f})", flush=True)
    res["n_nufo_voxels"] = int(wm_nufo.sum())
    res["nufo_subsampled"] = bool(wm_nufo.sum() < n_wm)
    nufo_p, afdmax_p, afdsum_p = nufo_afd_scilpy(pred, wm_nufo, a.abs_thr, a.rel_thr, a.sh_basis, a.nbr_processes)
    nufo_g, afdmax_g, afdsum_g = nufo_afd_scilpy(gt, wm_nufo, a.abs_thr, a.rel_thr, a.sh_basis, a.nbr_processes)
    wm = wm_nufo  # the NuFO stats below are over the subsample
    res["rel_thr"] = a.rel_thr
    res["nbr_processes"] = a.nbr_processes
    res["sh_basis"] = a.sh_basis
    res["afd_sum_pred_mean"] = float(afdsum_p[wm].mean())
    res["afd_sum_gt_mean"] = float(afdsum_g[wm].mean())
    for nm, arr in (("pred", nufo_p), ("gt", nufo_g)):
        v = arr[wm]
        res[f"nufo_{nm}_mean"] = float(v.mean())
        res[f"nufo_{nm}_hist"] = {str(k): int((v == k).sum()) for k in range(0, 5)}
    res["nufo_agreement"] = float((nufo_p[wm] == nufo_g[wm]).mean())
    res["nufo_mae"] = float(np.abs(nufo_p[wm].astype(float) - nufo_g[wm]).mean())
    res["afd_max_pred_mean"] = float(afdmax_p[wm].mean())
    res["afd_max_gt_mean"] = float(afdmax_g[wm].mean())
    res["nufo_frame_dependent"] = False
    print(f"  NuFO: pred {res['nufo_pred_mean']:.3f} gt {res['nufo_gt_mean']:.3f} "
          f"exact-agreement {res['nufo_agreement']:.3f} MAE {res['nufo_mae']:.3f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"  wrote {a.out}\nFOD_SCILPY_DONE", flush=True)
