"""Benchmark pipeline: prepare a config, train an INR, score the held-out data.

One pipeline serves every reconstruction experiment in the paper. The three
drivers this replaces were 90% (HCP) and 68% (ParkMR) identical line-for-line;
what actually varied was the dataset paths, which shells are trained on and
which are held out, and whether ground-truth fODFs exist to score against.
Those are the two dataclasses below.

    from inrdiff.benchmark import configure, run, DATASETS, TASKS
    configure(DATASETS["hcp"], TASKS["shell-holdout"])
    summary = run(subject="107422")
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml

from inrdiff.acquisition_adaptation import (
    AcquisitionPool, aggregate_metrics, evaluate_all_volumes,
)
from inrdiff.main import load_config as load_train_config, train_with_config
from inrdiff.utils import extract_b0_prior, load_data

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    """Where the data lives and what a scan looks like."""
    name: str
    root_env: str                 # environment variable holding the cohort root
    default_root: str
    layout: str                   # formatted with root=..., subject=...
    default_subject: str
    shells: tuple                 # every non-b0 shell present
    has_fodf_gt: bool             # MSMT-CSD ground truth available to score against
    eval_batch_size: int | None = None   # lock only where the brain is large


@dataclass(frozen=True)
class TaskSpec:
    """Which part of the acquisition is withheld from training."""
    name: str
    train_shells: tuple           # non-b0 shells the model may see
    held_out_shells: tuple        # scored, never trained on
    train_bvec_per_shell: int | None = 81   # directions kept per training shell
    train_bvec_seed: int = 42
    epochs: int = 6
    runs_root: str = "OUTPUTS"


DATASETS = {
    "hcp": DatasetSpec(
        name="hcp", root_env="BINR_HCP_ROOT", default_root="/data/HCP",
        layout="{root}/{subject}/Diffusion", default_subject="107422",
        shells=(1000, 2000, 3000), has_fodf_gt=True, eval_batch_size=50_000,
    ),
    "parkmr": DatasetSpec(
        name="parkmr", root_env="BINR_PARKMR_ROOT",
        default_root="/data/ParkMR",
        layout="{root}/{subject}", default_subject="PD20230223Control1",
        shells=(100, 700, 1400, 2000), has_fodf_gt=False,
    ),
}

TASKS = {
    # b=2000 withheld entirely; the model must interpolate a shell it never saw.
    "shell-holdout": TaskSpec("shell-holdout", (1000, 3000), (2000,)),
    # trained on b=1000 alone; b=2000 and b=3000 are extrapolation.
    "single-shell":  TaskSpec("single-shell", (1000,), (2000, 3000)),
    # all shells present, only directions withheld (set --directions).
    "direction-budget": TaskSpec("direction-budget", (1000, 2000, 3000), ()),
    # every shell and direction trained on; scores the fit itself.
    "dense": TaskSpec("dense", (1000, 2000, 3000), (), train_bvec_per_shell=None),
    # spatial: train on a 1/factor^3 voxel subset, predict every other voxel.
    "superres": TaskSpec("superres", (1000, 2000, 3000), (), train_bvec_per_shell=None),
    # ParkMR: inner shells withheld, outer shells trained.
    "parkmr-holdout": TaskSpec("parkmr-holdout", (1400, 2000), (100, 700)),
}

# Set by configure(); the pipeline below reads these.
DATASET: DatasetSpec = DATASETS["hcp"]
TASK: TaskSpec = TASKS["shell-holdout"]
LOCKED_DATA_PATH_TEMPLATES: dict = {}
LOCKED_DATA_KEYS: dict = {}
HELD_OUT_SHELLS: tuple = ()
TRAIN_SHELLS: tuple = ()
LOCKED_EPOCHS: int = 6
BENCHMARK_SUBJECT: str = "107422"
BENCHMARK_RUNS_ROOT: str = "OUTPUTS"
BENCHMARK_BASE_CONFIG = "configs/ball_fodf_hcp.yml"
SUPERRES_FACTOR: int | None = None

ARCH_KEYS = (
    "network_name", "hidden_size", "num_layers", "in_size", "out_size",
    "spatial_freq_num", "model_output_type", "shore_output", "shore_radial_order",
    "ball_stick_use_fodf", "ball_stick_fodf_l_max", "ball_stick_n_sticks",
    "ball_stick_use_spot", "fodf_export_axis_flip", "bval_train", "train_seed",
)


def configure(dataset: DatasetSpec, task: TaskSpec, *, directions=None, epochs=None,
              superres_factor=None):
    """Bind the pipeline to one (dataset, task) pair."""
    global DATASET, TASK, LOCKED_DATA_PATH_TEMPLATES, LOCKED_DATA_KEYS
    global HELD_OUT_SHELLS, TRAIN_SHELLS, LOCKED_EPOCHS, BENCHMARK_SUBJECT, BENCHMARK_RUNS_ROOT
    global SUPERRES_FACTOR
    SUPERRES_FACTOR = superres_factor
    DATASET, TASK = dataset, task
    root = os.environ.get(dataset.root_env, dataset.default_root)
    d = dataset.layout.format(root=root, subject="subject")
    LOCKED_DATA_PATH_TEMPLATES = {
        "data": f"{d}/data.nii.gz",
        "mask": f"{d}/nodif_brain_mask.nii.gz",
        "bvec": f"{d}/bvecs",
        "bval": f"{d}/bvals",
    }
    LOCKED_DATA_KEYS = {
        "shore_output": True, "sh_output": False, "use_b0_prior": False,
        "b0_threshold": 10, "shore_b0_normalize": False, "shore_radial_order": 8,
        "shore_zeta": 700, "bval_train": list(task.train_shells), "bval_val": "all",
        "bval_encoding": "linear", "normalize_0_1": True,
        "train_bvec_per_shell": (task.train_bvec_per_shell if directions is None
                                 else directions),
        "train_bvec_seed": task.train_bvec_seed,
    }
    HELD_OUT_SHELLS = tuple(task.held_out_shells)
    TRAIN_SHELLS = tuple(task.train_shells)
    LOCKED_EPOCHS = int(epochs) if epochs else task.epochs
    BENCHMARK_SUBJECT = dataset.default_subject
    BENCHMARK_RUNS_ROOT = f"OUTPUTS/{dataset.name}_{task.name}"


def _architecture_fingerprint(prepared_config_path) -> dict:
    """Architecture and export-frame keys, recorded with every run."""
    try:
        with open(prepared_config_path) as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as e:
        return {"_error": f"could not read prepared config: {e}"}
    out = {k: cfg.get(k) for k in ARCH_KEYS if k in cfg}
    out.setdefault("fodf_export_axis_flip", [-1.0, 1.0, 1.0])
    return out


def _compute_y_range(data: torch.Tensor, bvals: torch.Tensor, bval_train):
    if bval_train is not None and bval_train != "all":
        keep = torch.zeros(bvals.shape[0], dtype=torch.bool)
        for b in bval_train:
            keep |= (torch.round(bvals / 100) * 100 == b)
        data = data[..., keep]
    return data.min().item(), data.max().item()


def _build_eval_pool(config: dict, subject: str):
    cfg = _resolve_subject(config, subject)
    dwi, mask, bvec, bval = load_data(cfg)
    normalize_before_b0 = cfg.get("normalize_before_b0", False)
    use_b0_prior = cfg.get("use_b0_prior", False)

    if use_b0_prior:
        result = extract_b0_prior(
            dwi,
            bvec,
            bval,
            b0_threshold=cfg.get("b0_threshold", 10),
            mask=mask,
            normalize_first=normalize_before_b0,
        )
        if normalize_before_b0:
            b0_image, dwi_data, bvec, bval, _, _ = result
        else:
            b0_image, dwi_data, bvec, bval = result
    else:
        b0_mask = bval <= cfg.get("b0_threshold", 10)
        b0_image = dwi[..., b0_mask].float().mean(dim=-1)
        non_b0 = ~b0_mask
        dwi_data = dwi[..., non_b0]
        bvec = bvec[non_b0]
        bval = bval[non_b0]

    shore_b0_normalize = cfg.get("shore_b0_normalize", False)
    e_clip = cfg.get("shore_e_clip", 1.5)
    if shore_b0_normalize:
        b0_safe = b0_image.float().clamp(min=1.0)
        dwi_eval = (dwi_data.float() / b0_safe[..., None]).clamp(0, e_clip)
    else:
        dwi_eval = dwi_data

    bval_train = cfg.get("bval_train", None)
    y_min, y_max = _compute_y_range(dwi_eval[mask.bool()], bval, bval_train)
    cfg["source_y_min"] = y_min
    cfg["source_y_max"] = y_max

    pool = AcquisitionPool(dwi_eval, mask.bool(), bvec, bval, b0_image)
    return cfg, pool

def _architecture_fingerprint(prepared_config_path) -> dict:
    """Architecture + export-frame keys from the prepared config."""
    try:
        import yaml as _yaml
        with open(prepared_config_path) as fh:
            cfg = _yaml.safe_load(fh) or {}
    except Exception as e:
        return {"_error": f"could not read prepared config: {e}"}
    out = {k: cfg.get(k) for k in ARCH_KEYS if k in cfg}
    # Default applied by _dump_inr_fodf_nifti when the key is absent.
    out.setdefault("fodf_export_axis_flip", [-1.0, 1.0, 1.0])
    return out


def _apply_locked_constraints(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    for key, value in LOCKED_DATA_PATH_TEMPLATES.items():
        cfg[key] = value
    for key, value in LOCKED_DATA_KEYS.items():
        cfg[key] = copy.deepcopy(value)
    cfg["epochs"] = LOCKED_EPOCHS
    # Benchmark protocol: cold start (no pretrain), no output freezing.
    cfg["pretrained_weights_path"] = None
    cfg["output_lr_mult"] = 1.0
    # HCP brain is ~8x larger than ParkMR (~200K vs ~26K voxels) — lock a
    # smaller eval batch to avoid OOM on the shared GPU.
    if DATASET.eval_batch_size is not None:
        cfg["eval_batch_size"] = DATASET.eval_batch_size
        cfg["dti_eval_batch_size"] = DATASET.eval_batch_size
    return cfg


def _resolve_subject(config: dict, subject: str) -> dict:
    cfg = copy.deepcopy(config)
    cfg["subject"] = subject
    for key, value in cfg.items():
        if isinstance(value, str) and "subject" in value:
            cfg[key] = value.replace("subject", subject)
    return cfg


def _prepare_config(base_config: Path, out_config: Path, subject: str):
    with base_config.open("r") as f:
        raw_config = yaml.safe_load(f)
    config = _apply_locked_constraints(raw_config)
    checkpoint_dir = out_config.parent / "checkpoints" / subject
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config["checkpoint_dirpath"] = str(checkpoint_dir)
    config["checkpoint_filename"] = "last.ckpt"
    config["evaluation_checkpoint"] = "last"
    config["source_checkpoint"] = "last"
    config["checkpoint_save_top_k"] = 0
    config["checkpoint_every_n_train_steps"] = int(1e12)
    config["disable_checkpointing"] = True
    config["checkpoint_path"] = None
    config = _resolve_subject(config, subject)
    with out_config.open("w") as f:
        yaml.safe_dump(config, f)


def _load_eval_pool(cfg: dict, subject: str):
    """Reuse the locked HCP evaluator's pool builder. It computes
    `source_y_min` / `source_y_max` (used by `_predict_raw_all_volumes` to
    denormalize predictions back into the data's native scale before
    computing MSE) and applies the same B0 / bvec / bval handling. The
    builder is generic over `cfg['data']` etc., so ParkMR paths flow
    through unchanged."""
    return _build_eval_pool_hcp(cfg, subject)


def _shell_indices(pool: AcquisitionPool, shells: tuple[int, ...], tol: int = 50) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    bvals = pool.bval.numpy() if hasattr(pool.bval, "numpy") else np.asarray(pool.bval)
    for s in shells:
        idx = np.where(np.abs(bvals - s) <= tol)[0].tolist()
        out[int(s)] = sorted(idx)
    return out


def _save_slice_plots(pred_raw, gt_raw, pool, shells, run_dir: Path,
                      slice_axis: int = 2, max_vols_per_shell: int = 3):
    """Scatter held-out predictions/GT into 3D, save axial-slice grid.

    For each shell in `shells`, picks up to `max_vols_per_shell` example
    volumes and renders GT | Pred | |Pred-GT| side-by-side. Writes
    `slices_<label>.png` under `run_dir`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask_arr = pool.mask.cpu().numpy().astype(bool) if hasattr(pool.mask, "cpu") else np.asarray(pool.mask, dtype=bool)
    vol_shape = mask_arr.shape  # e.g. (X, Y, Z)
    sl = vol_shape[slice_axis] // 2
    shell_idx = _shell_indices(pool, tuple(shells))

    rows = []
    for s in shells:
        idxs = shell_idx.get(int(s), [])
        for v in idxs[:max_vols_per_shell]:
            rows.append((int(s), int(v)))
    if not rows:
        return

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows), squeeze=False)
    for r, (s, v) in enumerate(rows):
        pred_vol = np.zeros(vol_shape, dtype=np.float32)
        gt_vol = np.zeros(vol_shape, dtype=np.float32)
        pred_vol[mask_arr] = pred_raw[:, v]
        gt_vol[mask_arr] = gt_raw[:, v]

        if slice_axis == 0:
            pred2d, gt2d = pred_vol[sl], gt_vol[sl]
        elif slice_axis == 1:
            pred2d, gt2d = pred_vol[:, sl], gt_vol[:, sl]
        else:
            pred2d, gt2d = pred_vol[..., sl], gt_vol[..., sl]
        diff2d = np.abs(pred2d - gt2d)

        vmax = float(max(gt2d.max(), pred2d.max(), 1e-6))
        kw = dict(cmap="gray", vmin=0.0, vmax=vmax, origin="lower")
        axes[r, 0].imshow(gt2d.T, **kw); axes[r, 0].set_title(f"GT  b={s}  vol={v}")
        axes[r, 1].imshow(pred2d.T, **kw); axes[r, 1].set_title(f"Pred b={s}  vol={v}")
        axes[r, 2].imshow(diff2d.T, cmap="magma", vmin=0.0, vmax=vmax * 0.3, origin="lower")
        axes[r, 2].set_title(f"|Pred-GT|  (vmax={vmax * 0.3:.0f})")
        for c in range(3):
            axes[r, c].axis("off")

    fig.tight_layout()
    out = run_dir / "slices_held_out.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [eval] saved slice plots -> {out}")


def _predict_raw_all_volumes(model, pool: AcquisitionPool, cfg: dict):
    """Single forward pass on brain voxels, reconstruct signal at every volume
    in the eval pool, and denormalize predictions back to the raw DWI signal
    scale used by `pool.dwi_residual`.

    Returns:
        pred_raw [N_brain, n_volumes]: predicted signal in raw units.
        gt_raw   [N_brain, n_volumes]: ground-truth signal in raw units.
    """
    import torch
    import numpy as np_pr
    from dipy.core.gradients import gradient_table
    from dipy.reconst.shore import shore_matrix as dipy_shore_matrix

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    src_y_min = float(cfg.get("source_y_min", 0.0))
    src_y_max = float(cfg.get("source_y_max", 1.0))
    y_range = max(src_y_max - src_y_min, 1e-12)
    print(f"  [eval] denormalization: src_y_min={src_y_min:.4f}, src_y_max={src_y_max:.4f}, y_range={y_range:.4f}")

    mask = pool.mask.bool()
    dim = pool.dwi_residual.shape[:3]
    xyz_grid = torch.stack(
        torch.meshgrid(
            torch.arange(dim[0]),
            torch.arange(dim[1]),
            torch.arange(dim[2]),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3).float()
    mask_flat = mask.reshape(-1)
    brain_xyz = xyz_grid[mask_flat]
    dim_t = torch.tensor(dim, dtype=torch.float32)
    if cfg.get("normalize_0_1", True):
        brain_xyz_norm = 2 * (brain_xyz / dim_t) - 1
    else:
        brain_xyz_norm = brain_xyz / dim_t
    brain_xyz_norm = brain_xyz_norm.to(device)

    bvals_np = pool.bval.float().numpy()
    bvecs_np = pool.bvec.float().numpy()
    gtab = gradient_table(bvals_np, bvecs=bvecs_np)
    M = dipy_shore_matrix(
        radial_order=int(cfg.get("shore_radial_order", 6)),
        zeta=float(cfg.get("shore_zeta", 700)),
        gtab=gtab,
    )
    col_norms = np_pr.linalg.norm(M, axis=0, keepdims=True)
    col_norms = np_pr.maximum(col_norms, 1e-10)
    M_norm = M / col_norms
    shore_M = torch.from_numpy(M_norm).float().to(device)

    eval_batch_size = int(cfg.get("eval_batch_size", 500_000))
    n_brain = brain_xyz_norm.shape[0]
    use_spot_eval = bool(cfg.get("ball_stick_use_spot", False))
    spot_K_eval = int(cfg.get("ball_stick_spot_K", 8))
    if use_spot_eval:
        vs_norm_np = (2.0 / np.asarray(dim, dtype=np.float32))  # [3]
        vs_norm_t = torch.from_numpy(vs_norm_np).to(device)
        # Smaller voxel-batch for SPOT (each voxel produces K sub-samples).
        spot_voxel_batch = max(1, eval_batch_size // spot_K_eval)
    parts = []
    with torch.no_grad():
        if use_spot_eval:
            for start in range(0, n_brain, spot_voxel_batch):
                end = min(start + spot_voxel_batch, n_brain)
                coords_b = brain_xyz_norm[start:end]
                B = coords_b.shape[0]
                offsets = (torch.rand(B, spot_K_eval, 3, device=device) - 0.5) * vs_norm_t
                sub_coords = coords_b.unsqueeze(1) + offsets       # [B, K, 3]
                outs = model(sub_coords.view(B * spot_K_eval, 3))  # [B*K, n_raw]
                parts.append(outs)
        else:
            for start in range(0, n_brain, eval_batch_size):
                end = min(start + eval_batch_size, n_brain)
                parts.append(model(brain_xyz_norm[start:end]))
    shore_coeffs = torch.cat(parts, dim=0)  # [N_brain(*K_spot), n_raw]

    if cfg.get("model_output_type", "shore") == "ball_stick":
        # Model output is raw ball-stick parameters per voxel. Evaluate the
        # closed-form forward at all measured (b, q); multiply by per-voxel
        # S0 to recover raw signal. Bypasses the SHORE basis entirely.
        from inrdiff.ball_stick import (signal_at as _bs_signal_at,
                                signal_at_fodf as _bs_signal_at_fodf,
                                compute_sh_basis_at_bvecs as _bs_sh_basis,
                                make_legendre_quadrature as _bs_make_quad,
                                sh_coef_to_l_index as _bs_sh_l_idx)
        n_sticks = int(cfg.get("ball_stick_n_sticks", 2))
        use_fodf = bool(cfg.get("ball_stick_use_fodf", False))
        l_max = int(cfg.get("ball_stick_fodf_l_max", 6))

        bvals_t = torch.from_numpy(bvals_np).float().to(device)
        bvecs_t = torch.from_numpy(bvecs_np).float().to(device)
        bvec_norm = bvecs_t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        bvecs_t = bvecs_t / bvec_norm

        if use_fodf:
            Y_eval = _bs_sh_basis(bvecs_np, l_max)
            uniq, b_idx = np_pr.unique(np_pr.round(bvals_np), return_inverse=True)
            quad = _bs_make_quad(l_max)
            l_idx = _bs_sh_l_idx(l_max)
            Y_eval_t = torch.from_numpy(Y_eval).float().to(device)
            b_idx_t = torch.from_numpy(b_idx).long().to(device)
            uniq_t = torch.from_numpy(uniq.astype(np_pr.float32)).to(device)
            Pl_w_t = torch.from_numpy(quad["Pl_w"]).to(device)
            t_t = torch.from_numpy(quad["t"]).to(device)
            l_idx_t = torch.from_numpy(l_idx).long().to(device)
            quad_data = {"t": t_t, "Pl_w": Pl_w_t}

        with torch.no_grad():
            E_parts = []
            for s in range(0, shore_coeffs.shape[0], eval_batch_size):
                e = shore_coeffs[s:s + eval_batch_size]
                if use_fodf:
                    E_parts.append(_bs_signal_at_fodf(
                        e, bvals_t, bvecs_t, l_max,
                        Y_at_bvecs=Y_eval_t,
                        b_idx_per_vol=b_idx_t,
                        unique_bvals=uniq_t,
                        quad_data=quad_data,
                        l_idx_per_coef=l_idx_t,
                        unpin_c00=bool(cfg.get("ball_stick_fodf_unpin_c00", False)),
                    ).cpu())
                else:
                    E_parts.append(_bs_signal_at(e, bvals_t, bvecs_t, n_sticks).cpu())
            E_arr_t = torch.cat(E_parts, dim=0)            # [N(*K_spot), n_vols]
            if use_spot_eval:
                E_arr_t = E_arr_t.view(-1, spot_K_eval, E_arr_t.shape[-1]).mean(dim=1)
            E_arr = E_arr_t.numpy()
        b0_arr = pool.b0_image.float().cpu().numpy() if hasattr(pool.b0_image, "cpu") else np_pr.asarray(pool.b0_image, dtype=np_pr.float32)
        mask_arr = mask.cpu().numpy().astype(bool) if hasattr(mask, "cpu") else np_pr.asarray(mask, dtype=bool)
        S0_per_voxel = b0_arr[mask_arr].astype(np_pr.float32)
        pred_raw = E_arr * S0_per_voxel[:, None]
        print(f"  [eval] ball-stick mode: E range=[{E_arr.min():.4f}, {E_arr.max():.4f}], "
              f"n_sticks={n_sticks}, S0_per_voxel mean={float(S0_per_voxel.mean()):.1f}")
    else:
        pred_norm = (shore_coeffs @ shore_M.T).cpu().numpy()  # [N_brain, n_vols]
        pred_raw = pred_norm * y_range + src_y_min
        print(f"  [eval] pred_norm range: [{pred_norm.min():.4f}, {pred_norm.max():.4f}]")
    gt_raw = pool.dwi_residual[mask].float().numpy()  # [N_brain, n_vols]

    # Stage-2 SH residual on top of frozen stage-1 ball-fODF, if attached.
    inr2 = getattr(model, "_stage2_inr", None)
    if inr2 is not None and cfg.get("model_output_type") == "ball_stick":
        from inrdiff.stage2_residual import predict_with_stage2
        train_shell_bvals = list(
            cfg.get("stage2_train_shells", cfg.get("bval_train", []))
        ) or []
        stage2_l_max = int(cfg.get("stage2_l_max", 6))
        with torch.no_grad():
            if model.pos_encoder is not None:
                coords_enc = brain_xyz_norm
                for enc in model.pos_encoder:
                    coords_enc = enc(coords_enc)
            else:
                coords_enc = brain_xyz_norm
        residual_norm = predict_with_stage2(
            model_1=model, inr2=inr2,
            brain_coords_encoded=coords_enc,
            bvals_np=bvals_np, bvecs_np=bvecs_np,
            cfg=cfg, l_max=stage2_l_max,
            train_shell_bvals=train_shell_bvals,
            eval_batch_size=eval_batch_size,
            device=device,
            shore_M_eval=shore_M,
        )
        pred_raw = pred_raw + residual_norm * y_range
        n_train_meas = int((np_pr.abs(bvals_np - train_shell_bvals[0]) <= 50).sum())
        for s in train_shell_bvals[1:]:
            n_train_meas += int((np_pr.abs(bvals_np - s) <= 50).sum())
        print(f"  [eval] stage-2 residual range "
              f"[{(residual_norm * y_range).min():.2f}, "
              f"{(residual_norm * y_range).max():.2f}] "
              f"applied via gate, n_train_meas={n_train_meas}")

    print(f"  [eval] pred_raw range: [{pred_raw.min():.2f}, {pred_raw.max():.2f}]")
    print(f"  [eval] gt_raw range: [{gt_raw.min():.2f}, {gt_raw.max():.2f}], "
          f"mean(|gt|)={float(abs(gt_raw).mean()):.2f}")
    return pred_raw, gt_raw


def _canonical_metrics_for_volumes(pred_raw: np.ndarray, gt_raw: np.ndarray,
                                   vol_idxs: list, label,
                                   S0_typical: float | None = None) -> dict | None:
    """Canonical metric set in raw signal units. See program.md §6 for definitions.

    `signal_range` and `mean_signal` are GT-only; the model's raw RMSE/MAE are
    compared against them. PSNR_dB uses the shell's own signal_range so values
    are directly comparable across shells with very different signal magnitudes.

    If `S0_typical` is provided, also emits `psnr_db_S0_ref` — a fixed-reference
    PSNR using S0_typical (a physical anchor: the brain-mean unattenuated signal)
    as the denominator. This is shell- and dataset-agnostic and is what most
    diffusion-MRI reconstruction papers report. Companion to rel_rmse_pct.
    """
    if len(vol_idxs) == 0:
        return None
    p = pred_raw[:, vol_idxs]
    g = gt_raw[:, vol_idxs]
    diff = p - g
    mse = float((diff ** 2).mean())
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt(max(mse, 0.0)))
    mean_sig = float(np.abs(g).mean())
    sig_range = float(g.max() - g.min())
    rel_rmse_pct = 100.0 * rmse / max(mean_sig, 1e-12)
    rel_mae_pct = 100.0 * mae / max(mean_sig, 1e-12)
    psnr_db = 20.0 * math.log10(max(sig_range, 1e-12) / max(rmse, 1e-12))
    # Fixed-reference PSNR using S0 (brain-mean unattenuated signal) as the
    # physical anchor. Diffusion signal is bounded above by S0 by definition
    # so this is a meaningful "max possible signal" denominator. Comparable
    # across shells, datasets, and papers.
    if S0_typical is not None and S0_typical > 0:
        psnr_db_S0_ref = 20.0 * math.log10(float(S0_typical) / max(rmse, 1e-12))
    else:
        psnr_db_S0_ref = float("nan")
    # Per-volume PSNR: each GT volume's own max as the denominator. Mean over
    # volumes in this group. Matches eyes-on inspection of a single 2D slice.
    per_vol_psnrs = []
    for c in range(g.shape[1]):
        g_c = g[:, c]
        d_c = diff[:, c]
        rmse_c = float(np.sqrt((d_c ** 2).mean()))
        peak_c = float(g_c.max())
        if peak_c > 1e-12 and rmse_c > 1e-12:
            per_vol_psnrs.append(20.0 * math.log10(peak_c / rmse_c))
    psnr_db_per_vol = float(np.mean(per_vol_psnrs)) if per_vol_psnrs else float("nan")
    return {
        "shell": label,
        "n_volumes": int(len(vol_idxs)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mean_signal": mean_sig,
        "signal_range": sig_range,
        "rel_rmse_pct": rel_rmse_pct,
        "rel_mae_pct": rel_mae_pct,
        "psnr_db": psnr_db,
        "psnr_db_S0_ref": psnr_db_S0_ref,
        "psnr_db_per_vol": psnr_db_per_vol,
        "S0_typical": float(S0_typical) if S0_typical is not None else float("nan"),
    }


def _dump_inr_fodf_nifti(model, pool: AcquisitionPool, cfg: dict, run_dir: Path):
    """Forward INR at all brain voxels, reconstruct full ball-fODF SH coefs,
    convert descoteaux07 → tournier07 (MRtrix), pad to 45 (L=8) bands, and
    save as (X, Y, Z, 45) NIfTI. Returns path or None if not in fODF mode.

    The output matches the layout of <BINR_HCP_ROOT>/<subject>/Diffusion/
    gt_fod/wm_fod.nii.gz (MSMT-CSD GT) so evaluate_fod_acc_afd.py can ingest.
    """
    if not (cfg.get("model_output_type") == "ball_stick"
            and cfg.get("ball_stick_use_fodf", False)):
        print("  [fod_dump] skipped (not in ball-fODF mode)")
        return None

    import nibabel as nib
    from dipy.reconst.shm import convert_sh_descoteaux_tournier
    from inrdiff.ball_stick import reconstruct_full_fodf_lm, decode_params_fodf

    l_max = int(cfg.get("ball_stick_fodf_l_max", 2))
    unpin_c00 = bool(cfg.get("ball_stick_fodf_unpin_c00", False))
    n_sh_target = 45  # L=8 even SH → matches GT MSMT-CSD FOD layout

    device = next(model.parameters()).device
    mask_t = pool.mask.bool()
    dim_vox = tuple(int(d) for d in pool.dwi_residual.shape[:3])
    xyz = torch.stack(torch.meshgrid(
        torch.arange(dim_vox[0]),
        torch.arange(dim_vox[1]),
        torch.arange(dim_vox[2]),
        indexing="ij",
    ), dim=-1).reshape(-1, 3).float()
    mask_flat = mask_t.reshape(-1)
    brain_xyz = xyz[mask_flat]
    dim_t = torch.tensor(dim_vox, dtype=torch.float32)
    if cfg.get("normalize_0_1", True):
        coords = 2 * (brain_xyz / dim_t) - 1
    else:
        coords = brain_xyz / dim_t
    coords = coords.to(device)

    batch = int(cfg.get("dti_eval_batch_size", 50_000))
    parts = []
    with torch.no_grad():
        for s in range(0, coords.shape[0], batch):
            parts.append(model(coords[s:s + batch]))
    raw_out = torch.cat(parts, dim=0)

    fodf_lm = reconstruct_full_fodf_lm(raw_out, l_max, unpin_c00=unpin_c00)
    fodf_lm_np = fodf_lm.detach().cpu().numpy().astype(np.float64)

    # Convert dipy descoteaux07 ↔ MRtrix tournier07. The two bases differ
    # in the sign convention on the m>0 vs m<0 phase factors; for the
    # even L=2/4/... blocks we use, dipy provides a direct converter.
    fodf_tournier = convert_sh_descoteaux_tournier(fodf_lm_np)

    # Frame correction: the INR fODF is x-reflected relative to MRtrix's
    # MSMT-CSD GT (a bvec/axis convention mismatch). Validated end-to-end:
    # single-fiber peak error 48°→~8° and ACC_shape 0.18→0.82 once corrected.
    # Applied geometrically (sphere reflect + SH refit) so it is independent of
    # SH sign conventions. Disable/override via cfg['fodf_export_axis_flip'].
    axis_flip = cfg.get("fodf_export_axis_flip", [-1.0, 1.0, 1.0])
    if axis_flip and tuple(float(s) for s in axis_flip) != (1.0, 1.0, 1.0):
        import sys as _sys
        _sys.path.insert(0, str(REPO_ROOT / "src"))
        from inrdiff.fodf_frame import reflect_sh
        fodf_tournier = reflect_sh(fodf_tournier, l_max, "tournier07", axis_flip)
        print(f"  [fod_dump] frame correction applied: axis_flip={axis_flip}")

    n_our = fodf_tournier.shape[1]
    n_brain = fodf_tournier.shape[0]
    fodf_padded = np.zeros((n_brain, n_sh_target), dtype=np.float32)
    fodf_padded[:, :min(n_our, n_sh_target)] = fodf_tournier[:, :n_sh_target].astype(np.float32)

    mask_np = mask_t.cpu().numpy().astype(bool)
    vol = np.zeros((*dim_vox, n_sh_target), dtype=np.float32)
    vol[mask_np] = fodf_padded

    mask_path = cfg.get("mask")
    src_img = nib.load(mask_path)
    out_path = run_dir / "pred_wm_fod.nii.gz"
    nib.save(nib.Nifti1Image(vol, affine=src_img.affine, header=src_img.header),
             str(out_path))
    print(f"  [fod_dump] L={l_max}: saved fODF as 45-coef tournier07 NIfTI "
          f"({n_our} INR bands + {n_sh_target - n_our} zero-padded) → {out_path}")

    # Sidecar: per-voxel (1 − f_ball) — our "fiber-density" analog of AFD.
    # We pin SH_00 = √(4π) by construction (identifiability), so c_00 is
    # constant and useless as a density signal. The compartment fraction
    # (1 − f_ball) carries the fiber budget instead; see program.md §8 and
    # the AFD-parametrization note in project_hcp_holdout_frontier.md.
    p_fodf = decode_params_fodf(raw_out, l_max, unpin_c00=unpin_c00)
    fb = p_fodf["f_ball"].detach().cpu().numpy().astype(np.float32)  # [N_brain]
    d_iso = p_fodf["D_iso"].detach().cpu().numpy().astype(np.float32)
    d_par = p_fodf["D_par"].detach().cpu().numpy().astype(np.float32)
    one_minus_fball = (1.0 - fb)
    sidecar = run_dir / "pred_one_minus_fball.nii.gz"
    vol_fb = np.zeros(dim_vox, dtype=np.float32)
    vol_fb[mask_np] = one_minus_fball
    nib.save(nib.Nifti1Image(vol_fb, affine=src_img.affine, header=src_img.header),
             str(sidecar))
    print(f"  [fod_dump] (1−f_ball) range [{one_minus_fball.min():.3f}, "
          f"{one_minus_fball.max():.3f}], mean={one_minus_fball.mean():.3f} → {sidecar}")
    return out_path


def _evaluate_acc_afd(pred_fod_path: Path, subject: str, l_max: int,
                      run_dir: Path):
    """Run MSMT-CSD_INR/evaluate_fod_acc_afd.py and return ACC/AFD dict.

    Importing the evaluator directly into our process avoids a subprocess
    + duplicate env. Falls back gracefully if GT FOD missing or basis
    convention conflicts.
    """
    if pred_fod_path is None:
        return None
    try:
        from inrdiff import fod_acc_afd as mod

        import nibabel as nib
        diff, _, _, _, brain_p = mod.dataset_paths("hcp", subject)
        gt_path = diff / "gt_fod" / "wm_fod.nii.gz"
        if not gt_path.is_file():
            print(f"  [acc_afd] GT FOD missing at {gt_path}; skipped")
            return None

        gt = nib.load(str(gt_path)).get_fdata().astype(np.float64)
        pred = nib.load(str(pred_fod_path)).get_fdata().astype(np.float64)
        if gt.shape != pred.shape:
            print(f"  [acc_afd] shape mismatch gt={gt.shape} pred={pred.shape}; "
                  f"skipped")
            return None

        wm_mask = mod.get_wm_mask("hcp", subject)
        n_wm = int(wm_mask.sum())

        # ACC over l>=2 SH bands ONLY (drop the l=0 / density coef). The l=0
        # contribution dominates the SH-coef dot product because our pinned
        # c_00 = √(4π) ≈ 3.55 has a much larger magnitude than GT's ~0.20,
        # which gives a near-+1 cosine on that single band and dominates the
        # full ACC numerator + denominator. The remaining bands (l>=2) are
        # what tractography directionality actually depends on, so we report
        # those as the headline. See profile_acc.py for the per-band breakdown.
        eps = 1e-12

        def _shape_acc(p4d, g4d):
            ps = p4d[..., 1:]
            gs = g4d[..., 1:]
            num = (ps * gs).sum(axis=-1)
            den = np.sqrt(np.sum(ps * ps, axis=-1) * np.sum(gs * gs, axis=-1)) + eps
            return num / den

        acc_shape_vol = _shape_acc(pred, gt)
        afd_pred = mod.afd(pred)
        afd_gt = mod.afd(gt)
        acc_shape_wm = float(np.mean(acc_shape_vol[wm_mask]))
        acc_shape_wm_std = float(np.std(acc_shape_vol[wm_mask]))
        afd_mae_wm = float(np.mean(np.abs(afd_pred[wm_mask] - afd_gt[wm_mask])))
        afd_gt_mean_wm = float(np.mean(afd_gt[wm_mask]))
        afd_relmae_wm = afd_mae_wm / max(afd_gt_mean_wm, 1e-9)

        brain_mask = nib.load(str(brain_p)).get_fdata().astype(bool)
        acc_shape_brain = float(np.mean(acc_shape_vol[brain_mask]))
        afd_mae_brain = float(np.mean(np.abs(afd_pred[brain_mask] - afd_gt[brain_mask])))

        # Apples-to-apples AFD analog: our (1−f_ball) sidecar. The pinned
        # c_00 makes raw AFD relMAE meaningless (≈1700%) — compare (1−f_ball)
        # to GT AFD with Pearson correlation + per-subject linear regression
        # rescale (since the absolute constant differs by the WM response
        # function magnitude).
        fb_metrics = None
        fb_path = pred_fod_path.parent / "pred_one_minus_fball.nii.gz"
        if fb_path.is_file():
            fb_vol = nib.load(str(fb_path)).get_fdata().astype(np.float64)
            if fb_vol.shape == wm_mask.shape:
                x = fb_vol[wm_mask]
                y = afd_gt[wm_mask]
                if x.std() > 1e-9 and y.std() > 1e-9:
                    pearson = float(np.corrcoef(x, y)[0, 1])
                    # Per-subject linear rescale: minimize |a*x + b − y|² so
                    # the comparison is in matched units. Report relMAE on
                    # the rescaled (1−f_ball).
                    A = np.vstack([x, np.ones_like(x)]).T
                    a_lr, b_lr = np.linalg.lstsq(A, y, rcond=None)[0]
                    x_rescaled = a_lr * x + b_lr
                    fb_mae_wm = float(np.mean(np.abs(x_rescaled - y)))
                    fb_relmae_wm = fb_mae_wm / max(afd_gt_mean_wm, 1e-9)
                    fb_metrics = {
                        "pearson_wm": pearson,
                        "lin_a": float(a_lr),
                        "lin_b": float(b_lr),
                        "rescaled_mae_wm": fb_mae_wm,
                        "rescaled_relmae_wm": fb_relmae_wm,
                        "mean_one_minus_fball_wm": float(x.mean()),
                    }

        out = {
            "label": f"binr_inr_L{l_max}",
            "subject": subject,
            "l_max": l_max,
            "n_wm_voxels": n_wm,
            "acc_wm_shape": acc_shape_wm,
            "acc_wm_shape_std": acc_shape_wm_std,
            "afd_mae_wm": afd_mae_wm,
            "afd_relmae_wm": afd_relmae_wm,
            "afd_gt_mean_wm": afd_gt_mean_wm,
            "acc_brain_shape": acc_shape_brain,
            "afd_mae_brain": afd_mae_brain,
            "one_minus_fball_vs_gt_afd": fb_metrics,
        }
        print(f"  [acc_afd] L={l_max}: ACC_WM_shape(l>=2)={acc_shape_wm:+.4f} "
              f"AFD_relMAE_WM={100*afd_relmae_wm:.1f}% "
              f"(n_wm={n_wm}; AFD_gt_mean={afd_gt_mean_wm:.3f})")
        if fb_metrics:
            print(f"  [acc_afd] (1−f_ball) vs GT AFD: pearson={fb_metrics['pearson_wm']:+.3f} "
                  f"rescaled_relMAE={100*fb_metrics['rescaled_relmae_wm']:.1f}% "
                  f"(scale a={fb_metrics['lin_a']:.4f} offset b={fb_metrics['lin_b']:+.4f})")
        return out
    except Exception as e:
        print(f"  [acc_afd] failed: {e}")
        return None


def _evaluate_canonical(model, pool: AcquisitionPool, cfg: dict, run_dir: Path):
    """Compute canonical metrics for held-out + train shells in one pass.

    Returns (held_out_rows, train_shell_rows). Both are lists of dicts with one
    row per shell + an 'overall' row. Metric set: mse, rmse, mae, mean_signal,
    signal_range, rel_rmse_pct, rel_mae_pct, psnr_db — all in raw DWI units.
    """
    eval_root = run_dir / "evaluation"
    eval_root.mkdir(parents=True, exist_ok=True)

    pred_raw, gt_raw = _predict_raw_all_volumes(model, pool, cfg)

    try:
        _save_slice_plots(pred_raw, gt_raw, pool, HELD_OUT_SHELLS, run_dir)
    except Exception as e:
        print(f"  [eval] slice plot failed: {e}")

    # FOD dump + ACC/AFD vs MSMT-CSD GT. Only meaningful in ball-fODF mode.
    fod_metrics_out = None
    try:
        fod_path = _dump_inr_fodf_nifti(model, pool, cfg, run_dir)
        if fod_path is not None:
            fod_metrics_out = _evaluate_acc_afd(
                fod_path, cfg.get("subject", BENCHMARK_SUBJECT),
                int(cfg.get("ball_stick_fodf_l_max", 2)),
                run_dir,
            )
    except Exception as e:
        print(f"  [eval] FOD/ACC/AFD pipeline failed: {e}")

    # Tractography metric: DTI fit on GT (one-time reference) + native
    # extraction of e₁ from the model (no DTI fit on pred unless forced).
    # Forcing a DTI fit on pred is an opt-in via cfg["dti_eval_force_dti_fit_on_pred"].
    dti_metrics_out = None
    try:
        from inrdiff.dti_eval import (fit_gt_reference, angular_error_summary,
                              extract_e1_pred_dti_fit,
                              load_gt_reference, save_gt_reference)
        bvals_full = pool.bval.float().cpu().numpy()
        bvecs_full = pool.bvec.float().cpu().numpy()
        dti_shell_b = float(cfg.get("dti_eval_shell", 2000))
        idxs = np.where(np.abs(bvals_full - dti_shell_b) <= 50)[0]
        if idxs.size >= 6:
            sh_bvecs = bvecs_full[idxs]
            sh_gt = gt_raw[:, idxs]
            if hasattr(pool, "b0_image") and pool.b0_image is not None:
                b0_full = (pool.b0_image.float().cpu().numpy()
                           if hasattr(pool.b0_image, "cpu")
                           else np.asarray(pool.b0_image, dtype=np.float32))
                mask_arr2 = (pool.mask.cpu().numpy().astype(bool)
                             if hasattr(pool.mask, "cpu")
                             else np.asarray(pool.mask, dtype=bool))
                S0_per_voxel = b0_full[mask_arr2].astype(np.float32)
            else:
                S0_per_voxel = np.full(gt_raw.shape[0], 1000.0, dtype=np.float32)
            fa_thresh = float(cfg.get("dti_eval_fa_threshold", 0.2))
            cache_dir = (REPO_ROOT / BENCHMARK_RUNS_ROOT / "_dti_gt_cache").resolve()
            gt_ref = load_gt_reference(
                cache_dir, subject=cfg.get("subject", BENCHMARK_SUBJECT),
                b_value=dti_shell_b, fa_threshold=fa_thresh,
            )
            if gt_ref is None:
                gt_ref = fit_gt_reference(
                    sh_gt, dti_shell_b, sh_bvecs, S0_per_voxel,
                    fa_threshold=fa_thresh,
                )
                save_gt_reference(gt_ref, cache_dir, subject=cfg.get("subject", BENCHMARK_SUBJECT))
                print(f"  [eval] GT DTI fit cached -> {cache_dir}")
            else:
                print(f"  [eval] GT DTI loaded from cache: {gt_ref.get('cached_path')}")

            # Pred-side e1: prefer native model extraction; fall back to DTI
            # fit on predicted signal if requested or if the model doesn't
            # expose a native orientation.
            force_dti_fit = bool(cfg.get("dti_eval_force_dti_fit_on_pred", False))
            e1_pred = None
            fa_pred = None
            md_pred = None
            extraction = "unknown"
            if not force_dti_fit and cfg.get("model_output_type") == "ball_stick":
                try:
                    import torch as _t
                    from inrdiff.ball_stick import (reconstruct_full_fodf_lm,
                                            extract_e1_from_fodf_coefs,
                                            extract_e1_from_ball_stick_outputs)
                    # Re-forward model at voxel centers for native param read.
                    use_fodf = bool(cfg.get("ball_stick_use_fodf", False))
                    n_sticks = int(cfg.get("ball_stick_n_sticks", 1))
                    l_max = int(cfg.get("ball_stick_fodf_l_max", 2))
                    unpin_c00 = bool(cfg.get("ball_stick_fodf_unpin_c00", False))
                    device_t = next(model.parameters()).device
                    mask_t = pool.mask.bool()
                    dim_t_vox = pool.dwi_residual.shape[:3]
                    xyz_grid_v = _t.stack(_t.meshgrid(
                        _t.arange(dim_t_vox[0]),
                        _t.arange(dim_t_vox[1]),
                        _t.arange(dim_t_vox[2]),
                        indexing="ij",
                    ), dim=-1).reshape(-1, 3).float()
                    mask_flat = mask_t.reshape(-1)
                    brain_xyz_vox = xyz_grid_v[mask_flat]
                    dim_t_tensor = _t.tensor(dim_t_vox, dtype=_t.float32)
                    coords_norm = 2 * (brain_xyz_vox / dim_t_tensor) - 1
                    coords_norm = coords_norm.to(device_t)
                    e1_batch = int(cfg.get("dti_eval_batch_size", 50_000))
                    raw_parts = []
                    with _t.no_grad():
                        for s in range(0, coords_norm.shape[0], e1_batch):
                            raw_parts.append(model(coords_norm[s:s + e1_batch]))
                    raw_out = _t.cat(raw_parts, dim=0)
                    if use_fodf:
                        full_sh = reconstruct_full_fodf_lm(raw_out, l_max,
                                                           unpin_c00=unpin_c00)
                        e1_pred = extract_e1_from_fodf_coefs(full_sh, l_max)
                        extraction = f"native(ball-fODF L={l_max})"
                    else:
                        e1_pred = extract_e1_from_ball_stick_outputs(raw_out, n_sticks)
                        extraction = f"native(ball-stick N={n_sticks})"
                except Exception as e:
                    print(f"  [eval] native e1 extraction failed: {e}; "
                          f"falling back to DTI fit on pred")
                    e1_pred = None
            if e1_pred is None:
                e1_pred, fa_pred, md_pred = extract_e1_pred_dti_fit(
                    pred_raw[:, idxs], dti_shell_b, sh_bvecs, S0_per_voxel,
                )
                extraction = "dti_fit_on_pred"
            dti_metrics_out = angular_error_summary(
                e1_pred, gt_ref, fa_pred=fa_pred, md_pred=md_pred,
            )
            dti_metrics_out["n_dirs_used"] = int(idxs.size)
            dti_metrics_out["pred_e1_source"] = extraction
            print(f"  [eval] tract metric @ b={dti_shell_b:.0f} "
                  f"({extraction}; FA>{fa_thresh}, n_wm={dti_metrics_out['n_voxels_fa_passed']}): "
                  f"mean={dti_metrics_out['mean_ang_err_deg_wm']:.2f}° "
                  f"median={dti_metrics_out['median_ang_err_deg_wm']:.2f}° "
                  f"(all-voxel mean={dti_metrics_out['mean_ang_err_deg']:.2f}°)")
        else:
            print(f"  [eval] DTI metric skipped: only {idxs.size} dirs at b={dti_shell_b}")
    except Exception as e:
        print(f"  [eval] DTI metric failed: {e}")

    # S0_typical: brain-mean of the unattenuated signal. Used as the physical
    # reference for `psnr_db_S0_ref` (paper-style PSNR comparable across
    # shells/datasets). Falls back to None if b0_image isn't available.
    S0_typical = None
    if hasattr(pool, "b0_image") and pool.b0_image is not None:
        b0_arr = pool.b0_image.float().cpu().numpy() if hasattr(pool.b0_image, "cpu") else np.asarray(pool.b0_image, dtype=np.float32)
        mask_arr = pool.mask.cpu().numpy().astype(bool) if hasattr(pool.mask, "cpu") else np.asarray(pool.mask, dtype=bool)
        if b0_arr.shape == mask_arr.shape:
            S0_typical = float(b0_arr[mask_arr].mean())
            print(f"  S0_typical (brain-mean B0): {S0_typical:.2f}")

    def _build(shells: tuple, label: str):
        shell_idx = _shell_indices(pool, shells)
        present = sorted(s for s, v in shell_idx.items() if len(v) > 0)
        all_idxs = []
        for s in present:
            all_idxs.extend(shell_idx[s])
        print(f"  {label} shells {shells}: "
              f"{ {s: len(shell_idx[s]) for s in present} } "
              f"→ total {len(all_idxs)} volumes")
        rows = []
        for s in present:
            r = _canonical_metrics_for_volumes(pred_raw, gt_raw, shell_idx[s], int(s), S0_typical=S0_typical)
            if r is not None:
                rows.append(r)
        overall = _canonical_metrics_for_volumes(pred_raw, gt_raw, all_idxs, "overall", S0_typical=S0_typical)
        if overall is not None:
            rows.append(overall)
        out_dir = eval_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / "metrics_by_shell.csv", index=False)
        per_vol_rows = []
        for s in present:
            for v in shell_idx[s]:
                g_c = gt_raw[:, v]
                d_c = pred_raw[:, v] - g_c
                rmse_c = float(np.sqrt((d_c ** 2).mean()))
                peak_c = float(g_c.max())
                psnr_c = (20.0 * math.log10(peak_c / max(rmse_c, 1e-12))
                          if peak_c > 1e-12 else float("nan"))
                per_vol_rows.append({
                    "shell": int(s), "vol_idx": int(v),
                    "psnr_db_per_vol": psnr_c,
                    "rmse": rmse_c, "peak": peak_c,
                })
        pd.DataFrame(per_vol_rows).to_csv(out_dir / "psnr_per_volume.csv", index=False)
        return rows

    held_out_rows = _build(HELD_OUT_SHELLS, "held_out")
    train_shell_rows = _build(TRAIN_SHELLS, "train_shells")

    # Optional: bvec-holdout sub-benchmark. Requires `train_bvec_per_shell`
    # to be active so the training kept only a subset of bvecs per shell.
    # The complement (per training-shell) is the intra-shell direction
    # holdout — tests directional generalization within the trained shells.
    bvec_holdout_rows = None
    keep_idx_cfg = cfg.get("_train_bvec_keep_indices", None)
    if keep_idx_cfg is not None:
        keep_set = set(int(i) for i in keep_idx_cfg)
        # For each training shell, find the dropped vols (in eval-pool indices).
        train_shell_idx = _shell_indices(pool, TRAIN_SHELLS)
        bvec_held_vols_per_shell = {}
        all_bvec_held = []
        for s, idxs in train_shell_idx.items():
            held = sorted(i for i in idxs if int(i) not in keep_set)
            if held:
                bvec_held_vols_per_shell[s] = held
                all_bvec_held.extend(held)
        if all_bvec_held:
            rows = []
            present = sorted(bvec_held_vols_per_shell.keys())
            for s in present:
                r = _canonical_metrics_for_volumes(
                    pred_raw, gt_raw, bvec_held_vols_per_shell[s],
                    int(s), S0_typical=S0_typical,
                )
                if r is not None:
                    rows.append(r)
            overall = _canonical_metrics_for_volumes(
                pred_raw, gt_raw, all_bvec_held, "overall",
                S0_typical=S0_typical,
            )
            if overall is not None:
                rows.append(overall)
            out_dir = eval_root / "bvec_holdout"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "metrics_by_shell.csv", index=False)
            per_vol_rows = []
            for s in present:
                for v in bvec_held_vols_per_shell[s]:
                    g_c = gt_raw[:, v]
                    d_c = pred_raw[:, v] - g_c
                    rmse_c = float(np.sqrt((d_c ** 2).mean()))
                    peak_c = float(g_c.max())
                    psnr_c = (20.0 * math.log10(peak_c / max(rmse_c, 1e-12))
                              if peak_c > 1e-12 else float("nan"))
                    per_vol_rows.append({
                        "shell": int(s), "vol_idx": int(v),
                        "psnr_db_per_vol": psnr_c,
                        "rmse": rmse_c, "peak": peak_c,
                    })
            pd.DataFrame(per_vol_rows).to_csv(out_dir / "psnr_per_volume.csv", index=False)
            print(f"  bvec-holdout: scored {len(all_bvec_held)} intra-shell "
                  f"directions held out from training")
            bvec_holdout_rows = rows

            # Comprehensive held-out: merge b-value extrapolation +
            # intra-shell interpolation into a single overall metric.
            held_out_shell_idx = _shell_indices(pool, HELD_OUT_SHELLS)
            all_bvalue_held = []
            for s in HELD_OUT_SHELLS:
                all_bvalue_held.extend(held_out_shell_idx.get(int(s), []))
            combined_idxs = sorted(set(all_bvalue_held) | set(all_bvec_held))
            combined = _canonical_metrics_for_volumes(
                pred_raw, gt_raw, combined_idxs, "combined_overall",
                S0_typical=S0_typical,
            )
            if combined is not None:
                held_out_rows.append(combined)
                print(f"  comprehensive held-out: combined "
                      f"{len(all_bvalue_held)} b-value + {len(all_bvec_held)} "
                      f"intra-shell = {len(combined_idxs)} held-out vols")

    return (held_out_rows, train_shell_rows, bvec_holdout_rows,
            dti_metrics_out, fod_metrics_out)


def _serialize_summary(obj):
    if isinstance(obj, dict):
        return {k: _serialize_summary(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_summary(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj

def _decimate_mask(full_mask_path, factor: int, out_path):
    """Keep every factor-th voxel on each axis. Returns (train_mask, full_mask)."""
    img = nib.load(str(full_mask_path))
    full = np.asarray(img.dataobj) > 0
    keep = np.zeros_like(full)
    keep[::factor, ::factor, ::factor] = True
    train = full & keep
    nib.save(nib.Nifti1Image(train.astype(np.uint8), img.affine, img.header), str(out_path))
    return train, full


def _psnr_per_volume(pred, gt):
    """Per-volume PSNR against each volume's own data range."""
    out = []
    for j in range(gt.shape[1]):
        g, p = gt[:, j], pred[:, j]
        rmse = float(np.sqrt(np.mean((p - g) ** 2)))
        rng = float(g.max() - g.min())
        if rmse > 1e-12 and rng > 1e-12:
            out.append(20.0 * np.log10(rng / rmse))
    return (float(np.mean(out)), float(np.std(out)), len(out)) if out else (float("nan"),) * 3


def _run_superres(subject: str, base_config: Path, run_dir: Path) -> dict:
    """Train on a decimated voxel lattice, score the voxels never seen."""
    cfg = _resolve_subject(_apply_locked_constraints(yaml.safe_load(open(base_config))), subject)
    full_mask = Path(cfg["mask"])
    if not full_mask.exists():
        raise FileNotFoundError(f"mask not found: {full_mask}")
    train_mask_path = run_dir / f"train_mask_factor{SUPERRES_FACTOR}.nii.gz"
    train, full = _decimate_mask(full_mask, SUPERRES_FACTOR, train_mask_path)
    cfg["mask"] = str(train_mask_path)          # training sees only the subset
    cfg["epochs"] = LOCKED_EPOCHS
    cfg["torch_compile"] = False
    cfg["ball_stick_compile"] = False
    ck = run_dir / "checkpoints" / subject
    ck.mkdir(parents=True, exist_ok=True)
    cfg["checkpoint_dirpath"] = str(ck)
    prepared = run_dir / "config.prepared.yml"
    prepared.write_text(yaml.safe_dump(cfg, sort_keys=True))

    n_tr, n_full = int(train.sum()), int(full.sum())
    print(f"[1/2] superres factor={SUPERRES_FACTOR}: train on {n_tr} of {n_full} voxels "
          f"({n_tr / n_full:.1%}), hold out {n_full - n_tr}", flush=True)
    t0 = time.perf_counter()
    model, trained_cfg, _ = train_with_config(load_train_config(str(prepared), subject))
    train_seconds = time.perf_counter() - t0
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    print("[2/2] predicting every brain voxel, scoring the held-out ones", flush=True)
    eval_cfg = dict(trained_cfg)
    eval_cfg["mask"] = str(full_mask)
    eval_cfg, pool = _load_eval_pool(eval_cfg, subject)
    pred_raw, gt_raw = _predict_raw_all_volumes(model, pool, eval_cfg)
    pm = np.asarray(pool.mask.cpu() if hasattr(pool.mask, "cpu") else pool.mask) > 0
    res = {}
    for name, sel in (("heldout", (full & ~train)[pm]), ("train", train[pm])):
        if sel.sum() == 0:
            continue
        m, s, nv = _psnr_per_volume(pred_raw[sel], gt_raw[sel])
        res[name] = dict(psnr_mean=m, psnr_std=s, n_volumes=nv, n_voxels=int(sel.sum()))
        print(f"  {name:8s} n_vox={int(sel.sum()):7d}  PSNR {m:.3f} +/- {s:.3f}", flush=True)
    return {"factor": SUPERRES_FACTOR, "n_train_voxels": n_tr,
            "n_heldout_voxels": n_full - n_tr, "n_full_voxels": n_full,
            "timing_seconds": {"train": train_seconds},
            "architecture": _architecture_fingerprint(prepared), "metrics": res}


# the pipeline calls the pool builder under its historical name
_build_eval_pool_hcp = _build_eval_pool


def run(subject: str | None = None, config: str | None = None) -> dict:
    """Prepare, train and score one subject. Returns the summary dict."""
    subject = subject or BENCHMARK_SUBJECT
    base_config = (Path(config).resolve() if config
                   else (REPO_ROOT / BENCHMARK_BASE_CONFIG).resolve())
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = (REPO_ROOT / BENCHMARK_RUNS_ROOT /
               f"{DATASET.name}_{TASK.name}_{subject}_{stamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if TASK.name == "superres":
        summary = {"run_dir": str(run_dir), "subject": subject, "dataset": DATASET.name,
                   "task": TASK.name, "epochs": LOCKED_EPOCHS,
                   **_run_superres(subject, base_config, run_dir),
                   "ts": datetime.now(UTC).isoformat()}
        (run_dir / "metrics_summary.json").write_text(
            json.dumps(_serialize_summary(summary), indent=2))
        print(f"\nwrote {run_dir / 'metrics_summary.json'}", flush=True)
        return summary

    prepared = run_dir / "config.prepared.yml"
    _prepare_config(base_config, prepared, subject)

    print(f"[1/2] training on {subject}: shells {TRAIN_SHELLS}, "
          f"{LOCKED_DATA_KEYS['train_bvec_per_shell']} dirs/shell, "
          f"{LOCKED_EPOCHS} epochs\n      config: {prepared}", flush=True)
    t0 = time.perf_counter()
    model, trained_cfg, _ = train_with_config(load_train_config(str(prepared), subject))
    train_seconds = time.perf_counter() - t0
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    print(f"[2/2] scoring held-out shells {HELD_OUT_SHELLS or '(none)'} "
          f"and withheld directions", flush=True)
    eval_cfg, pool = _load_eval_pool(trained_cfg, subject)
    if "_train_bvec_keep_indices" in trained_cfg:
        eval_cfg["_train_bvec_keep_indices"] = trained_cfg["_train_bvec_keep_indices"]
    trained_cfg = eval_cfg
    t0 = time.perf_counter()
    (held_rows, train_rows, bvec_rows,
     dti_metrics, fod_metrics) = _evaluate_canonical(model, pool, trained_cfg, run_dir)
    eval_seconds = time.perf_counter() - t0

    summary = {
        "run_dir": str(run_dir), "subject": subject,
        "dataset": DATASET.name, "task": TASK.name,
        "train_shells": list(TRAIN_SHELLS), "held_out_shells": list(HELD_OUT_SHELLS),
        "epochs": LOCKED_EPOCHS,
        "architecture": _architecture_fingerprint(prepared),
        "timing_seconds": {"train": train_seconds, "evaluate": eval_seconds},
        "held_out": held_rows, "train_shells_metrics": train_rows,
        "bvec_holdout": bvec_rows,
        "dti_angular_error": dti_metrics, "fod_acc_afd": fod_metrics,
        "ts": datetime.now(UTC).isoformat(),
    }
    (run_dir / "metrics_summary.json").write_text(
        json.dumps(_serialize_summary(summary), indent=2))
    print(f"\nwrote {run_dir / 'metrics_summary.json'}", flush=True)
    return summary
