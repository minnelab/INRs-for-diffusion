"""Generate DWI predictions from checkpoint and compute MD/FA metrics."""

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import yaml
import argparse
import os
import gc
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from inrdiff.acquisition_adaptation import build_model, AcquisitionPool
from inrdiff.utils import load_data, extract_b0_prior
from inrdiff.dataset import encode_bval


def predict_shore_masked(model, mask, dim, bval, bvec, config, device="cuda",
                         b0_masked=None):
    """SHORE prediction returning masked-only predictions [N_brain, C].

    Args:
        b0_masked: If shore_b0_loss mode, raw B0 values for brain voxels [N_brain].
            Used to convert E-space predictions back to DWI.
    """
    model.eval()
    src_y_min = config.get("source_y_min", 0.0)
    src_y_max = config.get("source_y_max", 1.0)
    y_range = src_y_max - src_y_min
    shore_b0_loss = config.get("shore_b0_loss", False)
    eval_batch_size = config.get("eval_batch_size", 500000)

    # Brain xyz coords normalized
    xyz_grid = torch.stack(torch.meshgrid(
        torch.arange(dim[0]), torch.arange(dim[1]), torch.arange(dim[2]),
        indexing='ij'), dim=-1).reshape(-1, 3).float()
    mask_flat = mask.bool().reshape(-1)
    brain_xyz = xyz_grid[mask_flat]
    del xyz_grid
    n_brain = brain_xyz.shape[0]

    dim_t = torch.tensor(dim, dtype=torch.float32)
    brain_xyz_norm = 2 * (brain_xyz / dim_t) - 1
    del brain_xyz
    brain_xyz_norm = brain_xyz_norm.to(device)

    # Build SHORE matrix
    from dipy.core.gradients import gradient_table as gt_builder
    from dipy.reconst.shore import shore_matrix as dipy_shore_matrix

    bvals_np = bval.float().numpy()
    bvecs_np = bvec.float().numpy()
    gtab = gt_builder(bvals_np, bvecs=bvecs_np)
    M = dipy_shore_matrix(
        radial_order=config.get("shore_radial_order", 6),
        zeta=config.get("shore_zeta", 700),
        gtab=gtab,
    )
    col_norms = np.linalg.norm(M, axis=0, keepdims=True)
    col_norms = np.maximum(col_norms, 1e-10)
    M_normalized = M / col_norms
    shore_M = torch.from_numpy(M_normalized).float().to(device)

    # Forward pass: (x,y,z) -> SHORE coefficients, then reconstruct
    with torch.no_grad():
        shore_parts = []
        for start in range(0, n_brain, eval_batch_size):
            end = min(start + eval_batch_size, n_brain)
            p = model(brain_xyz_norm[start:end])
            shore_parts.append(p.cpu())
        shore_coeffs = torch.cat(shore_parts, dim=0)  # [N_brain, n_shore]
        del shore_parts

        # Reconstruct signals: [N_brain, n_volumes]
        shore_M_cpu = shore_M.cpu()
        pred_signal_norm = shore_coeffs @ shore_M_cpu.T
        del shore_coeffs, shore_M, shore_M_cpu

        if shore_b0_loss and b0_masked is not None:
            # Model outputs E-space: coeffs @ M.T = E (attenuation)
            # Convert to DWI: DWI = E_norm * B0_norm * y_range + y_min
            # where B0_norm = (B0 - y_min) / y_range
            b0_t = torch.from_numpy(b0_masked).float()
            b0_norm = (b0_t - src_y_min) / y_range
            pred_signal = pred_signal_norm * b0_norm.unsqueeze(-1) * y_range + src_y_min
        else:
            pred_signal = pred_signal_norm * y_range + src_y_min
        del pred_signal_norm

    return pred_signal.numpy()  # [N_brain, n_volumes]


def predict_all_volumes(model, pool, config, device="cuda"):
    """Predict all DWI residual volumes and return as [X,Y,Z,C] tensor."""
    model.eval()
    mask = pool.mask.bool()
    dim = pool.dwi_residual.shape[:3]
    max_bval = config.get("max_bval", 3000)
    bval_encoding = config.get("bval_encoding", "linear")

    # Brain xyz coords normalized
    xyz_grid = torch.stack(torch.meshgrid(
        torch.arange(dim[0]), torch.arange(dim[1]), torch.arange(dim[2]),
        indexing='ij'), dim=-1).reshape(-1, 3).float()
    mask_flat = mask.reshape(-1)
    brain_xyz = xyz_grid[mask_flat]
    n_brain = brain_xyz.shape[0]

    dim_t = torch.tensor(dim, dtype=torch.float32)
    brain_xyz_norm = 2 * (brain_xyz / dim_t) - 1
    brain_xyz_norm = brain_xyz_norm.to(device)

    # Normalization params
    src_y_min = config.get("source_y_min", 0.0)
    src_y_max = config.get("source_y_max", 1.0)
    y_range = src_y_max - src_y_min

    eval_batch_size = config.get("eval_batch_size", 500000)

    # Output: full volume predictions [X,Y,Z,C]
    pred_residual = torch.zeros(*dim, pool.n_volumes)

    sh_output = config.get("sh_output", False)

    with torch.no_grad():
        if sh_output:
            # SH output path: one forward pass per shell
            from e3nn import o3
            from inrdiff.networks.utils import xyz_to_angles
            sh_lmax = config.get("sh_lmax", 8)
            l_values = list(range(0, sh_lmax + 1, 2))

            shell_labels = torch.round(pool.bval / 1000) * 1000
            unique_shells = torch.unique(shell_labels).sort()[0]
            unique_shells = unique_shells[unique_shells > 50]

            for shell_bval in unique_shells:
                shell_mask = (shell_labels == shell_bval)
                shell_indices = torch.where(shell_mask)[0].tolist()

                # Build coords: (x,y,z, encoded_bval)
                actual_bval = pool.bval[shell_indices[0]]
                bval_enc = encode_bval(actual_bval.unsqueeze(0), bval_encoding, max_bval)
                bval_col = bval_enc.expand(n_brain, 1).to(device)
                coords = torch.cat([brain_xyz_norm, bval_col], dim=-1)  # [N_brain, 4]

                # Forward pass in sub-batches
                sh_parts = []
                for start in range(0, n_brain, eval_batch_size):
                    end = min(start + eval_batch_size, n_brain)
                    p = model(coords[start:end])
                    sh_parts.append(p)
                sh_coeffs = torch.cat(sh_parts, dim=0)  # [N_brain, n_sh]

                # Compute Y_matrix for this shell's directions
                shell_bvecs = pool.bvec[shell_indices].float().to(device)
                alpha, beta = xyz_to_angles(shell_bvecs)
                Y = o3.spherical_harmonics_alpha_beta(l_values, alpha, beta).to(device)

                # Convert to signal: [N_brain, n_dirs]
                pred_signal_norm = (sh_coeffs @ Y.T).cpu()

                # Unnormalize
                pred_signal = pred_signal_norm * y_range + src_y_min

                for local_i, vol_idx in enumerate(shell_indices):
                    pred_residual[..., vol_idx][mask] = pred_signal[:, local_i]

        else:
            # Original scalar output path
            all_bparams = torch.cat([
                encode_bval(pool.bval, bval_encoding, max_bval).unsqueeze(-1),
                pool.bvec,
            ], dim=-1).to(device)

            vols_per_batch = max(1, eval_batch_size // n_brain)

            for batch_start in range(0, pool.n_volumes, vols_per_batch):
                batch_vols = list(range(batch_start, min(batch_start + vols_per_batch, pool.n_volumes)))
                n_vols = len(batch_vols)

                xyz_rep = brain_xyz_norm.unsqueeze(0).expand(n_vols, -1, -1).reshape(-1, 3)
                bp = all_bparams[batch_vols]
                bp_rep = bp.unsqueeze(1).expand(-1, n_brain, -1).reshape(-1, 4)
                coords = torch.cat([xyz_rep, bp_rep], dim=-1)

                preds = []
                for start in range(0, coords.shape[0], eval_batch_size):
                    end = min(start + eval_batch_size, coords.shape[0])
                    p = model(coords[start:end])
                    preds.append(p)
                pred_norm = torch.cat(preds, dim=0).squeeze(-1).cpu()

                # Unnormalize: pred is in [0,1] space, convert back to residual space
                pred_residual_flat = pred_norm * y_range + src_y_min
                pred_per_vol = pred_residual_flat.reshape(n_vols, n_brain)

                for i, vol_idx in enumerate(batch_vols):
                    pred_residual[..., vol_idx][mask] = pred_per_vol[i]

    return pred_residual


def fit_dti(dwi_data, bvals, bvecs, mask):
    """Fit DTI model and return MD and FA maps."""
    gtab = gradient_table(bvals, bvecs=bvecs)
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(dwi_data, mask=mask)
    return tenfit.md, tenfit.fa


def compute_metrics(pred, gt, mask):
    """Compute PSNR, SSIM, MAE within mask."""
    pred_m = pred[mask]
    gt_m = gt[mask]
    data_range = gt_m.max() - gt_m.min()
    if data_range < 1e-10:
        return {'psnr': 0, 'ssim': 0, 'mae': 0}
    return {
        'psnr': psnr(gt_m, pred_m, data_range=data_range),
        'ssim': ssim(gt_m, pred_m, data_range=data_range),
        'mae': float(np.mean(np.abs(pred_m - gt_m))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Acquisition-style YAML config with source_checkpoint")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--fa_thresh", type=float, default=0.25)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    subject = args.subject
    config["subject"] = subject
    for key in list(config.keys()):
        if isinstance(config[key], str) and "subject" in config[key]:
            config[key] = config[key].replace("subject", subject)

    device = "cuda"

    # Load data
    print(f"Loading subject {subject}...")
    dwi, mask, bvec, bval = load_data(config)
    normalize_before_b0 = config.get("normalize_before_b0", False)
    use_b0_prior = config.get("use_b0_prior", False)

    if use_b0_prior:
        result = extract_b0_prior(
            dwi, bvec, bval,
            b0_threshold=config.get("b0_threshold", 10),
            mask=mask,
            normalize_first=normalize_before_b0,
        )
        if normalize_before_b0:
            b0_image, dwi_data, bvec_nb, bval_nb, dwi_min, dwi_max = result
        else:
            b0_image, dwi_data, bvec_nb, bval_nb = result
    else:
        # No B0 prior: remove B0 volumes but don't subtract
        b0_mask = bval <= config.get("b0_threshold", 10)
        b0_image = dwi[..., b0_mask].mean(dim=-1)
        non_b0 = ~b0_mask
        dwi_data = dwi[..., non_b0]
        bvec_nb = bvec[non_b0]
        bval_nb = bval[non_b0]
    print(f"  {dwi_data.shape[-1]} non-B0 volumes, mask: {mask.bool().sum().item()} voxels")
    print(f"  B0 mean signal: {b0_image[mask.bool()].mean():.1f}")
    del dwi  # Free original full DWI tensor

    # Source normalization params — must match training dataset's y_min/y_max
    def _compute_y_range(residual, bval_vals, bval_train=None):
        if bval_train is not None and bval_train != 'all':
            keep = torch.zeros(bval_vals.shape[0], dtype=torch.bool)
            for b in bval_train:
                keep |= (torch.round(bval_vals / 100) * 100 == b)
            residual = residual[..., keep]
        return residual.min().item(), residual.max().item()

    bval_train = config.get("bval_train", None)
    shore_b0_normalize = config.get("shore_b0_normalize", False)
    shore_output = config.get("shore_output", False)
    e_clip = config.get("shore_e_clip", 1.5)

    def _maybe_b0_normalize(data, b0_img):
        if shore_b0_normalize:
            b0_safe = b0_img.float().clamp(min=1.0)
            E = data.float() / b0_safe[..., None]
            return E.clamp(0, e_clip)
        return data

    source_subject = config.get("source_subject", subject)
    if shore_b0_normalize:
        # With skip_y_normalize, model outputs E-space directly — no y denorm needed
        config["source_y_min"] = 0.0
        config["source_y_max"] = 1.0
    elif source_subject == subject:
        data_for_range = _maybe_b0_normalize(dwi_data, b0_image)
        y_min, y_max = _compute_y_range(data_for_range[mask.bool()], bval_nb, bval_train)
        del data_for_range
        config["source_y_min"] = y_min
        config["source_y_max"] = y_max
    else:
        source_cfg = config.copy()
        for key in list(source_cfg.keys()):
            if isinstance(source_cfg[key], str) and subject in source_cfg[key]:
                source_cfg[key] = source_cfg[key].replace(subject, source_subject)
        src_dwi, src_mask, src_bvec, src_bval = load_data(source_cfg)
        if use_b0_prior:
            src_result = extract_b0_prior(
                src_dwi, src_bvec, src_bval,
                b0_threshold=config.get("b0_threshold", 10),
                mask=src_mask,
                normalize_first=normalize_before_b0,
            )
            src_data = src_result[1]
            src_bval_nb = src_result[3]
            src_b0 = src_result[0] if shore_b0_normalize else None
        else:
            src_b0_mask = src_bval <= config.get("b0_threshold", 10)
            src_b0 = src_dwi[..., src_b0_mask].float().mean(dim=-1) if shore_b0_normalize else None
            src_data = src_dwi[..., ~src_b0_mask]
            src_bval_nb = src_bval[~src_b0_mask]
        data_for_range = _maybe_b0_normalize(src_data, src_b0) if shore_b0_normalize else src_data
        y_min, y_max = _compute_y_range(data_for_range[src_mask.bool()], src_bval_nb, bval_train)
        config["source_y_min"] = y_min
        config["source_y_max"] = y_max
        del src_dwi, src_data
    print(f"  y_range (bval_train={bval_train}): [{config['source_y_min']:.4f}, {config['source_y_max']:.4f}]")

    # Load model
    print("Loading model...")
    model = build_model(config, device=device)

    dim = tuple(dwi_data.shape[:3])
    n_volumes = dwi_data.shape[-1]
    mask_bool = mask.bool()

    shore_b0_loss = config.get("shore_b0_loss", False)
    if shore_output:
        # Memory-efficient SHORE path: no pool needed
        print("Predicting (SHORE masked)...")
        b0_masked = b0_image[mask_bool].float().numpy() if shore_b0_loss else None
        pred_masked = predict_shore_masked(model, mask, dim, bval_nb, bvec_nb, config,
                                           device=device, b0_masked=b0_masked)
        # pred_masked: [N_brain, n_volumes] numpy array

        del model
        torch.cuda.empty_cache()
        gc.collect()

        # Build full volumes for DTI fitting
        mask_np = mask.numpy().astype(bool)
        b0_np = b0_image.numpy().astype(np.float64)

        if shore_b0_normalize:
            # pred is in E-space, GT is raw DWI
            gt_full = np.zeros((*dim, n_volumes), dtype=np.float64)
            gt_full[mask_np] = dwi_data.numpy()[mask_np].astype(np.float64)
            del dwi_data
            gc.collect()

            pred_full = np.zeros((*dim, n_volumes), dtype=np.float64)
            pred_full[mask_np] = pred_masked * b0_np[mask_np, None]
            del pred_masked
            print(f"  SHORE B0 denormalization: DWI = E * S0")
        else:
            gt_full = np.zeros((*dim, n_volumes), dtype=np.float64)
            gt_full[mask_np] = dwi_data.numpy()[mask_np].astype(np.float64)
            del dwi_data
            gc.collect()

            pred_full = np.zeros((*dim, n_volumes), dtype=np.float64)
            pred_full[mask_np] = pred_masked
            del pred_masked
    else:
        # Non-SHORE: use pool-based prediction
        if shore_b0_normalize:
            b0_safe = b0_image.float().clamp(min=1.0)
            dwi_for_pool = (dwi_data.float() / b0_safe[..., None]).clamp(0, e_clip)
        else:
            dwi_for_pool = dwi_data
        pool = AcquisitionPool(dwi_for_pool, mask, bvec_nb, bval_nb, b0_image)
        del dwi_for_pool

        print("Predicting all volumes...")
        pred_output = predict_all_volumes(model, pool, config, device=device)

        del model, pool
        torch.cuda.empty_cache()
        gc.collect()

        pred_np = pred_output.numpy()
        del pred_output
        mask_np = mask.numpy().astype(bool)
        b0_np = b0_image.numpy().astype(np.float64)

        if use_b0_prior:
            gt_full = dwi_data.numpy() + b0_np[..., None]
            pred_full = pred_np + b0_np[..., None]
            if normalize_before_b0:
                dwi_range = (dwi_max - dwi_min).item()
                dwi_min_val = dwi_min.item()
                gt_full = gt_full * dwi_range + dwi_min_val
                pred_full = pred_full * dwi_range + dwi_min_val
                b0_np = b0_np * dwi_range + dwi_min_val
                print(f"  Denormalized to raw space: dwi_min={dwi_min_val:.1f}, dwi_max={dwi_max.item():.1f}")
        elif shore_b0_normalize:
            gt_full = dwi_data.numpy().astype(np.float64)
            pred_full = pred_np * b0_np[..., None]
            print(f"  SHORE B0 denormalization: DWI = E * S0")
        else:
            gt_full = dwi_data.numpy()
            pred_full = pred_np

    # Clamp negative values
    pred_full = np.clip(pred_full, 0, None)

    # Prepend B0 volume for DTI fitting
    b0_vol = b0_np[..., None]
    gt_with_b0 = np.concatenate([b0_vol, gt_full], axis=-1)
    del gt_full
    pred_with_b0 = np.concatenate([b0_vol, pred_full], axis=-1)
    del pred_full
    gc.collect()

    bvals_with_b0 = np.concatenate([[0], bval_nb.numpy()])
    bvecs_with_b0 = np.concatenate([[[0, 0, 0]], bvec_nb.numpy()], axis=0)

    print(f"  Full DWI: {gt_with_b0.shape[-1]} volumes (1 B0 + {n_volumes} DWI)")
    print(f"  b-values: {np.unique(np.round(bvals_with_b0/1000)*1000)}")

    # Fit DTI
    print("Fitting DTI on GT...")
    gt_md, gt_fa = fit_dti(gt_with_b0, bvals_with_b0, bvecs_with_b0, mask_np)
    del gt_with_b0
    gc.collect()
    print("Fitting DTI on predictions...")
    pred_md, pred_fa = fit_dti(pred_with_b0, bvals_with_b0, bvecs_with_b0, mask_np)
    del pred_with_b0
    gc.collect()

    # Metrics
    md_metrics = compute_metrics(pred_md, gt_md, mask_np)
    fa_metrics = compute_metrics(pred_fa, gt_fa, mask_np)

    print(f"\n{'='*60}")
    print(f"MD/FA Evaluation (whole-brain, subject {subject})")
    print(f"{'='*60}")
    print(f"{'Metric':<6} {'PSNR':>8} {'SSIM':>8} {'MAE':>10}")
    print(f"{'-'*60}")
    print(f"{'MD':<6} {md_metrics['psnr']:>8.2f} {md_metrics['ssim']:>8.4f} {md_metrics['mae']:>10.6f}")
    print(f"{'FA':<6} {fa_metrics['psnr']:>8.2f} {fa_metrics['ssim']:>8.4f} {fa_metrics['mae']:>10.6f}")

    # WM FA
    wm_mask = (gt_fa > args.fa_thresh) & mask_np
    if wm_mask.sum() > 0:
        fa_wm = compute_metrics(pred_fa, gt_fa, wm_mask)
        print(f"\n{'='*60}")
        print(f"FA Evaluation (WM only, FA > {args.fa_thresh}, {wm_mask.sum()} voxels)")
        print(f"{'='*60}")
        print(f"{'FA-WM':<6} {fa_wm['psnr']:>8.2f} {fa_wm['ssim']:>8.4f} {fa_wm['mae']:>10.6f}")

    # Diagnostics
    print(f"\n{'='*60}")
    print(f"Diagnostics")
    print(f"{'='*60}")
    print(f"GT MD  - range: [{gt_md[mask_np].min():.6f}, {gt_md[mask_np].max():.6f}], mean: {gt_md[mask_np].mean():.6f}")
    print(f"Pred MD- range: [{pred_md[mask_np].min():.6f}, {pred_md[mask_np].max():.6f}], mean: {pred_md[mask_np].mean():.6f}")
    print(f"GT FA  - range: [{gt_fa[mask_np].min():.4f}, {gt_fa[mask_np].max():.4f}], mean: {gt_fa[mask_np].mean():.4f}")
    print(f"Pred FA- range: [{pred_fa[mask_np].min():.4f}, {pred_fa[mask_np].max():.4f}], mean: {pred_fa[mask_np].mean():.4f}")


if __name__ == "__main__":
    main()
