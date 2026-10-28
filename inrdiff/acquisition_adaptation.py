"""
Acquisition-like adaptation pipeline for DWI INR models.

Simulates sequential DWI acquisition on a target subject, starting from
B0 + pre-trained weights, to evaluate acquisition strategies.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import argparse
import os
import sys
import copy
from tqdm import tqdm
from pathlib import Path
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from inrdiff.utils import load_data, extract_b0_prior
from inrdiff.dataset import DWIReprDataset, PreBatchedDWIReprDataset, encode_bval
from inrdiff.model import INRLightningModule
from inrdiff.evaluation import load_model_from_checkpoint
from inrdiff.networks.inrs import WIRENMLP, SIRENMLP, ReLUMLP, ResidualGatedMLP, INCODEMLP
from inrdiff.networks.leaky_relu_inr import LeakyReLUMLP
from inrdiff.networks.pos_encoders import BvecAngleEncoder, SphericalPosEncoder, SpatialBvalEncoder, SpatialOnlyEncoder

model_mapping = {
    "leaky_relu_mlp": LeakyReLUMLP,
    "wiren_mlp": WIRENMLP,
    "siren_mlp": SIRENMLP,
    "relu_mlp": ReLUMLP,
    "residual_gated_mlp": ResidualGatedMLP,
    "incode_mlp": INCODEMLP,
}


# ---------------------------------------------------------------------------
# AcquisitionPool
# ---------------------------------------------------------------------------
class AcquisitionPool:
    """Manages which of the non-B0 volumes are 'acquired' vs available."""

    def __init__(self, dwi_residual, mask, bvec, bval, b0_image):
        """
        Args:
            dwi_residual: [X, Y, Z, C] residual volumes (DWI - B0)
            mask: [X, Y, Z] brain mask
            bvec: [C, 3] gradient directions
            bval: [C] b-values
            b0_image: [X, Y, Z] mean B0 image
        """
        self.dwi_residual = dwi_residual
        self.mask = mask
        self.bvec = bvec
        self.bval = bval
        self.b0_image = b0_image
        self.n_volumes = dwi_residual.shape[-1]

        self.acquired_indices = []
        self.available_indices = list(range(self.n_volumes))

        # Precompute shell membership
        self.shells = {}
        for i in range(self.n_volumes):
            shell = round(bval[i].item() / 1000) * 1000
            self.shells.setdefault(shell, []).append(i)

    def acquire(self, indices):
        """Move volume indices from available to acquired."""
        for idx in indices:
            if idx in self.available_indices:
                self.available_indices.remove(idx)
                self.acquired_indices.append(idx)

    def get_acquired_data(self):
        """Return (dwi, bvec, bval) subset for acquired volumes."""
        idx = sorted(self.acquired_indices)
        idx_t = torch.tensor(idx, dtype=torch.long)
        dwi = self.dwi_residual[..., idx_t]
        bvec = self.bvec[idx_t]
        bval = self.bval[idx_t]
        return dwi, bvec, bval

    def get_acquired_shells(self):
        """Return dict mapping shell -> list of acquired indices in that shell."""
        result = {}
        for idx in self.acquired_indices:
            shell = round(self.bval[idx].item() / 1000) * 1000
            result.setdefault(shell, []).append(idx)
        return result


# ---------------------------------------------------------------------------
# Selection Strategies
# ---------------------------------------------------------------------------
class RandomSelection:
    """Uniform random selection from remaining volumes."""

    def select(self, pool, model=None, dataset=None, config=None):
        n = config.get("volumes_per_round", 1)
        chosen = np.random.choice(pool.available_indices, size=min(n, len(pool.available_indices)), replace=False)
        return chosen.tolist()


class UniformSpreadSelection:
    """Round-robin across shells; within each shell pick most angularly distant."""

    def __init__(self):
        self._shell_cycle = None

    def select(self, pool, model=None, dataset=None, config=None):
        n = config.get("volumes_per_round", 1)
        selected = []
        shells_sorted = sorted(pool.shells.keys())
        if self._shell_cycle is None:
            self._shell_cycle = 0

        for _ in range(n):
            if not pool.available_indices:
                break
            # Round-robin shell
            for attempt in range(len(shells_sorted)):
                shell = shells_sorted[self._shell_cycle % len(shells_sorted)]
                self._shell_cycle += 1
                candidates = [i for i in pool.shells[shell] if i in pool.available_indices and i not in selected]
                if candidates:
                    break
            else:
                # Fallback: pick from any available
                candidates = [i for i in pool.available_indices if i not in selected]

            if not candidates:
                break

            # Pick direction most distant from already-acquired in this shell
            acquired_in_shell = [i for i in pool.acquired_indices if round(pool.bval[i].item() / 1000) * 1000 == shell]
            acquired_in_shell += [i for i in selected if round(pool.bval[i].item() / 1000) * 1000 == shell]

            if not acquired_in_shell:
                selected.append(candidates[0])
            else:
                acquired_dirs = pool.bvec[torch.tensor(acquired_in_shell)]  # [K, 3]
                best_idx = None
                best_dist = -1
                for c in candidates:
                    c_dir = pool.bvec[c]  # [3]
                    # Antipodal-aware cosine distance
                    cos_sim = torch.abs(F.cosine_similarity(c_dir.unsqueeze(0), acquired_dirs, dim=1))
                    min_dist = (1 - cos_sim.max()).item()
                    if min_dist > best_dist:
                        best_dist = min_dist
                        best_idx = c
                selected.append(best_idx)

        return selected


class UncertaintySelection:
    """MC Dropout uncertainty-based selection."""

    def select(self, pool, model=None, dataset=None, config=None):
        n = config.get("volumes_per_round", 1)
        mc_samples = config.get("mc_samples", 10)
        subsample = config.get("uncertainty_subsample", 30)
        eval_batch_size = config.get("eval_batch_size", 500000)
        device = next(model.parameters()).device

        candidates = pool.available_indices.copy()
        if len(candidates) > subsample:
            candidates = list(np.random.choice(candidates, size=subsample, replace=False))

        # Get brain voxel coordinates
        mask_flat = pool.mask.bool().reshape(-1)
        xyz_grid = torch.stack(torch.meshgrid(
            torch.arange(pool.mask.shape[0]),
            torch.arange(pool.mask.shape[1]),
            torch.arange(pool.mask.shape[2]), indexing='ij'),
            dim=-1).reshape(-1, 3)
        brain_coords = xyz_grid[mask_flat]  # [N_brain, 3]
        # Subsample brain voxels for speed
        max_voxels = 10000
        if brain_coords.shape[0] > max_voxels:
            voxel_idx = np.random.choice(brain_coords.shape[0], size=max_voxels, replace=False)
            brain_coords = brain_coords[voxel_idx]

        model.train()  # Enable dropout
        variances = {}

        for c_idx in candidates:
            bval_c = pool.bval[c_idx].unsqueeze(0).unsqueeze(0)  # [1,1]
            bvec_c = pool.bvec[c_idx].unsqueeze(0)  # [1, 3]
            bparam = torch.cat([bval_c, bvec_c], dim=-1)  # [1, 4]
            bparam_exp = bparam.expand(brain_coords.shape[0], -1)  # [N, 4]
            coords = torch.cat([brain_coords.float(), bparam_exp], dim=-1)  # [N, 7]

            # Normalize coords like the dataset does
            if dataset is not None:
                dim_t = torch.tensor(dataset.dim[:3], dtype=torch.float32)
                coords[:, :3] = 2 * (coords[:, :3] / dim_t) - 1
                bval_enc = config.get("bval_encoding", "linear") if config else "linear"
                coords[:, 3] = encode_bval(coords[:, 3], bval_enc, dataset.max_bval)

            preds = []
            for _ in range(mc_samples):
                pred_parts = []
                for start in range(0, coords.shape[0], eval_batch_size):
                    end = min(start + eval_batch_size, coords.shape[0])
                    batch = coords[start:end].to(device)
                    with torch.no_grad():
                        p = model(batch)
                    pred_parts.append(p.cpu())
                preds.append(torch.cat(pred_parts, dim=0))

            preds = torch.stack(preds, dim=0)  # [mc_samples, N, 1]
            var = preds.var(dim=0).mean().item()
            variances[c_idx] = var

        model.eval()

        # Select top-n by variance
        sorted_candidates = sorted(variances.keys(), key=lambda k: variances[k], reverse=True)
        return sorted_candidates[:n]


STRATEGIES = {
    "random": RandomSelection,
    "uniform": UniformSpreadSelection,
    "uncertainty": UncertaintySelection,
}


# ---------------------------------------------------------------------------
# Model loading (lightweight, no PL Trainer needed)
# ---------------------------------------------------------------------------
def create_network(config):
    """Create network and position encoders from config."""
    network_name = config.get("network_name", "wiren_mlp")
    network_kwargs = {
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
    }
    if network_name == "wiren_mlp":
        if "wire_omega" in config:
            network_kwargs["wire_omega"] = config["wire_omega"]
        if "wire_sigma" in config:
            network_kwargs["wire_sigma"] = config["wire_sigma"]
        if "wire_input_skip" in config:
            network_kwargs["wire_input_skip"] = config["wire_input_skip"]
    elif network_name == "siren_mlp" and "siren_factor" in config:
        network_kwargs["siren_factor"] = config["siren_factor"]
    elif network_name == "incode_mlp":
        if "incode_first_omega_0" in config:
            network_kwargs["incode_first_omega_0"] = config["incode_first_omega_0"]
        if "incode_hidden_omega_0" in config:
            network_kwargs["incode_hidden_omega_0"] = config["incode_hidden_omega_0"]
        if "incode_latent_dim" in config:
            network_kwargs["incode_latent_dim"] = config["incode_latent_dim"]
        if "incode_prior_mode" in config:
            network_kwargs["incode_prior_mode"] = config["incode_prior_mode"]
        if "incode_context_dim" in config:
            network_kwargs["incode_context_dim"] = config["incode_context_dim"]
        if "incode_layerwise_modulation" in config:
            network_kwargs["incode_layerwise_modulation"] = config["incode_layerwise_modulation"]
        if "incode_harmonizer_hidden" in config:
            network_kwargs["incode_harmonizer_hidden"] = tuple(config["incode_harmonizer_hidden"])
        if "incode_harmonizer_bias" in config:
            network_kwargs["incode_harmonizer_bias"] = config["incode_harmonizer_bias"]
        if "incode_harmonizer_weight_std" in config:
            network_kwargs["incode_harmonizer_weight_std"] = config["incode_harmonizer_weight_std"]
        if "incode_activation_mode" in config:
            network_kwargs["incode_activation_mode"] = config["incode_activation_mode"]
        if "incode_enforce_positive" in config:
            network_kwargs["incode_enforce_positive"] = config["incode_enforce_positive"]

    inr = model_mapping[network_name](
        config["in_size"],
        config.get("out_size", 1),
        **network_kwargs,
    )
    if config.get("shore_output", False):
        pos_encoder = [
            SpatialOnlyEncoder(
                coord_size=3,
                freq_num=config.get("spatial_freq_num", 128),
                freq_scale=config.get("spatial_freq_scale", 1.0),
            ),
        ]
    elif config.get("sh_output", False):
        pos_encoder = [
            SpatialBvalEncoder(
                coord_size=4, freq_num=128, freq_scale=1.0,
                bval_freq_num=config.get("bval_freq_num", 16),
                bval_freq_scale=config.get("bval_freq_scale", 1.0),
            ),
        ]
    elif config.get("pos_encoder", True):
        pos_encoder = [
            BvecAngleEncoder(7, (4, 5, 6)),
            SphericalPosEncoder(
                l_max=8, coord_size=4, freq_num=128, freq_scale=1.0,
                bval_freq_num=config.get("bval_freq_num", None),
                bval_freq_scale=config.get("bval_freq_scale", None),
            ),
        ]
    else:
        pos_encoder = None
    return inr, pos_encoder


def load_checkpoint_state(config):
    """Load checkpoint and return state_dict + hyperparams."""
    ckpt_path = config["source_checkpoint"]
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    hparams = checkpoint.get("hyper_parameters", {})
    return state_dict, hparams


def build_model(config, device="cuda"):
    """Build INRLightningModule and load checkpoint weights.

    Extracts the network and pos_encoder directly from the checkpoint's
    stored hyperparameters so architecture always matches.
    """
    state_dict, hparams = load_checkpoint_state(config)

    # Extract network and pos_encoder from checkpoint hparams
    network = hparams.get("network", None)
    pos_encoder = hparams.get("pos_encoder", None)

    if network is None:
        # Fallback: build from config
        network, pos_encoder = create_network(config)

    max_bval = hparams.get("max_bval", config.get("max_bval", 3000))
    if isinstance(max_bval, torch.Tensor):
        max_bval = max_bval.item()
    # Store resolved max_bval back into config
    config["max_bval"] = max_bval

    sh_output = config.get("sh_output", False) or hparams.get("sh_output", False)
    shore_output = config.get("shore_output", False) or hparams.get("shore_output", False)

    pl_module = INRLightningModule(
        network=network,
        train_loader=None,
        val_loaders_dict=None,
        pos_encoder=pos_encoder,
        lr=config.get("finetune_lr", 0.0001),
        max_bval=max_bval,
        use_dropout=config.get("use_dropout", True),
        sh_output=sh_output,
        shore_output=shore_output,
    )
    pl_module.load_state_dict(state_dict, strict=False)
    pl_module.to(device)
    pl_module.eval()
    return pl_module


# ---------------------------------------------------------------------------
# Precomputed Training Data (built once, indexed per round)
# ---------------------------------------------------------------------------
class PrecomputedTrainingData:
    """Precompute all normalized coords and values for all volumes once.

    Each round, we just gather rows for the acquired volume indices.
    This avoids rebuilding DWIReprDataset every round.
    """

    def __init__(self, pool, config, device="cuda"):
        mask = pool.mask.bool()
        dim = pool.dwi_residual.shape[:3]
        n_brain = mask.sum().item()
        max_bval = config.get("max_bval", 3000)

        # Source normalization params
        src_y_min = config.get("source_y_min", 0.0)
        src_y_max = config.get("source_y_max", 1.0)
        self.y_min = src_y_min
        self.y_max = src_y_max

        # Build brain xyz coords, normalized
        xyz = torch.stack(torch.meshgrid(
            torch.arange(dim[0]),
            torch.arange(dim[1]),
            torch.arange(dim[2]), indexing='ij'),
            dim=-1).reshape(-1, 3).float()
        brain_xyz = xyz[mask.reshape(-1)]  # [N_brain, 3]
        dim_t = torch.tensor(dim, dtype=torch.float32)
        if config.get("normalize_0_1", True):
            brain_xyz_norm = 2 * (brain_xyz / dim_t) - 1
        else:
            brain_xyz_norm = brain_xyz / dim_t

        # For each volume: build normalized coords [N_brain, 7] and values [N_brain, 1]
        # Store per-volume on CPU, GPU-transfer happens at training time
        self.x_per_vol = []  # list of [N_brain, 7] tensors
        self.y_per_vol = []  # list of [N_brain, 1] tensors

        bval_encoding = config.get("bval_encoding", "linear")
        for v in range(pool.n_volumes):
            bparam = torch.cat([encode_bval(pool.bval[v:v+1], bval_encoding, max_bval), pool.bvec[v]])  # [4]
            bparam_exp = bparam.unsqueeze(0).expand(n_brain, -1)
            x = torch.cat([brain_xyz_norm, bparam_exp], dim=-1)  # [N_brain, 7]

            y_raw = pool.dwi_residual[..., v][mask].float()
            if config.get("normalize_0_1", True):
                y_norm = (y_raw - src_y_min) / (src_y_max - src_y_min)
            else:
                y_norm = y_raw

            self.x_per_vol.append(x.half())  # store as fp16 to save memory
            self.y_per_vol.append(y_norm.half())

        print(f"  Precomputed training data: {pool.n_volumes} volumes x {n_brain} brain voxels")

    def get_training_batches(self, acquired_indices, batch_size, device="cuda"):
        """Gather data for acquired volumes, shuffle, batch, move to GPU."""
        x_parts = [self.x_per_vol[i] for i in acquired_indices]
        y_parts = [self.y_per_vol[i] for i in acquired_indices]
        x_all = torch.cat(x_parts, dim=0).float()
        y_all = torch.cat(y_parts, dim=0).float().unsqueeze(-1)

        # Shuffle
        perm = torch.randperm(x_all.shape[0])
        x_all = x_all[perm]
        y_all = y_all[perm]

        # Batch and move to GPU
        N = x_all.shape[0]
        n_batches = max(1, N // batch_size)
        x_all = x_all[:n_batches * batch_size].view(n_batches, batch_size, -1).to(device)
        y_all = y_all[:n_batches * batch_size].view(n_batches, batch_size, -1).to(device)
        return x_all, y_all, n_batches


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------
def finetune(model, pool, config, source_state_dict, precomp_data,
             optimizer=None, device="cuda"):
    """Fine-tune model on acquired volumes.

    If reset_each_round is True (default), resets weights from checkpoint.
    Otherwise continues training from current weights.
    Returns optimizer for reuse.
    """
    reset = config.get("reset_each_round", False)
    if reset:
        model.load_state_dict(source_state_dict, strict=False)
        model.to(device)
        optimizer = None  # force new optimizer after reset

    model.train()

    B = config.get("finetune_batch_size", 65536)
    x_batches, y_batches, n_batches = precomp_data.get_training_batches(
        pool.acquired_indices, B, device=device
    )

    steps = config.get("finetune_steps", 100)
    lr = config.get("finetune_lr", 0.0001)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        bi = step % n_batches
        pred = model(x_batches[bi])
        loss = F.mse_loss(pred, y_batches[bi])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    del x_batches, y_batches
    torch.cuda.empty_cache()
    return optimizer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_all_volumes(model, pool, config, ref_dataset, device="cuda",
                         eval_vol_indices=None, precomp_data=None):
    """Evaluate model on volumes using batched multi-volume forward passes.

    Computes PSNR and SSIM in [0,1] normalized space with data_range=1.0,
    matching the original evaluation pipeline (evaluation.py / logging_utils.py).

    SSIM is computed on reconstructed 3D volumes (brain voxel predictions placed
    back into volume shape) using compute_image_metrics from logging_utils.py.

    Concatenates brain voxels from multiple volumes into large batches for
    high GPU utilization. Returns per-volume DataFrame with mse, psnr, ssim, b_value.
    """
    from inrdiff.logging_utils import compute_image_metrics

    model.eval()
    eval_batch_size = config.get("eval_batch_size", 500000)
    compute_ssim = bool(config.get("eval_compute_ssim", True))
    eval_max_voxels = config.get("eval_max_voxels", None)
    eval_voxel_seed = int(config.get("eval_voxel_seed", 42))

    mask = pool.mask.bool()
    dim = pool.dwi_residual.shape[:3]
    max_bval = config.get("max_bval", 3000)

    # Pre-compute brain-only xyz coords once
    xyz_grid = torch.stack(torch.meshgrid(
        torch.arange(dim[0]),
        torch.arange(dim[1]),
        torch.arange(dim[2]), indexing='ij'),
        dim=-1).reshape(-1, 3).float()
    mask_flat = mask.reshape(-1)
    brain_xyz = xyz_grid[mask_flat]  # [N_brain, 3]
    n_brain = brain_xyz.shape[0]
    voxel_indices = None
    if eval_max_voxels is not None and int(eval_max_voxels) > 0 and n_brain > int(eval_max_voxels):
        rng = np.random.default_rng(eval_voxel_seed)
        voxel_indices = np.sort(rng.choice(n_brain, size=int(eval_max_voxels), replace=False))
        brain_xyz = brain_xyz[voxel_indices]
        n_brain = brain_xyz.shape[0]
        # SSIM on sparse voxel subsets is not meaningful; use voxel-space MSE/PSNR only.
        compute_ssim = False

    # Pre-normalize spatial coords
    dim_t = torch.tensor(dim, dtype=torch.float32)
    if config.get("normalize_0_1", True):
        brain_xyz_norm = 2 * (brain_xyz / dim_t) - 1
    else:
        brain_xyz_norm = brain_xyz / dim_t
    brain_xyz_norm = brain_xyz_norm.to(device)  # Keep on GPU

    # Get normalization params for computing [0,1] GT
    src_y_min = config.get("source_y_min", 0.0)
    src_y_max = config.get("source_y_max", 1.0)
    y_range = src_y_max - src_y_min

    bval_encoding = config.get("bval_encoding", "linear")

    # Pre-compute all GT brain voxels in [0,1] normalized space [270, N_brain]
    gt_residuals = pool.dwi_residual[mask].T  # [270, N_brain]
    if config.get("normalize_0_1", True):
        gt_norm = (gt_residuals - src_y_min) / y_range  # [270, N_brain] in [0,1]
    else:
        gt_norm = gt_residuals
    if voxel_indices is not None:
        gt_norm = gt_norm[:, voxel_indices]

    # Pre-initialize metrics for reuse across volumes
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device) if compute_ssim else None
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device) if compute_ssim else None

    results = []
    if eval_vol_indices is not None:
        vol_indices = eval_vol_indices
    else:
        vol_indices = list(range(pool.n_volumes))
    vol_indices_set = set(vol_indices)

    sh_output = config.get("sh_output", False)
    shore_output = config.get("shore_output", False)

    def _compute_metrics(pred_01, gt_01):
        mse_01 = F.mse_loss(pred_01, gt_01).item()
        if compute_ssim:
            pred_vol = torch.zeros(dim, dtype=torch.float32)
            gt_vol = torch.zeros(dim, dtype=torch.float32)
            pred_vol[mask] = pred_01
            gt_vol[mask] = gt_01
            psnr_val, ssim_val, _ = compute_image_metrics(
                pred_vol, gt_vol, mask=mask,
                device=device, compute_lpips=False,
                psnr_metric=psnr_metric, ssim_metric=ssim_metric,
            )
        else:
            psnr_val = float(-10.0 * np.log10(max(mse_01, 1e-12)))
            ssim_val = float("nan")
        return mse_01, psnr_val, ssim_val

    if shore_output:
        # Coefficient-output path: single forward pass per voxel, reconstruct all
        # volumes via a fixed design matrix. The basis (SHORE or Bessel) is
        # selected by config['basis'].
        import numpy as np_acq

        basis_name = str(config.get("basis", "shore")).lower()
        bvals_np = pool.bval.float().numpy()
        bvecs_np = pool.bvec.float().numpy()

        if basis_name == "bessel":
            from inrdiff.bessel_basis import bessel_fourier_matrix

            M, _info, _q_max = bessel_fourier_matrix(
                bvals_np,
                bvecs_np,
                radial_order=int(config.get("bessel_radial_order", 2)),
                lmax=int(config.get("bessel_lmax", 8)),
                q_max=config.get("bessel_q_max", None),
            )
        elif basis_name == "shore":
            from dipy.core.gradients import gradient_table
            from dipy.reconst.shore import shore_matrix as dipy_shore_matrix

            gtab = gradient_table(bvals_np, bvecs=bvecs_np)
            M = dipy_shore_matrix(
                radial_order=config.get("shore_radial_order", 6),
                zeta=config.get("shore_zeta", 700),
                gtab=gtab,
            )
        else:
            raise ValueError(
                f"Unknown basis '{basis_name}' — expected 'shore' or 'bessel'"
            )

        col_norms = np_acq.linalg.norm(M, axis=0, keepdims=True)
        col_norms = np_acq.maximum(col_norms, 1e-10)
        M_normalized = M / col_norms
        shore_M = torch.from_numpy(M_normalized).float().to(device)

        # Forward pass for all brain voxels (just xyz, no bval)
        shore_parts = []
        for start in range(0, n_brain, eval_batch_size):
            end = min(start + eval_batch_size, n_brain)
            p = model(brain_xyz_norm[start:end])
            shore_parts.append(p)
        shore_coeffs = torch.cat(shore_parts, dim=0)  # [N_brain, n_shore]

        # Reconstruct all signals: [N_brain, n_volumes]
        model_output_type = str(config.get("model_output_type", "shore"))
        if model_output_type == "ball_stick":
            from inrdiff.ball_stick import (signal_at as _bs_signal_at,
                                    signal_at_fodf as _bs_signal_at_fodf,
                                    compute_sh_basis_at_bvecs as _bs_sh_basis,
                                    make_legendre_quadrature as _bs_make_quad,
                                    sh_coef_to_l_index as _bs_sh_l_idx)
            n_sticks = int(config.get("ball_stick_n_sticks", 2))
            use_fodf = bool(config.get("ball_stick_use_fodf", False))
            l_max = int(config.get("ball_stick_fodf_l_max", 6))

            bvals_t = torch.from_numpy(bvals_np).float().to(device)
            bvecs_t = torch.from_numpy(bvecs_np).float().to(device)
            bvec_norm_ = bvecs_t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            bvecs_t = bvecs_t / bvec_norm_

            if not hasattr(pool, "b0_image") or pool.b0_image is None:
                raise RuntimeError("ball-stick eval needs pool.b0_image for S0")
            b0_arr = (pool.b0_image.float().cpu().numpy()
                      if hasattr(pool.b0_image, "cpu")
                      else np_acq.asarray(pool.b0_image, dtype=np_acq.float32))
            mask_arr = (pool.mask.cpu().numpy().astype(bool)
                        if hasattr(pool.mask, "cpu")
                        else np_acq.asarray(pool.mask, dtype=bool))
            S0_full = b0_arr[mask_arr].astype(np_acq.float32)
            if voxel_indices is not None:
                S0_full = S0_full[voxel_indices]
            S0_t = torch.from_numpy(S0_full).to(device)

            if use_fodf:
                Y_eval = _bs_sh_basis(bvecs_np, l_max)
                uniq, b_idx = np_acq.unique(np_acq.round(bvals_np), return_inverse=True)
                quad = _bs_make_quad(l_max)
                l_idx = _bs_sh_l_idx(l_max)
                Y_t = torch.from_numpy(Y_eval).float().to(device)
                b_idx_t = torch.from_numpy(b_idx).long().to(device)
                uniq_t = torch.from_numpy(uniq.astype(np_acq.float32)).to(device)
                Pl_w_t = torch.from_numpy(quad["Pl_w"]).to(device)
                t_t = torch.from_numpy(quad["t"]).to(device)
                l_idx_t = torch.from_numpy(l_idx).long().to(device)
                quad_data = {"t": t_t, "Pl_w": Pl_w_t}

            with torch.no_grad():
                if use_fodf:
                    E_t = _bs_signal_at_fodf(
                        shore_coeffs, bvals_t, bvecs_t, l_max,
                        Y_at_bvecs=Y_t, b_idx_per_vol=b_idx_t,
                        unique_bvals=uniq_t, quad_data=quad_data,
                        l_idx_per_coef=l_idx_t,
                    )
                else:
                    E_t = _bs_signal_at(shore_coeffs, bvals_t, bvecs_t, n_sticks)
                pred_raw_t = E_t * S0_t[:, None]
            pred_all = ((pred_raw_t - src_y_min) / y_range).cpu()
        else:
            pred_all = (shore_coeffs @ shore_M.T).cpu()

        for vol_idx in vol_indices:
            pred_01 = pred_all[:, vol_idx]
            gt_01 = gt_norm[vol_idx]
            mse_01, psnr_val, ssim_val = _compute_metrics(pred_01, gt_01)

            results.append({
                "vol_idx": vol_idx,
                "b_value": round(pool.bval[vol_idx].item() / 1000) * 1000,
                "mse": mse_01,
                "psnr": psnr_val,
                "ssim": ssim_val,
            })

    elif sh_output:
        # SH output path: one forward pass per shell, reconstruct all directions
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

            # Skip shell if no eval volumes in it
            eval_in_shell = [vi for vi in shell_indices if vi in vol_indices_set]
            if not eval_in_shell:
                continue

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

            # Convert to signal: [N_brain, n_dirs_in_shell]
            pred_signal = sh_coeffs @ Y.T
            pred_signal = pred_signal.cpu()

            for local_i, vol_idx in enumerate(shell_indices):
                if vol_idx not in vol_indices_set:
                    continue

                pred_01 = pred_signal[:, local_i]
                gt_01 = gt_norm[vol_idx]
                mse_01, psnr_val, ssim_val = _compute_metrics(pred_01, gt_01)

                results.append({
                    "vol_idx": vol_idx,
                    "b_value": round(pool.bval[vol_idx].item() / 1000) * 1000,
                    "mse": mse_01,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                })

    else:
        # Original scalar output path
        all_bparams = torch.cat([
            encode_bval(pool.bval, bval_encoding, max_bval).unsqueeze(-1),
            pool.bvec,
        ], dim=-1).to(device)

        vols_per_batch = max(1, eval_batch_size // n_brain)

        for batch_start in range(0, len(vol_indices), vols_per_batch):
            batch_vols = vol_indices[batch_start:batch_start + vols_per_batch]
            n_vols = len(batch_vols)

            xyz_rep = brain_xyz_norm.unsqueeze(0).expand(n_vols, -1, -1).reshape(-1, 3)
            bp = all_bparams[batch_vols]
            bp_rep = bp.unsqueeze(1).expand(-1, n_brain, -1).reshape(-1, 4)
            coords = torch.cat([xyz_rep, bp_rep], dim=-1)

            total = coords.shape[0]
            preds = []
            for start in range(0, total, eval_batch_size):
                end = min(start + eval_batch_size, total)
                p = model(coords[start:end])
                preds.append(p)
            pred_norm = torch.cat(preds, dim=0).squeeze(-1)

            pred_per_vol = pred_norm.reshape(n_vols, n_brain).cpu()

            for i, vol_idx in enumerate(batch_vols):
                pred_01 = pred_per_vol[i]
                gt_01 = gt_norm[vol_idx]
                mse_01, psnr_val, ssim_val = _compute_metrics(pred_01, gt_01)

                results.append({
                    "vol_idx": vol_idx,
                    "b_value": round(pool.bval[vol_idx].item() / 1000) * 1000,
                    "mse": mse_01,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                })

    df = pd.DataFrame(results)
    return df


def aggregate_metrics(vol_metrics_df):
    """Aggregate per-volume metrics into overall and per-shell summaries."""
    metric_cols = [c for c in ["psnr", "ssim", "mse"] if c in vol_metrics_df.columns]
    overall = {c: vol_metrics_df[c].mean() for c in metric_cols}
    per_shell = vol_metrics_df.groupby("b_value").agg(
        {c: "mean" for c in metric_cols}
    ).reset_index()
    return overall, per_shell


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_acquisition_curves(results_df, output_path):
    """4-panel figure: overall PSNR, per-shell PSNR, overall SSIM, per-shell SSIM."""
    has_ssim = "ssim" in results_df.columns
    n_rows = 2 if has_ssim else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Panel (0,0): Overall PSNR
    axes[0, 0].plot(results_df["n_acquired"], results_df["psnr"], "b-o", markersize=3)
    axes[0, 0].set_xlabel("# Acquired Volumes")
    axes[0, 0].set_ylabel("PSNR (dB)")
    axes[0, 0].set_title("Overall PSNR")
    axes[0, 0].grid(True, alpha=0.3)

    # Panel (0,1): Per-shell PSNR
    for shell in [1000, 2000, 3000]:
        col = f"psnr_b{shell}"
        if col in results_df.columns:
            axes[0, 1].plot(results_df["n_acquired"], results_df[col], "-o", markersize=3, label=f"b={shell}")
    axes[0, 1].set_xlabel("# Acquired Volumes")
    axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].set_title("Per-Shell PSNR")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if has_ssim:
        # Panel (1,0): Overall SSIM
        axes[1, 0].plot(results_df["n_acquired"], results_df["ssim"], "r-o", markersize=3)
        axes[1, 0].set_xlabel("# Acquired Volumes")
        axes[1, 0].set_ylabel("SSIM")
        axes[1, 0].set_title("Overall SSIM")
        axes[1, 0].grid(True, alpha=0.3)

        # Panel (1,1): Per-shell SSIM
        for shell in [1000, 2000, 3000]:
            col = f"ssim_b{shell}"
            if col in results_df.columns:
                axes[1, 1].plot(results_df["n_acquired"], results_df[col], "-o", markersize=3, label=f"b={shell}")
        axes[1, 1].set_xlabel("# Acquired Volumes")
        axes[1, 1].set_ylabel("SSIM")
        axes[1, 1].set_title("Per-Shell SSIM")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved acquisition curves to {output_path}")


def plot_strategy_comparison(csv_paths, labels, output_path):
    """Overlay multiple strategy CSVs on one plot."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for csv_path, label in zip(csv_paths, labels):
        df = pd.read_csv(csv_path)
        axes[0].plot(df["n_acquired"], df["psnr"], "-o", markersize=2, label=label)
        for shell in [1000, 2000, 3000]:
            col = f"psnr_b{shell}"
            if col in df.columns:
                axes[1].plot(df["n_acquired"], df[col], "-", markersize=1, label=f"{label} b={shell}")

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Overall PSNR")
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].set_title("Per-Shell PSNR")
    axes[1].set_ylabel("PSNR (dB)")
    for ax in axes:
        ax.set_xlabel("# Acquired Volumes")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved strategy comparison to {output_path}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(config, device="cuda"):
    """Main acquisition adaptation pipeline."""
    subject = config.get("subject", "114621")

    # Replace subject placeholder in paths
    for key in list(config.keys()):
        if isinstance(config[key], str) and "subject" in config[key]:
            config[key] = config[key].replace("subject", subject)

    print(f"=== Acquisition Adaptation Pipeline ===")
    print(f"Target subject: {subject}")
    print(f"Strategy: {config.get('strategy', 'random')}")
    print(f"Source checkpoint: {config['source_checkpoint']}")

    # 1. Load target subject's full DWI
    print("\n[1/5] Loading target subject data...")
    dwi, mask, bvec, bval = load_data(config)
    normalize_before_b0 = config.get("normalize_before_b0", False)
    result = extract_b0_prior(
        dwi, bvec, bval,
        b0_threshold=config.get("b0_threshold", 10),
        mask=mask,
        normalize_first=normalize_before_b0,
    )
    if normalize_before_b0:
        b0_image, dwi_residual, bvec, bval, dwi_min, dwi_max = result
    else:
        b0_image, dwi_residual, bvec, bval = result
    print(f"  Residual volumes: {dwi_residual.shape[-1]}, mask voxels: {mask.bool().sum().item()}")

    # Extract source normalization params from source subject's data if not in config
    # Must match training: compute y_min/y_max from selected b-value shells only
    def _compute_y_range(residual, bval_vals, bval_train=None):
        if bval_train is not None and bval_train != 'all':
            keep = torch.zeros(bval_vals.shape[0], dtype=torch.bool)
            for b in bval_train:
                keep |= (torch.round(bval_vals / 100) * 100 == b)
            residual = residual[..., keep]
        return residual.min().item(), residual.max().item()

    bval_train = config.get("bval_train", None)
    if config.get("source_y_min") is None:
        source_cfg = config.copy()
        source_subject = config.get("source_subject", "107422")
        for key in list(source_cfg.keys()):
            if isinstance(source_cfg[key], str) and subject in source_cfg[key]:
                source_cfg[key] = source_cfg[key].replace(subject, source_subject)
        print(f"  Loading source subject {source_subject} to extract normalization params...")
        src_dwi, src_mask, src_bvec, src_bval = load_data(source_cfg)
        src_result = extract_b0_prior(
            src_dwi, src_bvec, src_bval,
            b0_threshold=config.get("b0_threshold", 10),
            mask=src_mask,
            normalize_first=normalize_before_b0,
        )
        src_residual = src_result[1]
        src_bval_nb = src_result[3]
        y_min, y_max = _compute_y_range(src_residual, src_bval_nb, bval_train)
        config["source_y_min"] = y_min
        config["source_y_max"] = y_max
        print(f"  Source y_range (bval_train={bval_train}): [{y_min:.4f}, {y_max:.4f}]")
        del src_dwi, src_mask, src_bvec, src_bval, src_residual

    pool = AcquisitionPool(dwi_residual, mask, bvec, bval, b0_image)
    print(f"  Shells: {sorted(pool.shells.keys())}")
    for shell, idxs in sorted(pool.shells.items()):
        print(f"    b={shell}: {len(idxs)} volumes")

    # 2. Load pre-trained checkpoint
    print("\n[2/5] Loading pre-trained model...")
    model = build_model(config, device=device)
    source_state_dict = copy.deepcopy(model.state_dict())

    # Build a reference object for normalization params
    ref_dataset_dummy = type('Obj', (object,), {
        'y_min': config.get("source_y_min", 0),
        'y_max': config.get("source_y_max", 1),
        'dim': dwi_residual.shape,
        'max_bval': config.get("max_bval", 3000),
    })()

    # 3. Precompute all training data (build once, reuse every round)
    print("\n[3/6] Precomputing training data for all volumes...")
    precomp_data = PrecomputedTrainingData(pool, config, device=device)

    # 4. Pick eval subsample (stratified across shells for representativeness)
    eval_subsample = config.get("eval_subsample", 30)
    if eval_subsample and eval_subsample < pool.n_volumes:
        # Stratified subsample: equal from each shell
        eval_vol_indices = []
        shells_sorted = sorted(pool.shells.keys())
        per_shell_n = max(1, eval_subsample // len(shells_sorted))
        for shell in shells_sorted:
            idxs = pool.shells[shell]
            chosen = list(np.random.choice(idxs, size=min(per_shell_n, len(idxs)), replace=False))
            eval_vol_indices.extend(chosen)
        eval_vol_indices = sorted(eval_vol_indices[:eval_subsample])
        print(f"  Eval subsample: {len(eval_vol_indices)} volumes (from {pool.n_volumes})")
    else:
        eval_vol_indices = None  # eval all

    # 5. Zero-shot evaluation
    print("\n[4/6] Evaluating zero-shot transfer...")
    vol_metrics = evaluate_all_volumes(model, pool, config, ref_dataset_dummy,
                                       device=device, eval_vol_indices=eval_vol_indices)
    overall_0, per_shell_0 = aggregate_metrics(vol_metrics)
    ssim_str = f", SSIM: {overall_0['ssim']:.4f}" if 'ssim' in overall_0 else ""
    print(f"  Zero-shot PSNR: {overall_0['psnr']:.2f} dB{ssim_str}")
    for _, row in per_shell_0.iterrows():
        s_str = f", SSIM={row['ssim']:.4f}" if 'ssim' in row else ""
        print(f"    b={int(row['b_value'])}: PSNR={row['psnr']:.2f}{s_str}")

    # 6. Acquisition loop
    print(f"\n[5/6] Running acquisition loop...")
    strategy_name = config.get("strategy", "random")
    strategy = STRATEGIES[strategy_name]()
    n_rounds = config.get("n_rounds", 27)
    volumes_per_round = config.get("volumes_per_round", 10)
    eval_every = config.get("eval_every", 5)

    results = []
    # Store zero-shot result
    zero_shot_row = {"round": 0, "n_acquired": 0}
    for k, v in overall_0.items():
        zero_shot_row[k] = v
    for _, row in per_shell_0.iterrows():
        for c in [c for c in row.index if c != "b_value"]:
            zero_shot_row[f"{c}_b{int(row['b_value'])}"] = row[c]
    results.append(zero_shot_row)

    import time
    t_start = time.time()
    optimizer = None  # persistent optimizer across rounds

    for rnd in range(1, n_rounds + 1):
        if not pool.available_indices:
            print(f"  Round {rnd}: No more volumes available")
            break

        # a. Select volumes
        selected = strategy.select(pool, model=model, dataset=ref_dataset_dummy, config=config)
        pool.acquire(selected)

        # b. Fine-tune (continues from previous round by default)
        optimizer = finetune(model, pool, config, source_state_dict, precomp_data,
                             optimizer=optimizer, device=device)

        # c. Evaluate
        should_eval = (rnd % eval_every == 0) or (rnd == 1) or (rnd == n_rounds)
        if should_eval:
            vol_metrics = evaluate_all_volumes(model, pool, config, ref_dataset_dummy,
                                               device=device, eval_vol_indices=eval_vol_indices)
            overall, per_shell = aggregate_metrics(vol_metrics)

            row = {"round": rnd, "n_acquired": len(pool.acquired_indices)}
            for k, v in overall.items():
                row[k] = v
            for _, ps_row in per_shell.iterrows():
                for c in [c for c in ps_row.index if c != "b_value"]:
                    row[f"{c}_b{int(ps_row['b_value'])}"] = ps_row[c]
            results.append(row)

            elapsed = time.time() - t_start
            eta = elapsed / rnd * (n_rounds - rnd) / 60
            ssim_str = f" SSIM={overall['ssim']:.4f}" if 'ssim' in overall else ""
            print(f"  Round {rnd}/{n_rounds}: acq={len(pool.acquired_indices)} PSNR={overall['psnr']:.2f} dB{ssim_str}  [{elapsed:.0f}s elapsed, ~{eta:.1f}min left]")

    # 6. Save results
    print(f"\n[6/6] Saving results...")
    results_df = pd.DataFrame(results)
    output_dir = config.get("output_dir", f"outputs/acquisition_{strategy_name}_{subject}")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "acquisition_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved CSV to {csv_path}")

    plot_path = os.path.join(output_dir, "acquisition_curves.png")
    plot_acquisition_curves(results_df, plot_path)

    # Save config for reproducibility
    with open(os.path.join(output_dir, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n=== Pipeline complete ===")
    print(f"Final PSNR ({len(pool.acquired_indices)} volumes): {results_df.iloc[-1]['psnr']:.2f} dB")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Acquisition-like adaptation for DWI INR")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--subject", type=str, default=None, help="Target subject ID")
    parser.add_argument("--strategy", type=str, default=None, choices=list(STRATEGIES.keys()))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compare", nargs="+", help="CSV files to compare strategies")
    parser.add_argument("--compare_labels", nargs="+", help="Labels for strategy comparison")
    args = parser.parse_args()

    # Strategy comparison mode
    if args.compare:
        labels = args.compare_labels or [Path(p).stem for p in args.compare]
        out = args.output_dir or "strategy_comparison.png"
        plot_strategy_comparison(args.compare, labels, out)
        return

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.subject:
        config["subject"] = args.subject
    if args.strategy:
        config["strategy"] = args.strategy
    if args.output_dir:
        config["output_dir"] = args.output_dir

    run_pipeline(config, device=args.device)


if __name__ == "__main__":
    main()
