import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from inrdiff.utils import load_data, split_datasets, extract_b0_prior
from inrdiff.model import INRLightningModule
from inrdiff.dataset import (
    DWIReprDataset,
    PreBatchedDWIReprDataset,
    PreBatchedSHOutputDWIReprDataset,
    PreBatchedShoreOutputDWIReprDataset,
    PreBatchedBesselOutputDWIReprDataset,
)
from inrdiff.val_dataset import ValidationSliceDataset
from inrdiff.networks.inrs import WIRENMLP, SIRENMLP, ReLUMLP, ResidualGatedMLP, INCODEMLP, PINMLP
from inrdiff.networks.leaky_relu_inr import LeakyReLUMLP
from inrdiff.networks.pos_encoders import BvecAngleEncoder, SphericalPosEncoder, SpatialBvalEncoder, SpatialOnlyEncoder
# from pytorch_lightning.profilers import PyTorchProfiler
import os
import copy
torch.set_float32_matmul_precision('medium') 


model_mapping = {
    "leaky_relu_mlp": LeakyReLUMLP,
    "wiren_mlp": WIRENMLP,
    "siren_mlp": SIRENMLP,
    "relu_mlp": ReLUMLP,
    "residual_gated_mlp": ResidualGatedMLP,
    "incode_mlp": INCODEMLP,
    "pin_mlp": PINMLP,
}


def configure_training_seed(config):
    """Optional deterministic seeding for reproducible experiment comparisons."""
    seed = config.get('train_seed', None)
    if seed is None:
        return
    seed = int(seed)
    workers = bool(config.get('train_seed_workers', True))
    deterministic = bool(config.get('train_deterministic', True))
    strict_deterministic = bool(config.get('train_strict_deterministic', False))
    deterministic_warn_only = bool(config.get('train_deterministic_warn_only', True))
    pl.seed_everything(seed, workers=workers)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        if strict_deterministic:
            torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
    print(
        f"Training seed configured: seed={seed}, workers={workers}, "
        f"deterministic={deterministic}, strict={strict_deterministic}, "
        f"warn_only={deterministic_warn_only}"
    )


def _load_data_cached(config):
    """Wrap load_data with a disk cache of the decompressed tensors.

    The .nii.gz decompression of the full 4D volume dominates setup (~28s on
    HCP). The loaded (dwi, mask, bvec, bval) tensors depend only on the data
    files, not on any network/training choice, so one cache is reused across a
    whole architecture sweep. Keyed on file paths + mtimes for auto-invalidation.
    Opt-in via `cache_loaded_data: true`.
    """
    if not config.get('cache_loaded_data', False):
        return load_data(config)
    import os, hashlib
    paths = [config['data'], config['mask'], config['bvec'], config['bval']]
    sig = "|".join(f"{p}:{os.path.getmtime(p)}" for p in paths)
    h = hashlib.md5(sig.encode()).hexdigest()[:16]
    cache_dir = config.get('data_cache_dir',
                           os.path.join(os.path.dirname(__file__), '..', '.binr_data_cache'))
    cache_file = os.path.join(cache_dir, f"load_{h}.pt")
    if os.path.exists(cache_file):
        d = torch.load(cache_file)
        print(f"[cache] loaded data tensors from {cache_file}")
        return d['dwi'], d['mask'], d['bvec'], d['bval']
    dwi, mask, bvec, bval = load_data(config)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({'dwi': dwi, 'mask': mask, 'bvec': bvec, 'bval': bval}, cache_file)
    print(f"[cache] wrote data tensors to {cache_file}")
    return dwi, mask, bvec, bval


def setup_data_handlers(config):
    dwi, mask, bvec, bval = _load_data_cached(config)

    # Apply B0 prior: extract B0 image and convert DWI to residuals
    b0_image = None
    normalize_before_b0 = config.get('normalize_before_b0', False)
    if config.get('use_b0_prior', False):
        result = extract_b0_prior(
            dwi, bvec, bval,
            b0_threshold=config.get('b0_threshold', 10),
            mask=mask,
            normalize_first=normalize_before_b0,
        )
        if normalize_before_b0:
            b0_image, dwi, bvec, bval, dwi_min, dwi_max = result
            print(f"B0 prior (normalize_first): dwi_min={dwi_min:.2f}, dwi_max={dwi_max:.2f}")
        else:
            b0_image, dwi, bvec, bval = result
        print(f"B0 prior: extracted B0 image, {dwi.shape[-1]} non-B0 volumes remain")

    # Split datasets if train_split and val_split are provided
    train_split = config.get('train_split', None)
    val_split = config.get('val_split', None)
    if train_split is not None and val_split is not None:
        print(f"Splitting dataset: train={train_split}, val={val_split} stratified by b-values")
        (train_dwi, train_bvec, train_bval), \
        (val_dwi, val_bvec, val_bval) = split_datasets(dwi, bvec, bval, 
                                                      train_split=train_split,
                                                      val_split=val_split)
    else:
        print("Using full dataset for both training and validation")
        train_dwi, train_bvec, train_bval = dwi, bvec, bval
        val_dwi, val_bvec, val_bval = dwi, bvec, bval

    # Set default values for normalization if not specified in config
    normalize_0_1 = config.get('normalize_0_1', False)
    normalize_in_out = config.get('normalize_in_out', True) #backwards compatibility
    print(f"normalize_0_1: {normalize_0_1}, normalize_in_out: {normalize_in_out}")

    sh_output = config.get('sh_output', False)
    shore_output = config.get('shore_output', False)

    # For SH/SHORE output without B0 prior, remove B0 volumes
    shore_b0_loss = shore_output and config.get('shore_b0_loss', False)
    b0_image_for_loss = None
    b0_image_for_prior = None
    if (sh_output or shore_output) and not config.get('use_b0_prior', False):
        b0_threshold = config.get('b0_threshold', 10)
        b0_mask = train_bval <= b0_threshold

        # Compute B0 image for shore_b0_loss / b0_anchor_weight before any modifications
        if b0_mask.any():
            b0_image_for_prior = train_dwi[..., b0_mask].float().mean(dim=-1)
            b0_anchor_weight_cfg = float(config.get('b0_anchor_weight', 0.0))
            if shore_b0_loss or b0_anchor_weight_cfg > 0:
                b0_image_for_loss = b0_image_for_prior

        # SHORE B0 normalization: E = DWI / S0 before removing B0 volumes
        if shore_output and config.get('shore_b0_normalize', False) and b0_mask.any():
            b0_image_train = train_dwi[..., b0_mask].float().mean(dim=-1)
            b0_safe = b0_image_train.clamp(min=1.0)
            train_dwi = train_dwi.float() / b0_safe[..., None]
            # Clip E to physical range: E = DWI/S0 should be <= 1 for diffusion
            e_clip = config.get('shore_e_clip', 1.5)
            train_dwi = train_dwi.clamp(0, e_clip)
            # B0/B0 ≈ 1, will be removed below
            print(f"SHORE B0 normalization: E = DWI / S0 (S0 mean={b0_image_train[mask.bool()].mean():.1f}), E clipped to [0, {e_clip}]")

        if b0_mask.any():
            non_b0 = ~b0_mask
            train_dwi = train_dwi[..., non_b0]
            train_bvec = train_bvec[non_b0]
            train_bval = train_bval[non_b0]
            mode_name = "SHORE" if shore_output else "SH"
            print(f"{mode_name} output: removed {b0_mask.sum().item()} B0 volumes, "
                  f"{train_dwi.shape[-1]} remain")

    # Training-side b-vector downsampling: randomly keep N directions per
    # non-B0 shell for the training INR, while leaving evaluation on the
    # full direction set. Used for E2 (research_plan §4). The selected
    # indices (in non-B0 coordinates) are written into config so the
    # evaluator can score on the COMPLEMENT (held-out directions only).
    train_bvec_per_shell = config.get('train_bvec_per_shell', None)
    if train_bvec_per_shell:
        seed = int(config.get('train_bvec_seed', 42))
        rng = np.random.default_rng(seed)
        # Round to the nearest 100, not 1000. ParkMR has shells like
        # b=100/700/1400/2000 — rounding to 1000 collapses b=100 into 0 and
        # excludes it from training entirely, biasing held-out evaluation.
        b0_threshold = float(config.get('b0_threshold', 10))
        bval_rounded = (train_bval.float() / 100.0).round() * 100
        unique_shells = sorted({
            int(b.item()) for b in bval_rounded if float(b.item()) > b0_threshold
        })
        keep_indices = []
        for shell in unique_shells:
            mask_shell = (bval_rounded.long() == shell).nonzero(as_tuple=False).squeeze(-1).tolist()
            n = min(int(train_bvec_per_shell), len(mask_shell))
            chosen = rng.choice(mask_shell, size=n, replace=False).tolist()
            keep_indices.extend(chosen)
        keep_indices = sorted(keep_indices)
        config['_train_bvec_keep_indices'] = list(keep_indices)
        train_dwi = train_dwi[..., keep_indices]
        train_bvec = train_bvec[keep_indices]
        train_bval = train_bval[keep_indices]
        print(f"Training b-vector downsampling: keeping {len(keep_indices)} directions "
              f"({int(train_bvec_per_shell)}/shell across {len(unique_shells)} shells "
              f"= {sorted(unique_shells)})")

    if shore_output:
        basis = str(config.get('basis', 'shore')).lower()
        shared_kwargs = dict(
            normalize_0_1=normalize_0_1,
            normalize_in_out=normalize_in_out,
            bval_selection=config.get('bval_train') if config.get('bval_train') else 'all',
            bval_encoding=config.get('bval_encoding', 'linear'),
            preload_to_gpu=config.get('preload_to_gpu', False),
            device='cuda:0',
            train_batch_size=config['train_batch_size'],
            skip_y_normalize=config.get('shore_b0_normalize', False),
            b0_loss_mode=shore_b0_loss,
            b0_image=b0_image_for_loss,
        )
        if basis == 'bessel':
            train_dataset = PreBatchedBesselOutputDWIReprDataset(
                train_dwi, mask, train_bvec, train_bval,
                bessel_radial_order=config.get('bessel_radial_order', 2),
                bessel_lmax=config.get('bessel_lmax', 8),
                bessel_q_max=config.get('bessel_q_max', None),
                **shared_kwargs,
            )
        elif basis == 'shore':
            train_dataset = PreBatchedShoreOutputDWIReprDataset(
                train_dwi, mask, train_bvec, train_bval,
                shore_radial_order=config.get('shore_radial_order', 6),
                shore_zeta=config.get('shore_zeta', 700),
                **shared_kwargs,
            )
        else:
            raise ValueError(f"Unknown basis '{basis}' — expected 'shore' or 'bessel'")
    elif sh_output:
        train_dataset = PreBatchedSHOutputDWIReprDataset(
            train_dwi, mask, train_bvec, train_bval,
            sh_lmax=config.get('sh_lmax', 8),
            normalize_0_1=normalize_0_1,
            normalize_in_out=normalize_in_out,
            bval_selection=config.get('bval_train') if config.get('bval_train') else 'all',
            bval_encoding=config.get('bval_encoding', 'linear'),
            preload_to_gpu=config.get('preload_to_gpu', False),
            device='cuda:0',
            train_batch_size=config['train_batch_size'],
        )
    else:
        train_dataset = PreBatchedDWIReprDataset(train_dwi, mask, train_bvec, train_bval,
                                      normalize_0_1=normalize_0_1,
                                      normalize_in_out=normalize_in_out,
                                      bval_selection=config.get('bval_train') if config.get('bval_train') else 'all',
                                      bval_encoding=config.get('bval_encoding', 'linear'),
                                      preload_to_gpu=config.get('preload_to_gpu', False),
                                      device='cuda:0',
                                      train_batch_size=config['train_batch_size'],
                                      )
    
    # Store pre-normalization params if normalize_before_b0 was used
    if normalize_before_b0:
        train_dataset.dwi_min = dwi_min
        train_dataset.dwi_max = dwi_max

    # Store B0 context image for optional INCODE contextual prior.
    if b0_image_for_prior is not None:
        train_dataset.b0_context_image = b0_image_for_prior
    elif b0_image is not None:
        train_dataset.b0_context_image = b0_image

    # Idea 5b — per-voxel mono-exp baseline. Fit (S0, D) per voxel from
    # training-shell signals via 2-shell log-linear regression. These
    # serve as targets at sampled q-points for the q_monoexp consistency
    # loss (model's signal at any b should follow S0·exp(-b·D)).
    # Uses normalized signal space (assumes y_min ≈ 0); ParkMR y_min ≈ -14
    # vs range 3607 → offset 0.4% which we approximate as 0.
    if config.get('q_monoexp_weight', 0.0) > 0 and shore_output \
            and hasattr(train_dataset, 'y_norm') and hasattr(train_dataset, 'bvals'):
        bvals_train_arr = train_dataset.bvals.float().cpu().numpy()
        bvals_rounded = np.round(bvals_train_arr / 100.0) * 100
        unique_shells = sorted(set(int(b) for b in bvals_rounded))
        if len(unique_shells) >= 2:
            # Two highest shells (e.g. 1400 and 2000 for ParkMR).
            b_lo, b_hi = unique_shells[-2], unique_shells[-1]
            mask_lo = bvals_rounded == b_lo
            mask_hi = bvals_rounded == b_hi
            # train_dataset.y_norm is shuffled+padded [N, n_volumes] in normalized space.
            y_norm_np = train_dataset.y_norm.cpu().numpy()
            s_lo = y_norm_np[:, mask_lo].mean(axis=1)
            s_hi = y_norm_np[:, mask_hi].mean(axis=1)
            eps = 1e-4
            s_lo_safe = np.maximum(s_lo, eps)
            s_hi_safe = np.maximum(s_hi, eps)
            D_voxel = (np.log(s_lo_safe) - np.log(s_hi_safe)) / float(b_hi - b_lo)
            D_voxel = np.clip(D_voxel, 0.0, 5e-3)
            S0_voxel = s_lo_safe * np.exp(b_lo * D_voxel)
            S0_voxel = np.clip(S0_voxel, 0.0, 5.0)
            # Reshape to [num_batches, batch_size]; dataset already padded.
            B = int(config['train_batch_size'])
            N = y_norm_np.shape[0]
            assert N % B == 0, f"unexpected non-divisible y_norm length {N} for batch {B}"
            num_batches = N // B
            train_dataset.q_monoexp_S0_batches = torch.from_numpy(
                S0_voxel.reshape(num_batches, B)
            ).float()
            train_dataset.q_monoexp_D_batches = torch.from_numpy(
                D_voxel.reshape(num_batches, B)
            ).float()
            print(
                f"q_monoexp targets fitted: shells [{b_lo}, {b_hi}], "
                f"S0 range [{S0_voxel.min():.3f}, {S0_voxel.max():.3f}], "
                f"D range [{D_voxel.min()*1e3:.3f}e-3, {D_voxel.max()*1e3:.3f}e-3]"
            )

    default_train_workers = 0 if config.get('preload_to_gpu', False) else 0
    train_num_workers = int(config.get('train_num_workers', default_train_workers))
    train_shuffle = bool(config.get('train_shuffle', False))

    train_dataloaders = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=train_shuffle,
        num_workers=train_num_workers,
    )
    
    # Define eval_gradients (indices of gradient directions to evaluate during training)
    # Default to evaluating low, medium, and high b-values if available
    if 'eval_gradients' in config:
        eval_gradients = config['eval_gradients']
    else:
        # Try to automatically select a few representative gradient directions
        # Check for both 'bval' and 'bvals' attributes to be more robust
        if hasattr(train_dataset, 'bvals'):
            b_values = train_dataset.bvals
        elif hasattr(train_dataset, 'bval'):
            b_values = train_dataset.bval
        else:
            # Fallback to using all gradients if no b-values found
            b_values = torch.arange(train_dataset.data.shape[-1])
            print("Warning: No b-values found in dataset. Using all gradients.")
            

        # Get indices for low, medium, and high b-values
        sorted_indices = torch.argsort(b_values)
        eval_gradients = [
            sorted_indices[0].item(),  # Lowest b-value
            sorted_indices[len(sorted_indices)//2].item(),  # Middle b-value
            sorted_indices[-1].item()  # Highest b-value
        ]
     
    if not config.get('val_skip', False):
        val_dataset = DWIReprDataset(val_dwi, mask, val_bvec, val_bval, 
                                      normalize_0_1=normalize_0_1, 
                                      normalize_in_out=normalize_in_out, 
                                      bval_selection=config.get('bval_val') if config.get('bval_val') else 'all'
                                    )
        val_slice_dataset = ValidationSliceDataset(val_dataset, 
                                             axis=config['val_slice_axis'], 
                                             offset=config['val_slice_offset'],
                                             b_indices=config.get('val_b_idx', [0, 15, 30]),
                                             normalize_0_1=config.get('normalize_0_1', True), 
                                             normalize_in_out=normalize_in_out)

        # currently same as train loader - TODO some of these stuff belongs into config
        val_volume_dataset = DWIReprDataset(val_dwi, mask, val_bvec, val_bval,
                                        normalize_0_1=config.get('normalize_0_1', True),
                                        normalize_in_out=normalize_in_out,
                                        subsampl_step=4,
                                        bval_selection=[0, 3000])
        val_slice_loader = DataLoader(val_slice_dataset, 
                                      batch_size=config['val_batch_size'], shuffle=False, num_workers=6)
        val_volume_loader = DataLoader(val_volume_dataset, 
                                      batch_size=val_volume_dataset.num_samples, shuffle=False, num_workers=6)
        val_dataloaders = {'slice': val_slice_loader, 'volume': val_volume_loader} 

        return val_slice_dataset, train_dataloaders, val_dataloaders, eval_gradients, b0_image

    else:
        return None, train_dataloaders, None, eval_gradients, b0_image



def get_checkpoint_callback(config):
    checkpoint_dirpath = config.get('checkpoint_dirpath', 'checkpoints')
    checkpoint_filename = config.get('checkpoint_filename', 'inr-{step}')
    checkpoint_save_top_k = config.get('checkpoint_save_top_k', -1)  # -1 means keep all checkpoints
    checkpoint_every_n_train_steps = config.get('checkpoint_every_n_train_steps', 5000)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        filename=checkpoint_filename,
        save_top_k=checkpoint_save_top_k,
        every_n_train_steps=checkpoint_every_n_train_steps,
        save_last=True,     # for now hardcoded
    )
    
    return checkpoint_callback


def load_config(config_path, subject=None):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    for key in config.keys():
        if isinstance(config[key], str) and "subject" in config[key] and subject is not None:
            config[key] = config[key].replace("subject", subject)
    return config


def train_with_config(config):
    import time as _time
    _t_setup0 = _time.perf_counter()
    config = copy.deepcopy(config)
    configure_training_seed(config)
    matmul_precision = config.get('float32_matmul_precision', None)
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(str(matmul_precision))
        print(f"Set float32 matmul precision: {matmul_precision}")
    _t_data0 = _time.perf_counter()
    val_slice_dataset, train_dataloaders, val_dataloaders, eval_gradients, b0_image = setup_data_handlers(config)
    print(f"[timing] setup_data_handlers: {_time.perf_counter() - _t_data0:.1f}s")

    # Store b0_image on train dataset for downstream access
    if b0_image is not None:
        train_dataloaders.dataset.b0_image = b0_image
    
    # setup input positional encoder (before network, so in_size can be overridden)
    use_pos_encoder = config.get('pos_encoder', True) #backwards compatibility
    sh_output = config.get('sh_output', False)
    shore_output = config.get('shore_output', False)
    shore_b0_loss = shore_output and config.get('shore_b0_loss', False)
    y_matrix = None
    shore_matrix = None
    train_dataset = train_dataloaders.dataset

    if shore_output:
        basis_name = str(config.get('basis', 'shore')).lower()
        print(f"Initializing SpatialOnlyEncoder for {basis_name.upper()} output mode")
        spatial_encoder = SpatialOnlyEncoder(
            coord_size=3,
            freq_num=config.get('spatial_freq_num', 128),
            freq_scale=config.get('spatial_freq_scale', 1.0),
            freq_learnable=config.get('spatial_freq_learnable', False),
        )
        pos_encoder = [spatial_encoder]
        expected_in_size = spatial_encoder.out_size
        # Per-voxel B0 input (Idea 2): adds 1 feature concat'd after pos_encoder.
        if config.get('use_b0_input', False):
            expected_in_size = expected_in_size + 1
            print(f"  use_b0_input=true: extending in_size by 1 for B0 lookup")
        if config['in_size'] != expected_in_size:
            print(f"  Overriding in_size: {config['in_size']} -> {expected_in_size} (from SpatialOnlyEncoder.out_size)")
            config['in_size'] = expected_in_size

        n_shore = train_dataset.n_shore
        # Ball-and-stick mode overrides out_size to 4·n_sticks + 3 instead
        # of the SHORE coefficient count.
        if config.get('model_output_type', 'shore') == 'ball_stick':
            from inrdiff.ball_stick import (n_raw_params as _bs_n_raw_params,
                                    n_raw_params_fodf as _bs_n_raw_params_fodf)
            n_sticks_cfg = int(config.get('ball_stick_n_sticks', 2))
            use_fodf = bool(config.get('ball_stick_use_fodf', False))
            l_max = int(config.get('ball_stick_fodf_l_max', 6))
            unpin_c00 = bool(config.get('ball_stick_fodf_unpin_c00', False))
            if use_fodf:
                bs_out = _bs_n_raw_params_fodf(l_max, unpin_c00=unpin_c00)
            else:
                bs_out = _bs_n_raw_params(n_sticks_cfg)
            if config.get('out_size', 1) != bs_out:
                tag = f"fodf l_max={l_max}" if use_fodf else f"sticks N={n_sticks_cfg}"
                print(f"  Overriding out_size: {config.get('out_size', 1)} -> {bs_out} (ball-{tag})")
                config['out_size'] = bs_out
        elif config.get('out_size', 1) != n_shore:
            print(f"  Overriding out_size: {config.get('out_size', 1)} -> {n_shore} ({basis_name} coefficients)")
            config['out_size'] = n_shore

        shore_matrix = train_dataset.shore_matrix.float()
        print(f"  {basis_name.upper()} design matrix shape: {shore_matrix.shape}")

    elif sh_output:
        print("Initializing SpatialBvalEncoder for SH output mode")
        spatial_encoder = SpatialBvalEncoder(
            coord_size=4,
            freq_num=128,
            freq_scale=1.0,
            bval_freq_num=config.get('bval_freq_num', 16),
            bval_freq_scale=config.get('bval_freq_scale', 1.0),
        )
        pos_encoder = [spatial_encoder]
        expected_in_size = spatial_encoder.out_size
        if config['in_size'] != expected_in_size:
            print(f"  Overriding in_size: {config['in_size']} -> {expected_in_size} (from SpatialBvalEncoder.out_size)")
            config['in_size'] = expected_in_size

        # Set out_size to number of SH coefficients
        n_sh = train_dataset.n_sh
        if config.get('out_size', 1) != n_sh:
            print(f"  Overriding out_size: {config.get('out_size', 1)} -> {n_sh} (SH coefficients)")
            config['out_size'] = n_sh

        y_matrix = train_dataset.y_matrix.float()
        print(f"  Y_matrix shape: {y_matrix.shape}")

    elif use_pos_encoder:
        print("Initializing pos encoder")
        sph_encoder = SphericalPosEncoder(
            l_max=8,
            coord_size=4,
            freq_num=128,
            freq_scale=1.,
            bval_freq_num=config.get('bval_freq_num', None),
            bval_freq_scale=config.get('bval_freq_scale', None),
        )
        pos_encoder = [BvecAngleEncoder(7, (4, 5, 6)), sph_encoder]
        # Override in_size from config if pos_encoder determines it dynamically
        expected_in_size = sph_encoder.out_size
        if config['in_size'] != expected_in_size:
            print(f"  Overriding in_size: {config['in_size']} -> {expected_in_size} (from pos_encoder.out_size)")
            config['in_size'] = expected_in_size
    else:
        pos_encoder = None

    # setup model architecture
    network_kwargs = {
        'hidden_size': config['hidden_size'],
        'num_layers': config['num_layers'],
    }
    network_name = config['network_name']
    if network_name == 'wiren_mlp':
        if 'wire_omega' in config:
            network_kwargs['wire_omega'] = config['wire_omega']
        if 'wire_sigma' in config:
            network_kwargs['wire_sigma'] = config['wire_sigma']
        if 'wire_input_skip' in config:
            network_kwargs['wire_input_skip'] = config['wire_input_skip']
    elif network_name == 'siren_mlp' and 'siren_factor' in config:
        network_kwargs['siren_factor'] = config['siren_factor']
    elif network_name == 'pin_mlp':
        for k in ('pin_first_omega', 'pin_hidden_omega', 'pin_c', 'pin_max_even_k',
                  'pin_b0_context', 'pin_context_dim', 'pin_harmonizer_hidden', 'pin_residual_skip'):
            if k in config:
                network_kwargs[k] = config[k]
    elif network_name == 'incode_mlp':
        if 'incode_first_omega_0' in config:
            network_kwargs['incode_first_omega_0'] = config['incode_first_omega_0']
        if 'incode_hidden_omega_0' in config:
            network_kwargs['incode_hidden_omega_0'] = config['incode_hidden_omega_0']
        if 'incode_latent_dim' in config:
            network_kwargs['incode_latent_dim'] = config['incode_latent_dim']
        if 'incode_prior_mode' in config:
            network_kwargs['incode_prior_mode'] = config['incode_prior_mode']
        if 'incode_context_dim' in config:
            network_kwargs['incode_context_dim'] = config['incode_context_dim']
        if 'incode_layerwise_modulation' in config:
            network_kwargs['incode_layerwise_modulation'] = config['incode_layerwise_modulation']
        if 'incode_harmonizer_hidden' in config:
            network_kwargs['incode_harmonizer_hidden'] = tuple(config['incode_harmonizer_hidden'])
        if 'incode_harmonizer_bias' in config:
            network_kwargs['incode_harmonizer_bias'] = config['incode_harmonizer_bias']
        if 'incode_harmonizer_weight_std' in config:
            network_kwargs['incode_harmonizer_weight_std'] = config['incode_harmonizer_weight_std']
        if 'incode_activation_mode' in config:
            network_kwargs['incode_activation_mode'] = config['incode_activation_mode']
        if 'incode_enforce_positive' in config:
            network_kwargs['incode_enforce_positive'] = config['incode_enforce_positive']
        if 'incode_input_skip' in config:
            network_kwargs['incode_input_skip'] = config['incode_input_skip']
        if 'incode_direct_modulation' in config:
            network_kwargs['incode_direct_modulation'] = config['incode_direct_modulation']
        if 'incode_first_layer_init' in config:
            network_kwargs['incode_first_layer_init'] = config['incode_first_layer_init']
        if 'incode_residual_skip' in config:
            network_kwargs['incode_residual_skip'] = config['incode_residual_skip']
        for k in ('incode_shore_factorized', 'incode_shore_radial_order', 'incode_shore_residual_init', 'incode_output_refine_hidden'):
            if k in config:
                network_kwargs[k] = config[k]
        # Idea 5: mono-exp residual factorization. Pull shore_pinv & bvals from
        # the dataset and forward to the network if use_monoexp_head is set.
        if config.get('use_monoexp_head', False) and shore_output \
                and hasattr(train_dataset, 'shore_pinv'):
            network_kwargs['use_monoexp_head'] = True
            network_kwargs['shore_pinv'] = train_dataset.shore_pinv.float()
            network_kwargs['bvals_train'] = train_dataset.bvals.float()
            network_kwargs['monoexp_S0_init'] = float(config.get('monoexp_S0_init', 0.5))
            network_kwargs['monoexp_D_init'] = float(config.get('monoexp_D_init', 1.0e-3))
            print(
                f"use_monoexp_head=true: shore_pinv {tuple(train_dataset.shore_pinv.shape)}, "
                f"bvals_train {tuple(train_dataset.bvals.shape)}, "
                f"S0_init={config.get('monoexp_S0_init', 0.5)}, D_init={config.get('monoexp_D_init', 1.0e-3)}"
            )

    inr = model_mapping[network_name](
        config['in_size'],
        config['out_size'],
        **network_kwargs,
    )

    # Optional contextual prior from B0 statistics (INCODE harmonizer, or PIN b0-context).
    if (
        ((network_name == 'incode_mlp'
          and config.get('incode_prior_mode', 'learned') in {'b0_context', 'fusion'})
         or (network_name == 'pin_mlp' and config.get('pin_b0_context', False)))
        and hasattr(inr, 'set_context_vector')
    ):
        def _compute_b0_context_vector(train_ds, context_dim: int):
            b0_img = getattr(train_ds, 'b0_context_image', None)
            mask = getattr(train_ds, 'mask', None)
            if b0_img is None or mask is None:
                return torch.zeros(context_dim, dtype=torch.float32)
            mask = mask.to(device=b0_img.device).bool()
            vox = b0_img[mask].float().reshape(-1)
            if vox.numel() == 0:
                return torch.zeros(context_dim, dtype=torch.float32)
            q = torch.quantile(vox, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=vox.device))
            mean = vox.mean()
            std = vox.std(unbiased=False)
            rms = torch.sqrt((vox.square()).mean())
            base = torch.stack([mean, std, vox.min(), vox.max(), q[0], q[1], q[2], q[3], q[4], rms])
            scale = mean.abs().clamp_min(1e-6)
            base = base / scale
            if context_dim <= base.numel():
                return base[:context_dim].detach().cpu()
            out = torch.zeros(context_dim, dtype=torch.float32)
            out[: base.numel()] = base.detach().cpu()
            return out

        context_dim = (int(config.get('pin_context_dim', 10)) if network_name == 'pin_mlp'
                       else int(config.get('incode_context_dim', 8)))
        context_vec = _compute_b0_context_vector(train_dataset, context_dim)
        inr.set_context_vector(context_vec)
        print(f"Initialized {network_name} B0 context vector (dim={context_dim})")

    # run training
    logger = TensorBoardLogger("tb_logs", name="my_model", log_graph=False, default_hp_metric=False)

    # Get the maximum b-value from the dataset for proper unnormalization
    data_bval_max = train_dataloaders.dataset.bvals.max()
    max_bval = config.get('max_bval', data_bval_max) if config.get('max_bval') else data_bval_max
    
    # Get steps_per_viz from config, default to 2000 if not specified
    steps_per_viz = config.get('steps_per_viz', 2000)
    if not os.path.exists(config["checkpoint_dirpath"]):
        os.makedirs(config["checkpoint_dirpath"])
        
    inr_lighning_kwargs={
        'network': inr,
        'train_loader': train_dataloaders,
        'val_loaders_dict': val_dataloaders,
        'lr': config['lr'],
        'log_coords_raw': config['log_coords_raw'],
        'log_coords_encoded': config['log_coords_encoded'],
        'eval_gradients': eval_gradients,
        'name': config.get('model_name', ''),
        'max_bval': max_bval,
        'steps_per_viz': steps_per_viz,
        'export_dwi': config["val_export_dwi"],
        'epochs': config['epochs'],
        'use_cos_lrsheduler': config.get('use_cos_lrsheduler', False),
        'train_log_interval': config.get('train_log_interval', 1),
        'weight_decay': config.get('weight_decay', 0.0),
        'use_dropout': config.get('use_dropout', False),
        'b0_image': b0_image,
        'bval_smooth_weight': config.get('bval_smooth_weight', 0.0),
        'bval_smooth_delta': config.get('bval_smooth_delta', 0.05),
        'sh_output': sh_output,
        'y_matrix': y_matrix,
        'shore_output': shore_output,
        'shore_matrix': shore_matrix,
        'shore_b0_loss': shore_b0_loss,
        'shore_aniso_weight': config.get('shore_aniso_weight', 0.0),
        'shore_shell_norm': config.get('shore_shell_norm', False),
        'shore_coeff_weight': config.get('shore_coeff_weight', 0.0),
        'shore_range_weight': config.get('shore_range_weight', 0.0),
        'shore_range_max': config.get('shore_range_max', 1.0),
        'b0_anchor_weight': config.get('b0_anchor_weight', 0.0),
        'q_smooth_weight': config.get('q_smooth_weight', 0.0),
        'q_mono_weight': config.get('q_mono_weight', 0.0),
        'q_st_weight': config.get('q_st_weight', 0.0),
        'q_convex_weight': config.get('q_convex_weight', 0.0),
        'q_monoexp_weight': config.get('q_monoexp_weight', 0.0),
        'b_inf_anchor_weight': config.get('b_inf_anchor_weight', 0.0),
    }

    # Idea 2: per-voxel B0 conditioning. Build a normalized 3D B0 volume
    # (matching the y-target normalization the dataset applied) and hand
    # it to the Lightning module so its forward() can grid_sample it as
    # an extra input feature.
    if config.get('use_b0_input', False) and shore_output and b0_image is not None:
        try:
            ds = train_dataloaders.dataset
            y_min = float(ds.y_min)
            y_max = float(ds.y_max)
            b0_full = b0_image.float().cpu()
            if config.get('normalize_0_1', False):
                denom = max(y_max - y_min, 1e-6)
                b0_norm_3d = (b0_full - y_min) / denom
            else:
                y_mean = float(ds.y_mean)
                y_std = float(ds.y_std)
                b0_norm_3d = (b0_full - y_mean) / max(y_std, 1e-6)
            inr_lighning_kwargs['b0_input_volume'] = b0_norm_3d
            print(f"use_b0_input: passing normalized B0 volume {tuple(b0_norm_3d.shape)} "
                  f"(range [{b0_norm_3d.min():.3f}, {b0_norm_3d.max():.3f}])")
        except (AttributeError, TypeError, ValueError) as e:
            print(f"use_b0_input requested but failed to build normalized volume: {e}")

    # Ball-and-stick mode (physics forward model). Network outputs raw
    # parameters per voxel; the lightning module evaluates the closed-form
    # signal via `ball_stick.signal_at`. Wire bvals + bvecs as buffers and
    # set out_size = 4·n_sticks + 3.
    if config.get('model_output_type', 'shore') == 'ball_stick' and shore_output:
        from inrdiff.ball_stick import (n_raw_params as _bs_n_raw_params,
                                n_raw_params_fodf as _bs_n_raw_params_fodf,
                                compute_sh_basis_at_bvecs as _bs_sh_basis,
                                make_legendre_quadrature as _bs_make_quad,
                                sh_coef_to_l_index as _bs_sh_l_idx)
        n_sticks = int(config.get('ball_stick_n_sticks', 2))
        use_fodf = bool(config.get('ball_stick_use_fodf', False))
        l_max = int(config.get('ball_stick_fodf_l_max', 6))
        use_spot = bool(config.get('ball_stick_use_spot', False))
        spot_K = int(config.get('ball_stick_spot_K', 8))
        ds = train_dataloaders.dataset
        bvals_full = ds.bvals.float().cpu()
        bvecs_full = ds.bvec.float().cpu() if hasattr(ds, 'bvec') else ds.bvecs.float().cpu()
        inr_lighning_kwargs['model_output_type'] = 'ball_stick'
        inr_lighning_kwargs['ball_stick_n_sticks'] = n_sticks
        inr_lighning_kwargs['ball_stick_use_fodf'] = use_fodf
        inr_lighning_kwargs['ball_stick_compile'] = bool(
            config.get('ball_stick_compile', config.get('torch_compile', False)))
        inr_lighning_kwargs['ball_stick_fodf_l_max'] = l_max
        inr_lighning_kwargs['ball_stick_use_spot'] = use_spot
        inr_lighning_kwargs['ball_stick_spot_K'] = spot_K
        inr_lighning_kwargs['bvals_train_full'] = bvals_full
        inr_lighning_kwargs['bvecs_train_full'] = bvecs_full
        if use_spot:
            # Per-axis voxel size in normalized coords (xyz in [-1, 1] over dim).
            ds = train_dataloaders.dataset
            data_shape = ds.image.shape if hasattr(ds, 'image') else None
            if data_shape is None or len(data_shape) < 3:
                raise RuntimeError("SPOT mode needs dataset.image to infer dim")
            dim_xyz = np.asarray(data_shape[:3], dtype=np.float32)
            vs_norm = (2.0 / dim_xyz).astype(np.float32)        # [3]
            inr_lighning_kwargs['ball_stick_spot_voxel_size_norm'] = torch.from_numpy(vs_norm)
            print(f"SPOT mode: voxel_size_norm={vs_norm.tolist()}, K={spot_K}")
        if use_fodf:
            Y_np = _bs_sh_basis(bvecs_full, l_max)         # [V, n_sh]
            bvals_np = bvals_full.cpu().numpy()
            uniq, b_idx = np.unique(np.round(bvals_np), return_inverse=True)
            quad = _bs_make_quad(l_max)
            l_idx = _bs_sh_l_idx(l_max)
            inr_lighning_kwargs['ball_stick_fodf_Y'] = torch.from_numpy(Y_np)
            inr_lighning_kwargs['ball_stick_fodf_b_idx_per_vol'] = torch.from_numpy(b_idx)
            inr_lighning_kwargs['ball_stick_fodf_unique_bvals'] = torch.from_numpy(uniq.astype(np.float32))
            inr_lighning_kwargs['ball_stick_fodf_Pl_w'] = torch.from_numpy(quad['Pl_w'])
            inr_lighning_kwargs['ball_stick_fodf_t'] = torch.from_numpy(quad['t'])
            inr_lighning_kwargs['ball_stick_fodf_l_idx_per_coef'] = torch.from_numpy(l_idx)
            # Dense sphere grid for non-negativity penalty.
            from inrdiff.ball_stick import fibonacci_sphere_grid as _bs_sphere
            nonneg_w = float(config.get('ball_stick_fodf_nonneg_weight', 0.0))
            inr_lighning_kwargs['ball_stick_fodf_nonneg_weight'] = nonneg_w
            inr_lighning_kwargs['ball_stick_fodf_laplace_weight'] = float(
                config.get('ball_stick_fodf_laplace_weight', 0.0))
            unpin_c00 = bool(config.get('ball_stick_fodf_unpin_c00', False))
            inr_lighning_kwargs['ball_stick_fodf_unpin_c00'] = unpin_c00
            inr_lighning_kwargs['ball_stick_fodf_b0_anchor_weight'] = float(
                config.get('ball_stick_fodf_b0_anchor_weight', 0.0))
            if nonneg_w > 0:
                n_sphere = int(config.get('ball_stick_fodf_nonneg_n_grid', 200))
                grid_np = _bs_sphere(n_sphere)
                Y_dense_np = _bs_sh_basis(grid_np, l_max)
                inr_lighning_kwargs['ball_stick_fodf_Y_dense'] = torch.from_numpy(Y_dense_np)
            expected_outs = _bs_n_raw_params_fodf(l_max, unpin_c00=unpin_c00)
            print(f"model_output_type=ball_stick (FODF, l_max={l_max}, "
                  f"unpin_c00={unpin_c00}): n_sh={Y_np.shape[1]}, "
                  f"n_unique_b={len(uniq)}, raw params per voxel={expected_outs}")
        else:
            expected_outs = _bs_n_raw_params(n_sticks)
            print(f"model_output_type=ball_stick (sticks): n_sticks={n_sticks}, "
                  f"raw params per voxel={expected_outs}")
        print(f"  bvals_train_full shape: {tuple(bvals_full.shape)}, "
              f"bvecs shape: {tuple(bvecs_full.shape)}")

    # Option 2 — per-voxel S0 output scaling. Pass the RAW (un-normalized) b0
    # volume; the model will divide by `b0_output_scale_anchor` (brain-mean
    # S0 by default) inside forward to get the per-voxel ratio. Stored
    # buffer is in raw units so the eval path doesn't depend on the train-
    # shell normalization scale.
    if config.get('use_b0_output_scale', False) and shore_output:
        ds = train_dataloaders.dataset
        b0_full_raw = getattr(ds, '_b0_image_raw', None)
        if b0_full_raw is None and b0_image is not None:
            b0_full_raw = b0_image
        if b0_full_raw is not None:
            b0_raw_3d = b0_full_raw.float().cpu()
            try:
                mask_3d = ds.mask.cpu().bool()
                anchor = float(b0_raw_3d[mask_3d].mean())
            except Exception:
                anchor = float(b0_raw_3d.mean())
            inr_lighning_kwargs['b0_output_scale_volume'] = b0_raw_3d
            inr_lighning_kwargs['b0_output_scale_anchor'] = anchor
            print(f"use_b0_output_scale: anchor={anchor:.2f}, b0 range "
                  f"[{b0_raw_3d.min():.0f}, {b0_raw_3d.max():.0f}]")
        else:
            print("use_b0_output_scale requested but no b0 image available; skipping")

    if not config.get('val_skip', False):
        inr_lighning_kwargs['val_slice_func'] = val_slice_dataset._fold_slice,

    # Create model or load from checkpoint
    if 'checkpoint_path' in config and config['checkpoint_path']:
        print(f"Loading model from checkpoint: {config['checkpoint_path']}")

        # Load model from fixed checkpoint - don't pass pos_encoder to use the one from checkpoint
        pl_module = INRLightningModule.load_from_checkpoint(
            config['checkpoint_path'],
            **inr_lighning_kwargs
        )
    else:
        pl_module = INRLightningModule(
            pos_encoder=pos_encoder,  # Only pass pos_encoder for new models
            **inr_lighning_kwargs
            )

    # Wire up B0 batches for shore_b0_loss mode
    if shore_b0_loss and hasattr(train_dataloaders.dataset, 'b0_batches') and train_dataloaders.dataset.b0_batches is not None:
        pl_module._b0_batches = train_dataloaders.dataset.b0_batches

    # Wire up shell indices for per-shell normalized loss
    if config.get('shore_shell_norm', False) and shore_output:
        bvals_raw = train_dataloaders.dataset.bvals.float().cpu()
        shell_labels = torch.round(bvals_raw / 1000) * 1000
        unique_shells = torch.unique(shell_labels).sort()[0]
        unique_shells = unique_shells[unique_shells > 50]
        shell_indices = [torch.where(shell_labels == s)[0] for s in unique_shells]
        pl_module._shell_indices = shell_indices

        shell_weights_cfg = config.get('shore_shell_weights', None)
        if isinstance(shell_weights_cfg, dict):
            weight_values = []
            for shell in unique_shells.tolist():
                shell_key_int = int(shell)
                shell_key_str = str(shell_key_int)
                w = shell_weights_cfg.get(shell_key_int, shell_weights_cfg.get(shell_key_str, 1.0))
                weight_values.append(float(w))
            pl_module._shell_weights = torch.tensor(weight_values, dtype=torch.float32)
            print(f"Using SHORE shell weights: {dict(zip([int(s) for s in unique_shells.tolist()], weight_values))}")

    # Wire up shore_pinv for coefficient-space loss
    if config.get('shore_coeff_weight', 0.0) > 0 and shore_output and hasattr(train_dataset, 'shore_pinv'):
        pl_module._shore_pinv = train_dataset.shore_pinv.to('cuda')

    # Wire up q=0 SHORE basis row + per-voxel B0 anchor for b0_anchor_weight loss.
    # Mechanism mirrors `src/eval_shore_holdout_ceiling.py`: at each step we
    # evaluate the model's predicted SHORE coefficients at q=0 (using the
    # column-normalized SHORE basis row) and pull the prediction toward the
    # per-voxel B0 image. This converts the basis-bound regime (held-out
    # ceiling 3.51) into the B0-anchored regime (ceiling 14.62 train-scale
    # / 30.91 joint-scale) on the locked ParkMR shell-holdout benchmark.
    b0_anchor_weight = float(config.get('b0_anchor_weight', 0.0))
    if b0_anchor_weight > 0 and shore_output and hasattr(train_dataset, 'shore_col_norms'):
        from dipy.core.gradients import gradient_table
        from dipy.reconst.shore import shore_matrix as _dipy_shore_matrix
        radial_order = int(config.get('shore_radial_order', 6))
        zeta = float(config.get('shore_zeta', 700))
        gtab_q0 = gradient_table(
            bvals=np.array([0.0]),
            bvecs=np.array([[1.0, 0.0, 0.0]]),
        )
        M_q0 = _dipy_shore_matrix(radial_order=radial_order, zeta=zeta, gtab=gtab_q0)
        col_norms = train_dataset.shore_col_norms.cpu().numpy()
        M_q0_norm = (M_q0 / col_norms).squeeze().astype('float32')  # [n_shore]
        pl_module._shore_q0_row = torch.from_numpy(M_q0_norm).float().to('cuda')
        if (
            hasattr(train_dataloaders.dataset, 'b0_batches')
            and train_dataloaders.dataset.b0_batches is not None
        ):
            pl_module._b0_batches = train_dataloaders.dataset.b0_batches
        else:
            raise RuntimeError(
                "b0_anchor_weight > 0 but train dataset has no b0_batches. "
                "Check that B0 volumes exist and use_b0_prior=False."
            )
        print(
            f"B0 anchor loss enabled: weight={b0_anchor_weight}, "
            f"q=0 SHORE row shape={tuple(pl_module._shore_q0_row.shape)}, "
            f"b0 anchor target range [{pl_module._b0_batches.min():.4f}, "
            f"{pl_module._b0_batches.max():.4f}]"
        )

    # Wire per-voxel mono-exp targets (Idea 5b) for q_monoexp loss.
    if (
        config.get('q_monoexp_weight', 0.0) > 0
        and hasattr(train_dataloaders.dataset, 'q_monoexp_S0_batches')
    ):
        pl_module._q_monoexp_S0_batches = (
            train_dataloaders.dataset.q_monoexp_S0_batches.to('cuda')
        )
        pl_module._q_monoexp_D_batches = (
            train_dataloaders.dataset.q_monoexp_D_batches.to('cuda')
        )
        print(
            f"q_monoexp loss enabled: weight={config.get('q_monoexp_weight', 0.0)}, "
            f"per-voxel targets fitted from training shells"
        )

    # Q-space auxiliary regularizers: smoothness (penalize 2nd derivative of
    # predicted signal w.r.t. q) and monotonic-decay (penalize signal
    # increasing as b grows along any direction). Both are INR-only mechanisms
    # — the per-voxel pinv minimum-norm solution does not enforce either.
    # Targets the failure mode where pinv at r=8 produces wildly oscillating
    # extrapolations to held-out shells.
    q_smooth_weight = float(config.get('q_smooth_weight', 0.0))
    q_mono_weight = float(config.get('q_mono_weight', 0.0))
    q_st_weight = float(config.get('q_st_weight', 0.0))
    q_convex_weight = float(config.get('q_convex_weight', 0.0))
    q_monoexp_weight = float(config.get('q_monoexp_weight', 0.0))
    b_inf_anchor_weight = float(config.get('b_inf_anchor_weight', 0.0))
    # q_st, q_convex, q_monoexp reuse q_smooth's M_center sample points;
    # activate sampling if any of these priors is on.
    needs_q_smooth_sampling = (
        q_smooth_weight > 0 or q_st_weight > 0 or q_convex_weight > 0
        or q_monoexp_weight > 0
    )
    needs_q_aux_block = needs_q_smooth_sampling or q_mono_weight > 0 or b_inf_anchor_weight > 0
    if needs_q_aux_block and shore_output \
            and hasattr(train_dataset, 'shore_col_norms'):
        from dipy.core.gradients import gradient_table
        from dipy.reconst.shore import shore_matrix as _dipy_shore_matrix
        radial_order = int(config.get('shore_radial_order', 6))
        zeta = float(config.get('shore_zeta', 700))
        col_norms_q = train_dataset.shore_col_norms.cpu().numpy()
        bvals_train_arr = train_dataset.bvals.cpu().numpy().astype(np.float64)
        max_b = float(np.max(bvals_train_arr))
        rng_q = np.random.default_rng(int(config.get('q_aux_seed', 1234)))

        def _build_M_normed(bvals_arr, bvecs_arr):
            gtab = gradient_table(bvals=bvals_arr, bvecs=bvecs_arr)
            M = _dipy_shore_matrix(radial_order=radial_order, zeta=zeta, gtab=gtab)
            return (M / col_norms_q).astype('float32')

        if needs_q_smooth_sampling:
            n_smooth = int(config.get('q_smooth_n', 64))
            delta_b = float(config.get('q_smooth_delta_b', 100.0))
            b_min = float(config.get('q_smooth_b_min', 50.0))
            b_max = float(config.get('q_smooth_b_max', max_b))
            b_centers = rng_q.uniform(b_min, b_max, size=n_smooth)
            dirs_smooth = rng_q.normal(size=(n_smooth, 3))
            dirs_smooth /= np.linalg.norm(dirs_smooth, axis=1, keepdims=True)
            b_minus = np.maximum(b_centers - delta_b, 0.0)
            b_plus = b_centers + delta_b
            pl_module._q_smooth_M_minus = torch.from_numpy(
                _build_M_normed(b_minus, dirs_smooth)).float().to('cuda')
            pl_module._q_smooth_M_center = torch.from_numpy(
                _build_M_normed(b_centers, dirs_smooth)).float().to('cuda')
            pl_module._q_smooth_M_plus = torch.from_numpy(
                _build_M_normed(b_plus, dirs_smooth)).float().to('cuda')
            # b_centers are needed by q_monoexp loss for per-voxel exp(-b·D).
            pl_module._q_smooth_b_centers = torch.from_numpy(b_centers).float().to('cuda')
            print(
                f"q-space sampling enabled (q_smooth/q_st/q_convex/q_monoexp share basis): "
                f"q_smooth_weight={q_smooth_weight}, q_st_weight={q_st_weight}, "
                f"q_convex_weight={q_convex_weight}, q_monoexp_weight={q_monoexp_weight}, "
                f"n={n_smooth}, delta_b={delta_b}, "
                f"b range [{b_centers.min():.0f}, {b_centers.max():.0f}]"
            )

        if q_mono_weight > 0:
            n_mono = int(config.get('q_mono_n', 64))
            delta_mono = float(config.get('q_mono_delta_b', 100.0))
            b_low = rng_q.uniform(50.0, max_b - delta_mono, size=n_mono)
            b_high = b_low + delta_mono
            dirs_mono = rng_q.normal(size=(n_mono, 3))
            dirs_mono /= np.linalg.norm(dirs_mono, axis=1, keepdims=True)
            pl_module._q_mono_M_low = torch.from_numpy(
                _build_M_normed(b_low, dirs_mono)).float().to('cuda')
            pl_module._q_mono_M_high = torch.from_numpy(
                _build_M_normed(b_high, dirs_mono)).float().to('cuda')
            print(
                f"q_mono loss enabled: weight={q_mono_weight}, "
                f"n={n_mono}, delta_b={delta_mono}"
            )

        if b_inf_anchor_weight > 0:
            n_inf = int(config.get('b_inf_anchor_n', 64))
            b_inf_value = float(config.get('b_inf_anchor_b', 5000.0))
            b_inf_arr = np.full(n_inf, b_inf_value, dtype=np.float64)
            dirs_inf = rng_q.normal(size=(n_inf, 3))
            dirs_inf /= np.linalg.norm(dirs_inf, axis=1, keepdims=True)
            pl_module._b_inf_M = torch.from_numpy(
                _build_M_normed(b_inf_arr, dirs_inf)).float().to('cuda')
            print(
                f"b_inf anchor loss enabled: weight={b_inf_anchor_weight}, "
                f"n={n_inf}, b_inf={b_inf_value}"
            )

    # Optional output-layer LR multiplier (separate param group).
    output_lr_mult = config.get('output_lr_mult', None)
    if output_lr_mult is not None:
        pl_module._output_lr_mult = float(output_lr_mult)
        print(f"Output-layer LR multiplier: {pl_module._output_lr_mult}x")

    cos_eta_min_ratio = config.get('cos_eta_min_ratio', None)
    if cos_eta_min_ratio is not None:
        pl_module.cos_eta_min_ratio = float(cos_eta_min_ratio)
        print(f"Cos scheduler eta_min_ratio: {pl_module.cos_eta_min_ratio}")

    # Optional warm-start from a cross-subject pretrained state dict.
    # The file is a bare state_dict of an INRLightningModule trained on a
    # non-benchmark subject. We strip subject-specific buffers (SHORE/SH
    # design matrices, INCODE B0 context vector) so the CURRENT subject's
    # values are preserved, then load the rest non-strictly.
    pretrained_path = config.get('pretrained_weights_path', None)
    if pretrained_path:
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"pretrained_weights_path not found: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        skip_prefixes = ('shore_matrix', 'y_matrix', 'network.context_vec')
        filtered = {k: v for k, v in state_dict.items() if not k.startswith(skip_prefixes)}
        missing, unexpected = pl_module.load_state_dict(filtered, strict=False)
        n_loaded = len(filtered)
        print(
            f"Warm-start: loaded {n_loaded} tensors from {pretrained_path} "
            f"(skipped {len(state_dict) - n_loaded} subject-specific; "
            f"missing={len(missing)}, unexpected={len(unexpected)})"
        )

    disable_checkpointing = config.get('disable_checkpointing', False)
    callbacks = []
    if not disable_checkpointing:
        callbacks.append(get_checkpoint_callback(config))

    devices = torch.cuda.device_count()
    if devices < 1:
        devices = 1

    trainer_precision = config.get('trainer_precision', '16-mixed')
    trainer_args = {
        'precision': trainer_precision,
        'max_epochs': config['epochs'],
        'logger': logger,
        'num_sanity_val_steps': config['val_sanity_steps'],
        'accelerator': 'gpu',
        'devices': devices,
    }
    # Profiler adds per-hook wrapping overhead across thousands of steps; only
    # enable it when explicitly requested for diagnostics.
    if config.get('profiler', None):
        trainer_args['profiler'] = config['profiler']
    trainer_gradient_clip_val = config.get('trainer_gradient_clip_val', 0.0)
    if trainer_gradient_clip_val > 0:
        trainer_args['gradient_clip_val'] = trainer_gradient_clip_val
    if devices > 1:
        trainer_args['strategy'] = 'ddp'
    if callbacks:
        trainer_args['callbacks'] = callbacks
    if disable_checkpointing:
        trainer_args['enable_checkpointing'] = False


    if config['val_mode'] == 'step':
        trainer_args['val_check_interval'] = config['val_check_interval']
    elif config['val_mode'] == 'epoch':
        trainer_args['check_val_every_n_epoch'] = config['val_check_interval']
    else:
        raise ValueError("Invalid validation mode. Choose 'step' or 'epoch'.")

    # Optionally compile the INR backbone (the dominant forward/backward cost).
    # Compiles only `self.network` to avoid graph breaks from grid_sample / the
    # analytic ball-stick forward. Falls back gracefully if compile is unavailable.
    if config.get('torch_compile', False):
        try:
            # Persist the Inductor compile artifacts across runs so the (~300s)
            # warmup is paid once. torch 2.4 ships fx_graph_cache=False; enable it
            # and point it at a persistent dir so same-config reruns load the
            # compiled graphs instead of recompiling.
            import os as _os
            cache_dir = config.get(
                'torchinductor_cache_dir',
                _os.path.join(_os.path.dirname(__file__), '..', '.torchinductor_cache'))
            _os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', _os.path.abspath(cache_dir))
            try:
                import torch._inductor.config as _ic
                _ic.fx_graph_cache = True
            except Exception:
                pass
            mode = config.get('torch_compile_mode', 'default')
            pl_module.network = torch.compile(pl_module.network, mode=mode)
            print(f"[torch.compile] network compiled (mode={mode}); "
                  f"fx_graph_cache=on dir={_os.environ['TORCHINDUCTOR_CACHE_DIR']}")
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[torch.compile] failed, running eager: {exc}")

    trainer = pl.Trainer(**trainer_args)
    print(f"[timing] setup→fit (pre-fit total): {_time.perf_counter() - _t_setup0:.1f}s")
    _t_fit0 = _time.perf_counter()
    trainer.fit(pl_module)
    print(f"[timing] trainer.fit: {_time.perf_counter() - _t_fit0:.1f}s")

    # Optional: export a compact state_dict for cross-subject warm-start.
    save_state_path = config.get('save_state_dict_path', None)
    if save_state_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_state_path)), exist_ok=True)
        torch.save(pl_module.state_dict(), save_state_path)
        print(f"Saved pretrain state_dict -> {save_state_path}")

    # Optional stage 2: train a second INR (SH residual head) on top of
    # the frozen stage-1 model. See src/stage2_residual.py.
    if bool(config.get('stage2_enabled', False)):
        from inrdiff.stage2_residual import train_stage2
        print("\n[stage2] training SH residual INR on frozen stage-1 ball-fODF")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inr2 = train_stage2(
            model_1=pl_module.to(device),
            train_dataloader=train_dataloaders,
            train_dataset=train_dataloaders.dataset,
            cfg=config,
            device=device,
        )
        pl_module._stage2_inr = inr2

    return pl_module, config, trainer


def main():
    parser = argparse.ArgumentParser(description='Train an INR model for DWI representation')
    parser.add_argument('config', type=str, help='Path to the YAML configuration file')
    parser.add_argument('--subject', type=str, default=None,
                      help='List of subject IDs to process. If not provided, processes all subjects.')
    
    args = parser.parse_args()
    config = load_config(args.config, args.subject)
    train_with_config(config)

if __name__ == "__main__":
    main()
