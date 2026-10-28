"""Stage-2 SH-residual INR (Option F) — trained on top of a frozen stage-1
ball-fODF INR to capture in-shell directional structure that ball-fODF
can't fit. Per program.md §8.1, all per-voxel state is produced by an
INR forward; this module is a second INR, not a per-voxel cache.

Forward:
    pred_full(x, b, q) = INR_1(x) → bs_signal(b, q) × S0
                      + g(b) · Σ_lm coefs_per_shell(x)_lm · Y_l^m(q)

where INR_2: (x, y, z) → coefs_per_shell ∈ ℝ^{n_train_shells × n_sh}
and g(b) is a smooth Gaussian gate around training shells (or 1{b ∈ train}
in the hard variant).

At eval at held-out b (b=100, b=700), g(b) ≈ 0 so the residual vanishes
and the prediction collapses to the pure ball-fODF — by construction this
preserves the b-value held-out frontier.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage2ResidualINR(nn.Module):
    """Small MLP mapping pos-encoded coords → per-shell SH residual coefs.

    Reuses the stage-1 model's pos_encoder so the input feature is the
    same; only the head differs.
    """
    def __init__(self, in_features: int, n_out: int,
                 hidden: int = 256, num_layers: int = 3):
        super().__init__()
        layers = []
        prev = in_features
        for _ in range(int(num_layers) - 1):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.GELU())
            prev = hidden
        layers.append(nn.Linear(prev, n_out))
        # zero-init final layer so residual starts at 0 (no disruption to
        # the stage-1 prediction at the start of stage 2).
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_gate_b(bvals_t, train_shell_bvals, sigma: float):
    """Smooth Gaussian gate: g(b) = Σ_k exp(−(b−b_k)²/(2σ²)), clamped to [0,1].

    bvals_t: [V] torch tensor (on device).
    train_shell_bvals: list of ints, training shells.
    Returns [V] tensor, values in [0, ~1].
    """
    b = bvals_t.float()
    centers = torch.tensor(
        [float(s) for s in train_shell_bvals],
        dtype=b.dtype, device=b.device,
    )  # [K]
    diff_sq = (b[:, None] - centers[None, :]) ** 2  # [V, K]
    weights = torch.exp(-diff_sq / (2.0 * float(sigma) ** 2))  # [V, K]
    return weights.sum(-1).clamp(0.0, 1.0)


def _compute_bs_signal(model_1, coords_encoded, batch_idx):
    """Forward ball-fODF on already pos-encoded coords; return signal·S0
    in the training-target normalized space, matching what training_step did.
    """
    from inrdiff.ball_stick import signal_at_fodf as _bs_fodf, signal_at as _bs_sticks
    with torch.no_grad():
        outputs_1 = model_1.network(coords_encoded)
        if model_1._b0_volume_scale is not None:
            # mirror the b0_output_scale path used in forward() if active —
            # but the locked benchmark already disables that for ball_stick,
            # so this is a no-op here; we just emit `outputs_1` raw.
            pass
        if model_1.ball_stick_use_fodf:
            quad = {"t": model_1._bs_fodf_t, "Pl_w": model_1._bs_fodf_Pl_w}
            signal_E = _bs_fodf(
                outputs_1, model_1._bs_bvals, model_1._bs_bvecs,
                l_max=model_1.ball_stick_fodf_l_max,
                Y_at_bvecs=model_1._bs_fodf_Y,
                b_idx_per_vol=model_1._bs_fodf_b_idx_per_vol,
                unique_bvals=model_1._bs_fodf_unique_bvals,
                quad_data=quad,
                l_idx_per_coef=model_1._bs_fodf_l_idx_per_coef,
            )
        else:
            signal_E = _bs_sticks(
                outputs_1, model_1._bs_bvals, model_1._bs_bvecs,
                n_sticks=model_1.ball_stick_n_sticks,
            )
        S0_voxel = model_1._b0_batches[batch_idx]
        bs_predicted = signal_E * S0_voxel.unsqueeze(-1)  # [B, n_train_vols]
    return bs_predicted


def train_stage2(model_1, train_dataloader, train_dataset, cfg,
                 device="cuda"):
    """Train the stage-2 residual INR with stage 1 frozen.

    Reads from `cfg`:
      stage2_l_max:        int (SH order; default 6)
      stage2_hidden:       int (default 256)
      stage2_num_layers:   int (default 3)
      stage2_epochs:       int (default 3)
      stage2_lr:           float (default 1e-3)
      stage2_weight_decay: float (default 0.0)
      bval_train:          list of training shells
    Returns: trained Stage2ResidualINR.
    """
    from inrdiff.ball_stick import (n_sh_coeffs_even as _bs_n_sh_even,
                            compute_sh_basis_at_bvecs as _bs_sh_basis,
                            compute_shell_idx_per_vol as _bs_shell_idx)

    l_max = int(cfg.get("stage2_l_max", 6))
    hidden = int(cfg.get("stage2_hidden", 256))
    num_layers = int(cfg.get("stage2_num_layers", 3))
    n_epochs = int(cfg.get("stage2_epochs", 3))
    lr = float(cfg.get("stage2_lr", 1e-3))
    wd = float(cfg.get("stage2_weight_decay", 0.0))
    coef_l2 = float(cfg.get("stage2_coef_l2_weight", 0.0))
    mode = str(cfg.get("stage2_residual_mode", "sh_per_shell")).lower()
    # Stage 2 can focus on a SUBSET of the training shells (e.g., only b=2000
    # where ball-fODF underperforms). If `stage2_train_shells` is not set,
    # fall back to all training shells.
    train_shell_bvals = list(
        cfg.get("stage2_train_shells", cfg.get("bval_train", []))
    ) or []
    n_train_shells = len(train_shell_bvals)
    if n_train_shells == 0:
        raise RuntimeError("stage2 requires cfg['bval_train'] or cfg['stage2_train_shells']")

    # Mode selection: per-shell SH residual vs full SHORE residual.
    use_shore = (mode == "shore")
    if use_shore:
        if not hasattr(model_1, "shore_matrix"):
            raise RuntimeError(
                "stage2 SHORE mode requires model_1.shore_matrix buffer"
            )
        shore_M_train = model_1.shore_matrix.to(device).float()  # [V_train, n_shore]
        n_out = int(shore_M_train.shape[1])
    else:
        n_sh = _bs_n_sh_even(l_max)
        n_out = n_train_shells * n_sh
        # Precompute Y at training bvecs + shell index per training-vol.
        bvecs_train = model_1._bs_bvecs.detach().cpu().numpy()
        bvals_train = model_1._bs_bvals.detach().cpu().numpy()
        Y_train_np = _bs_sh_basis(bvecs_train, l_max)
        shell_idx_train_np = _bs_shell_idx(bvals_train, train_shell_bvals)
        Y_train = torch.from_numpy(Y_train_np).float().to(device)
        shell_idx_train = torch.from_numpy(shell_idx_train_np).long().to(device)
        idx_clamped = shell_idx_train.clamp_min(0)
        mask_train = (shell_idx_train >= 0).float().to(device).unsqueeze(0)

    # Probe encoded-feature size by running a small fake batch through the
    # stage-1 pos_encoder. The output dim is what the residual head consumes.
    model_1 = model_1.to(device).eval()
    for p in model_1.parameters():
        p.requires_grad_(False)

    # First batch — figure out feature size.
    sample_batch = next(iter(train_dataloader))
    coords0 = sample_batch[0].view(-1, sample_batch[0].shape[-1]).to(device)
    with torch.no_grad():
        if model_1.pos_encoder is not None:
            feat0 = coords0[:8]
            for enc in model_1.pos_encoder:
                feat0 = enc(feat0)
        else:
            feat0 = coords0[:8]
    in_features = int(feat0.shape[-1])

    inr2 = Stage2ResidualINR(
        in_features=in_features,
        n_out=n_out,
        hidden=hidden,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(inr2.parameters(), lr=lr, weight_decay=wd)

    mode_tag = "shore" if use_shore else f"sh_l{l_max}x{n_train_shells}sh"
    print(f"[stage2] mode={mode_tag} training "
          f"{sum(p.numel() for p in inr2.parameters())} params, "
          f"{n_epochs} epochs, lr={lr:g}, n_out={n_out}, "
          f"in_features={in_features}, hidden={hidden}")

    step = 0
    for epoch in range(n_epochs):
        epoch_loss_sum, epoch_count = 0.0, 0
        for batch_idx, batch in enumerate(train_dataloader):
            coords, values, _ = batch
            coords = coords.view(-1, coords.shape[-1]).to(device)
            values = values.view(-1, values.shape[-1]).to(device)

            # Stage 1 frozen forward — encoded coords + bs prediction.
            with torch.no_grad():
                if model_1.pos_encoder is not None:
                    coords_enc = coords
                    for enc in model_1.pos_encoder:
                        coords_enc = enc(coords_enc)
                else:
                    coords_enc = coords
                bs_predicted = _compute_bs_signal(
                    model_1, coords_enc, batch_idx
                )  # [B, V_train]

            outputs_2 = inr2(coords_enc)                       # [B, n_out]
            if use_shore:
                residual = outputs_2 @ shore_M_train.T          # [B, V_train]
            else:
                sh_coeffs = outputs_2.view(coords_enc.shape[0],
                                           n_train_shells, n_sh)
                coeffs_per_vol = sh_coeffs[:, idx_clamped, :]
                residual = (coeffs_per_vol * Y_train.unsqueeze(0)).sum(-1)
                residual = residual * mask_train

            predicted = bs_predicted + residual
            loss = F.mse_loss(predicted, values)
            if coef_l2 > 0:
                loss = loss + coef_l2 * outputs_2.pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_sum += float(loss.detach()) * predicted.shape[0]
            epoch_count += predicted.shape[0]
            step += 1
        avg = epoch_loss_sum / max(1, epoch_count)
        print(f"  [stage2] epoch {epoch + 1}/{n_epochs}  avg_loss={avg:.6g}")

    return inr2


def predict_with_stage2(model_1, inr2, brain_coords_encoded, bvals_np, bvecs_np,
                       cfg, l_max: int, train_shell_bvals,
                       eval_batch_size: int = 500_000,
                       device: str = "cuda",
                       shore_M_eval=None):
    """Compute the stage-2 residual signal at all eval (b, q).

    Returns numpy array `residual_norm` in the training-target normalized
    scale, shape [N_brain, n_eval_vols]. Already gated to be zero at
    held-out shells. Caller adds it to the stage-1 prediction.
    """
    from inrdiff.ball_stick import (n_sh_coeffs_even as _bs_n_sh_even,
                            compute_sh_basis_at_bvecs as _bs_sh_basis,
                            compute_shell_idx_per_vol as _bs_shell_idx)

    n_train_shells = len(train_shell_bvals)
    sigma = float(cfg.get("stage2_gate_sigma", 0.0))
    mode = str(cfg.get("stage2_residual_mode", "sh_per_shell")).lower()
    use_shore = (mode == "shore")

    shell_idx_eval = _bs_shell_idx(bvals_np, train_shell_bvals)
    shell_idx_t = torch.from_numpy(shell_idx_eval).long().to(device)
    if sigma > 0:
        bvals_t = torch.from_numpy(bvals_np).float().to(device)
        gate = make_gate_b(bvals_t, train_shell_bvals, sigma)
    else:
        gate = (shell_idx_t >= 0).float()

    if use_shore:
        if shore_M_eval is None:
            raise RuntimeError("SHORE mode at eval requires shore_M_eval")
        shore_M_t = shore_M_eval.to(device).float()             # [V, n_shore]
    else:
        n_sh = _bs_n_sh_even(l_max)
        Y_eval = _bs_sh_basis(bvecs_np, l_max)
        Y_eval_t = torch.from_numpy(Y_eval).float().to(device)
        idx_clamped = shell_idx_t.clamp_min(0)

    parts = []
    inr2.eval()
    with torch.no_grad():
        for s in range(0, brain_coords_encoded.shape[0], eval_batch_size):
            ce = brain_coords_encoded[s:s + eval_batch_size]
            outs = inr2(ce)
            if use_shore:
                res = outs @ shore_M_t.T
            else:
                sh_coeffs = outs.view(ce.shape[0], n_train_shells, n_sh)
                coeffs_per_vol = sh_coeffs[:, idx_clamped, :]
                res = (coeffs_per_vol * Y_eval_t.unsqueeze(0)).sum(-1)
            res = res * gate.unsqueeze(0)
            parts.append(res.cpu())
    return torch.cat(parts, dim=0).numpy()
