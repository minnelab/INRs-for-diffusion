"""Ball-and-multi-stick diffusion signal forward model.

The model produces, per voxel, normalized diffusion attenuation
`E(b, q) = signal(b, q) / S0` from a small physics-parameterized
representation. The full raw signal is `S0(voxel) · E(b, q)` and S0
is supplied per voxel from the b0 image (via grid_sample lookup).

Per-voxel parameters (predicted by the INR):
    log_D_iso             1  scalar
    log_D_par             1  scalar
    raw_fractions       N+1  ball + N sticks (softmax → simplex)
    raw_sticks         3N    N stick directions (normalized)
Total raw params = 4N + 3.

Forward at (b, q):
    f_ball, f_sticks      = softmax(raw_fractions)
    D_iso, D_par           = softplus + scale → physically positive
    v_i                    = raw_sticks_i / ||raw_sticks_i||
    E(b, q) = f_ball · exp(−b · D_iso) +
              Σ_i f_stick_i · exp(−b · D_par · (q · v_i)^2)

This forward is differentiable and applies at ANY b — extrapolation to
held-out shells is the model evaluated at the held-out (b, q).

Naming against the manuscript: the arrays called `fODF_lm` here hold the
RESCALED coefficients the paper writes a_lm = 4pi/(2l+1) · c_lm, not the
fODF coefficients c_lm themselves. So `fODF_00` = sqrt(4pi) = 2sqrt(pi)
= 3.5449 is the paper's a_00, while the paper's c_00 = 1/(2sqrt(pi))
= 0.2821. Convert with c_lm = (2l+1)/(4pi) · a_lm before reading any
stored coefficient as a density on the sphere.

Only even orders are carried, since the fODF is taken antipodally
symmetric (see `n_sh_coeffs_even`): n_sh = (l_max+1)(l_max+2)/2.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# Typical brain diffusivities (s/mm² · 1e-3). Used as scale factors so the
# raw network outputs hover near zero at init and map to physical ranges.
D_ISO_SCALE = 3.0e-3      # ~free water
D_PAR_SCALE = 1.7e-3      # ~axonal parallel diffusivity
_EPS_DIR = 1e-6


def n_raw_params(n_sticks: int) -> int:
    """Return the number of raw INR outputs per voxel for N sticks."""
    return 4 * int(n_sticks) + 3


def n_sh_coeffs_even(l_max: int) -> int:
    """Number of real symmetric (even-order) SH coefficients up to l_max."""
    total = 0
    for l in range(0, int(l_max) + 1, 2):
        total += 2 * l + 1
    return total


def n_raw_params_fodf(l_max: int, unpin_c00: bool = False) -> int:
    """Ball-fODF mode: 3 base + (n_sh - 1) fODF coefs (default).

    With unpin_c00=False, fODF_00 is constrained to √(4π) so that
    E(b=0, q) = 1 by construction (mirrors the simplex constraint in the
    discrete ball-stick mode).

    With unpin_c00=True, fODF_00 is also a learned raw output; the b=0
    constraint must then be enforced via an auxiliary loss
    (ball_stick_fodf_b0_anchor_weight in `model.py`). One extra raw param.
    """
    base = 3 + n_sh_coeffs_even(l_max) - 1
    return base + (1 if unpin_c00 else 0)


def compute_shell_idx_per_vol(bvals, train_shell_bvals, tol: int = 100):
    """Map each volume to its training-shell index (-1 if not in training)."""
    import numpy as np
    bv = bvals.detach().cpu().numpy() if hasattr(bvals, "detach") else (
        bvals.cpu().numpy() if hasattr(bvals, "cpu") else np.asarray(bvals)
    )
    idx = np.full(bv.shape[0], -1, dtype=np.int64)
    for s_idx, s_b in enumerate(train_shell_bvals):
        in_shell = np.abs(bv - float(s_b)) <= int(tol)
        idx[in_shell] = int(s_idx)
    return idx


def extract_e1_from_fodf_coefs(fODF_lm, l_max: int, n_grid: int = 500):
    """Find peak direction of the L=l_max real-symmetric SH fODF.

    fODF_lm: torch.Tensor [V, n_sh] full coefficients (includes l=0).
    Returns numpy [V, 3] unit vectors at the argmax of fODF(v) on a
    fibonacci sphere grid.
    """
    import numpy as np
    grid = fibonacci_sphere_grid(int(n_grid))         # [N, 3]
    Y_grid = compute_sh_basis_at_bvecs(grid, int(l_max))  # [N, n_sh]
    Y_t = torch.from_numpy(Y_grid).to(fODF_lm.dtype).to(fODF_lm.device)
    f_at_grid = fODF_lm @ Y_t.T                       # [V, N]
    argmax_idx = f_at_grid.argmax(dim=-1).cpu().numpy()
    return grid[argmax_idx].astype(np.float32)


def extract_e1_from_ball_stick_outputs(raw_outputs, n_sticks: int):
    """Return per-voxel principal stick direction (highest f_stick weight)."""
    import numpy as np
    p = decode_params(raw_outputs, int(n_sticks))
    f_sticks = p["f_sticks"].detach().cpu().numpy()   # [V, N]
    v_sticks = p["v"].detach().cpu().numpy()          # [V, N, 3]
    dominant = np.argmax(f_sticks, axis=1)            # [V]
    e1 = v_sticks[np.arange(v_sticks.shape[0]), dominant]  # [V, 3]
    return e1.astype(np.float32)


def reconstruct_full_fodf_lm(raw_outputs, l_max: int, unpin_c00: bool = False):
    """Recover full SH-coefficient fODF (including fODF_00) from
    the network output produced in ball-fODF mode.

    When unpin_c00=True, the last raw channel is log_c00_scale and
    fODF_00 = softplus(log_c00_scale) · √(4π) per voxel.
    """
    p = decode_params_fodf(raw_outputs, int(l_max), unpin_c00=unpin_c00)
    return p["fODF_lm"]


def compute_sh_basis_at_bvecs(bvecs, l_max: int):
    """Real symmetric (even-order) SH basis at bvecs. Zero rows for B0."""
    import numpy as np
    from dipy.reconst.shm import real_sym_sh_basis
    bv = bvecs.detach().cpu().numpy() if hasattr(bvecs, "detach") else (
        bvecs.cpu().numpy() if hasattr(bvecs, "cpu") else np.asarray(bvecs)
    )
    bv = bv.astype(np.float32, copy=False)
    norms = np.linalg.norm(bv, axis=-1)
    safe_norms = np.maximum(norms, 1e-8)[..., None]
    bv_unit = bv / safe_norms
    theta = np.arccos(np.clip(bv_unit[:, 2], -1.0, 1.0))
    phi = np.arctan2(bv_unit[:, 1], bv_unit[:, 0])
    Y, _, _ = real_sym_sh_basis(int(l_max), theta, phi)
    Y = np.asarray(Y, dtype=np.float32)
    Y[norms < 1e-6] = 0.0
    return Y


def fibonacci_sphere_grid(n: int):
    """Quasi-uniform spherical Fibonacci grid (unit vectors on S²).

    Returns numpy array [n, 3]. Used to sample the fODF densely for
    CSD-style non-negativity penalty.
    """
    import numpy as np
    n = int(n)
    i = np.arange(n, dtype=np.float64)
    phi = (1.0 + 5 ** 0.5) / 2.0
    z = 1.0 - (2 * i + 1.0) / n           # z in (-1, 1)
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    theta = 2.0 * np.pi * i / phi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def make_legendre_quadrature(l_max: int, n_nodes: int = 32):
    """Precompute Gauss-Legendre quadrature data for R_l(x) computation.

    Returns a dict with:
      t      [K]       quadrature nodes in [-1, 1]
      w      [K]       quadrature weights
      Pl_w   [n_l, K]  (2l+1)/2 · w_k · P_l(t_k), so that
                       R_l(x) = Σ_k Pl_w[l, k] · exp(-x · t_k²)
      l_values [n_l]   the even orders (0, 2, ..., l_max)
    """
    import numpy as np
    from numpy.polynomial.legendre import leggauss
    t, w = leggauss(int(n_nodes))
    even_ls = list(range(0, int(l_max) + 1, 2))
    Pl_w = []
    for l in even_ls:
        coeffs = np.zeros(l + 1)
        coeffs[l] = 1.0
        # Evaluate P_l at nodes
        P_l_at_t = np.polynomial.legendre.legval(t, coeffs)
        Pl_w.append((2 * l + 1) / 2.0 * w * P_l_at_t)
    return {
        "t": t.astype(np.float32),
        "w": w.astype(np.float32),
        "Pl_w": np.stack(Pl_w, axis=0).astype(np.float32),  # [n_l, K]
        "l_values": np.array(even_ls, dtype=np.int64),
    }


def sh_coef_to_l_index(l_max: int):
    """For each real-sym SH coefficient index, return its l index in {0,2,...,l_max}.

    For L=6 the layout per dipy.real_sym_sh_basis is (l, m) ordered by l then m
    (m ∈ {-l,...,l}). We assign each coefficient to its position in
    [0, 2, 4, 6] (i.e., even-l index 0,1,2,3).
    """
    import numpy as np
    l_per_coef = []
    for l in range(0, int(l_max) + 1, 2):
        l_per_coef += [l] * (2 * l + 1)
    arr = np.array(l_per_coef, dtype=np.int64)
    even_ls = np.arange(0, int(l_max) + 1, 2, dtype=np.int64)
    even_to_idx = {int(l): i for i, l in enumerate(even_ls)}
    return np.array([even_to_idx[int(l)] for l in arr], dtype=np.int64)


def decode_params_fodf(raw, l_max: int, unpin_c00: bool = False):
    """Decode ball-fODF raw outputs into physical params + fODF SH coefs.

    Returns: f_ball [B], D_iso [B], D_par [B], fODF_lm [B, n_sh]
    f_fiber = 1 - f_ball implicit.

    If unpin_c00=True the layout is
        [log_D_iso, log_D_par, logit_f_ball, fODF_rest..., log_c00_scale]
    and fODF_00 = √(4π) · softplus(log_c00_scale) (positive, initially ~√(4π)·log2).
    The b=0 constraint that the pin enforces analytically must then be
    re-imposed via the b=0 anchor loss in `model.py:training_step`.
    """
    import math as _math
    expected = n_raw_params_fodf(l_max, unpin_c00=unpin_c00)
    if raw.shape[-1] != expected:
        raise ValueError(
            f"ball-fODF expects {expected} raw params for l_max={l_max} "
            f"(unpin_c00={unpin_c00}), got {raw.shape[-1]}"
        )
    n_sh = n_sh_coeffs_even(l_max)
    idx = 0
    log_D_iso = raw[:, idx]; idx += 1
    log_D_par = raw[:, idx]; idx += 1
    logit_f_ball = raw[:, idx]; idx += 1
    fODF_rest = raw[:, idx:idx + n_sh - 1]               # [B, n_sh - 1]
    idx += n_sh - 1
    sqrt_4pi = float(_math.sqrt(4.0 * _math.pi))
    if unpin_c00:
        log_c00_scale = raw[:, idx]                       # [B]
        idx += 1
        # softplus → positive; at raw=0 gives log(2)·√(4π) ≈ 0.69·3.545 = 2.46,
        # which lets the model start with c_00 ≈ 70% of the pin value.
        # The b=0 anchor loss pulls it toward √(4π) for high-WM voxels.
        fODF_00 = (F.softplus(log_c00_scale) * sqrt_4pi).unsqueeze(-1)
    else:
        fODF_00 = torch.full(
            (fODF_rest.shape[0], 1), sqrt_4pi,
            dtype=fODF_rest.dtype, device=fODF_rest.device,
        )
    fODF_lm = torch.cat([fODF_00, fODF_rest], dim=-1)   # [B, n_sh]
    D_iso = F.softplus(log_D_iso) * D_ISO_SCALE
    D_par = F.softplus(log_D_par) * D_PAR_SCALE
    f_ball = torch.sigmoid(logit_f_ball)
    return {
        "f_ball": f_ball,
        "D_iso": D_iso,
        "D_par": D_par,
        "fODF_lm": fODF_lm,
    }


def e_at_b0_fodf(raw, l_max: int, unpin_c00: bool = False):
    """Analytic E(b=0) per voxel under the ball-fODF parametrization.

    At b=0 the ball term contributes f_ball, and only the l=0 SH coef
    contributes to the aniso term: R_0(0)=1 (with our normalization),
    Y_00 = 1/(2√π). Hence
        E(b=0) = f_ball + (1 − f_ball) · c_00 / (2√π).
    With c_00 pinned to √(4π) = 2√π this collapses to 1 by construction.
    """
    import math as _math
    p = decode_params_fodf(raw, l_max, unpin_c00=unpin_c00)
    inv_2sqrtpi = 1.0 / (2.0 * _math.sqrt(_math.pi))
    return p["f_ball"] + (1.0 - p["f_ball"]) * p["fODF_lm"][:, 0] * inv_2sqrtpi


def signal_at_fodf(raw, bvals, bvecs, l_max: int,
                   Y_at_bvecs, b_idx_per_vol, unique_bvals,
                   quad_data, l_idx_per_coef, unpin_c00: bool = False):
    """Closed-form ball + smooth-fODF signal at all measurements.

    Args:
      raw:           [B, n_raw_params_fodf(l_max)]
      bvals:         [V]  (unused directly here; b_idx_per_vol does lookup)
      bvecs:         [V, 3] (unused directly; Y_at_bvecs is precomputed)
      Y_at_bvecs:    [V, n_sh]  real-sym SH basis at measurement bvecs
      b_idx_per_vol: [V] int — index into unique_bvals for each measurement
      unique_bvals:  [n_b]
      quad_data:     dict from make_legendre_quadrature
      l_idx_per_coef:[n_sh] int — maps each SH coef to its even-l index

    Returns E [B, V] in [0, ~1].
    """
    p = decode_params_fodf(raw, l_max, unpin_c00=unpin_c00)
    # Ball: f_ball * exp(-b * D_iso)
    ball = p["f_ball"][:, None] * torch.exp(
        -bvals[None, :] * p["D_iso"][:, None]
    )                                                    # [B, V]

    # R_l(x) at unique b's, per voxel:
    #   x[B, n_b] = unique_bvals[None, :] * D_par[:, None]
    #   exp_xt2[B, n_b, K] = exp(-x[:, :, None] * t_k_sq[None, None, :])
    #   R[B, n_b, n_l] = einsum('lk,bnk->bnl', Pl_w, exp_xt2)
    t = quad_data["t"]                                   # [K]
    Pl_w = quad_data["Pl_w"]                             # [n_l, K]
    t_sq = t * t                                          # [K]
    x = unique_bvals[None, :] * p["D_par"][:, None]      # [B, n_b]
    exp_xt2 = torch.exp(-x[..., None] * t_sq[None, None, :])  # [B, n_b, K]
    R = torch.einsum("lk,bnk->bnl", Pl_w, exp_xt2)       # [B, n_b, n_l]

    # Look up R per measurement and per coef-l-index:
    # R_per_meas[B, V, n_l] = R[:, b_idx_per_vol, :]
    R_per_meas = R[:, b_idx_per_vol, :]                  # [B, V, n_l]
    # Expand to [B, V, n_sh] by indexing on l_idx_per_coef:
    R_per_coef = R_per_meas[:, :, l_idx_per_coef]        # [B, V, n_sh]

    # fiber signal = (1 - f_ball) * Σ_lm fODF_lm * Y_l^m(q) * R_l
    fODF = p["fODF_lm"]                                   # [B, n_sh]
    fiber_per_meas = torch.einsum(
        "bm,vm,bvm->bv", fODF, Y_at_bvecs, R_per_coef
    )                                                     # [B, V]
    fiber = (1.0 - p["f_ball"])[:, None] * fiber_per_meas

    return ball + fiber


def decode_params(raw: torch.Tensor, n_sticks: int):
    """Decode raw network outputs [B, 4N+3] into physical parameters.

    Returns a dict with:
        f_ball:    [B]        weight on isotropic ball
        f_sticks:  [B, N]     weights on N sticks
        D_iso:     [B]        > 0
        D_par:     [B]        > 0
        v:         [B, N, 3]  unit vectors per stick

    All weights non-negative and `f_ball + sum_i f_sticks_i = 1`.
    """
    N = int(n_sticks)
    expected = n_raw_params(N)
    if raw.shape[-1] != expected:
        raise ValueError(
            f"ball-and-stick expects {expected} raw params for N={N}, "
            f"got {raw.shape[-1]}"
        )
    idx = 0
    log_D_iso = raw[:, idx]; idx += 1
    log_D_par = raw[:, idx]; idx += 1
    raw_frac = raw[:, idx:idx + (N + 1)]; idx += (N + 1)
    raw_v = raw[:, idx:idx + 3 * N].view(-1, N, 3); idx += 3 * N

    f = F.softmax(raw_frac, dim=-1)              # [B, N+1] simplex
    f_ball = f[:, 0]
    f_sticks = f[:, 1:]

    # Softplus keeps diffusivities positive; multiplied by physical scale.
    D_iso = F.softplus(log_D_iso) * D_ISO_SCALE
    D_par = F.softplus(log_D_par) * D_PAR_SCALE

    # Normalize raw stick directions to unit norm.
    v = raw_v / raw_v.norm(dim=-1, keepdim=True).clamp_min(_EPS_DIR)

    return {
        "f_ball": f_ball,
        "f_sticks": f_sticks,
        "D_iso": D_iso,
        "D_par": D_par,
        "v": v,
    }


def signal_at(raw: torch.Tensor, bvals: torch.Tensor, bvecs: torch.Tensor,
              n_sticks: int) -> torch.Tensor:
    """Compute attenuation E(b, q) per voxel at all measurement (b, q) pairs.

    Args:
        raw:    [B, 4N+3] raw INR outputs per voxel
        bvals:  [V]       b-values
        bvecs:  [V, 3]    unit-norm q-directions
        n_sticks: number of stick compartments

    Returns:
        E:      [B, V]    normalized diffusion attenuation (multiply by
                          per-voxel S0 outside this function for raw signal).
    """
    p = decode_params(raw, n_sticks)

    # Ball term: [B, V] = f_ball[:, None] · exp(-bvals[None, :] · D_iso[:, None])
    ball = p["f_ball"][:, None] * torch.exp(
        -bvals[None, :] * p["D_iso"][:, None]
    )

    # Stick term: cosine between v_i and q_v for each (B, stick, vol).
    # v: [B, N, 3], bvecs: [V, 3] → q·v: [B, N, V]
    q_dot_v = torch.einsum("bni,vi->bnv", p["v"], bvecs)
    # exp(-b · D_par · (q·v)^2): [B, N, V]
    stick_exp = torch.exp(
        -bvals[None, None, :] * p["D_par"][:, None, None] * (q_dot_v ** 2)
    )
    # Weighted sum over sticks: [B, V]
    sticks = (p["f_sticks"][:, :, None] * stick_exp).sum(dim=1)

    return ball + sticks  # [B, V] in units of S0
