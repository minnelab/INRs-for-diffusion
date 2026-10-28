"""Bessel-Fourier basis for q-space DWI representation.

Alternative to SHORE. Uses spherical Bessel functions for the radial part and
real spherical harmonics for the angular part:

    Phi_{nlm}(q, theta, phi) = j_l(lambda_{nl} * q / q_max) * Y_l^m(theta, phi)

where lambda_{nl} is the n-th positive zero of j_l (Dirichlet boundary
condition on the q-ball of radius q_max), and Y_l^m are real spherical
harmonics (e3nn `integral` normalization). Only even l are used for
antipodal symmetry of the diffusion signal.

The raw design matrix is column-normalized downstream (same policy as SHORE),
so the absolute radial normalization constant does not affect the fit.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import spherical_jn
from e3nn import o3

from inrdiff.networks.utils import xyz_to_angles


def spherical_jn_zeros(l: int, n_zeros: int) -> np.ndarray:
    """First `n_zeros` positive roots of the spherical Bessel function j_l(x)."""
    if n_zeros <= 0:
        return np.empty(0, dtype=np.float64)
    if l == 0:
        # j_0(x) = sin(x)/x, zeros at x = k*pi, k=1,2,...
        return np.arange(1, n_zeros + 1, dtype=np.float64) * np.pi

    zeros: list[float] = []
    x = 1e-3
    step = 0.05
    y_prev = float(spherical_jn(l, x))
    max_x = 5.0 + 4.0 * (l + n_zeros) * np.pi
    while len(zeros) < n_zeros and x < max_x:
        x_next = x + step
        y_next = float(spherical_jn(l, x_next))
        if y_prev == 0.0 and x > 1e-2:
            zeros.append(x)
        elif y_prev * y_next < 0.0:
            root = brentq(lambda t: float(spherical_jn(l, float(t))),
                          x, x_next, xtol=1e-10, rtol=1e-10)
            zeros.append(float(root))
        x, y_prev = x_next, y_next
    if len(zeros) < n_zeros:
        raise RuntimeError(
            f"Could not locate {n_zeros} zeros of spherical j_{l} within x<{max_x}"
        )
    return np.asarray(zeros[:n_zeros], dtype=np.float64)


def bvals_to_q(bvals: np.ndarray) -> np.ndarray:
    """Convert b-values (s/mm^2) to q-magnitudes via the Stejskal-Tanner relation
    q = sqrt(b / (4*pi^2 * tau)) with tau=1/(4*pi^2), i.e. q = sqrt(b).

    Absolute units do not matter because q_max scales them; the relative
    spacing between shells is what we care about.
    """
    b = np.asarray(bvals, dtype=np.float64)
    b = np.clip(b, 0.0, None)
    return np.sqrt(b)


def bessel_fourier_matrix(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    radial_order: int,
    lmax: int,
    q_max: float | None = None,
):
    """Build a Bessel-Fourier design matrix.

    Args:
        bvals: (n_volumes,) b-values.
        bvecs: (n_volumes, 3) unit gradient directions.
        radial_order: number of radial Bessel modes per angular order l.
        lmax: maximum spherical harmonic degree (must be even, >= 0).
        q_max: q-ceiling used to rescale q into the ball [0, 1]. Default:
            max(q) * 1.01, chosen so every measurement strictly sits inside
            the basis support.

    Returns:
        M: (n_volumes, n_coeffs) design matrix.
        coeff_info: list of (n, l, m) tuples aligned with the columns of M.
        q_max_used: the q_max actually used (for logging).
    """
    if lmax % 2 != 0 or lmax < 0:
        raise ValueError(f"lmax must be a non-negative even integer, got {lmax}")
    if radial_order <= 0:
        raise ValueError(f"radial_order must be positive, got {radial_order}")

    bvals_np = np.asarray(bvals, dtype=np.float64)
    q = bvals_to_q(bvals_np)
    if q_max is None:
        q_max = float(q.max() * 1.01) if q.size else 1.0
    q_max = float(max(q_max, 1e-6))

    # Angular: real spherical harmonics via e3nn, concatenated across even l.
    bvecs_t = torch.as_tensor(np.asarray(bvecs, dtype=np.float64), dtype=torch.float32)
    alpha, beta = xyz_to_angles(bvecs_t)
    l_values = list(range(0, lmax + 1, 2))
    Y = o3.spherical_harmonics_alpha_beta(
        l_values, alpha, beta, normalization="integral"
    ).cpu().numpy()  # (n_vol, sum(2l+1))

    q_scaled = q / q_max  # (n_vol,)

    columns = []
    coeff_info: list[tuple[int, int, int]] = []
    sh_offset = 0
    for l in l_values:
        n_m = 2 * l + 1
        Y_l = Y[:, sh_offset:sh_offset + n_m]  # (n_vol, 2l+1)
        sh_offset += n_m

        zeros_l = spherical_jn_zeros(l, radial_order)
        # Radial normalization on the unit ball: N_{nl} = sqrt(2) / |j_{l+1}(lambda_{nl})|.
        # Column-normalization downstream will re-balance, but this keeps the
        # raw conditioning reasonable.
        j_lp1 = spherical_jn(l + 1, zeros_l)
        norm_nl = np.sqrt(2.0) / np.clip(np.abs(j_lp1), 1e-10, None)

        for n_idx, (lam, N_nl) in enumerate(zip(zeros_l, norm_nl), start=1):
            R_nl = N_nl * spherical_jn(l, lam * q_scaled)  # (n_vol,)
            for m_offset in range(n_m):
                columns.append(R_nl * Y_l[:, m_offset])
                coeff_info.append((n_idx, l, m_offset - l))

    M = np.stack(columns, axis=1).astype(np.float64)  # (n_vol, n_coeffs)
    return M, coeff_info, q_max
