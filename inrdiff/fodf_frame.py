"""Geometric axis-reflection of an fODF expressed in real SH coefficients.

The INR fODF is x-reflected relative to the MRtrix MSMT-CSD GT frame (a bvec
convention mismatch; validated by peak-direction alignment). We correct it by
reflecting the function on the sphere and refitting SH — independent of SH
basis sign conventions, so it is correct for both descoteaux07 and tournier07.
"""
import numpy as np


def reflection_operator(l_max, basis_type, axis_signs, legacy=False,
                        sphere_name="repulsion724"):
    """[n_coef, n_coef] matrix M such that (coeffs @ M) is the fODF reflected
    by diag(axis_signs). axis_signs e.g. (-1, 1, 1) for an x-flip."""
    from dipy.data import get_sphere
    from dipy.core.sphere import Sphere
    from dipy.reconst.shm import sh_to_sf_matrix
    sph = get_sphere(sphere_name)
    R = np.asarray(axis_signs, dtype=np.float64)
    sph_R = Sphere(xyz=sph.vertices * R[None, :])

    def _matrices(s):
        try:
            B, invB = sh_to_sf_matrix(s, sh_order_max=l_max, basis_type=basis_type,
                                      legacy=legacy, return_inv=True)
        except TypeError:
            B, invB = sh_to_sf_matrix(s, sh_order=l_max, basis_type=basis_type,
                                      return_inv=True)
        return B, invB

    B_R, _ = _matrices(sph_R)          # SH->SF at reflected directions
    _, invB = _matrices(sph)           # SF->SH at original directions
    # f'(v_k) = f(R v_k) = coeffs @ B_R ; refit: c' = (coeffs @ B_R) @ invB
    return B_R @ invB


def reflect_sh(coeffs, l_max, basis_type, axis_signs, legacy=False):
    """coeffs: [..., n_coef] -> reflected coeffs, same shape."""
    M = reflection_operator(l_max, basis_type, axis_signs, legacy=legacy)
    return coeffs @ M
