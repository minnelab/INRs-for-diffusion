from abc import ABC, abstractmethod
from typing import Any, Union, Tuple, List, Iterable

import torch
from torch import nn
import numpy as np
from e3nn import o3

from inrdiff.networks.utils import xyz_to_angles, angles_to_xyz

class AbstractPosEncoder(nn.Module, ABC):
    def __init__(self, coord_size: int, **kwargs):
        super().__init__()
        self._in_size = coord_size

    @property
    def in_size(self):
        return self._in_size

    @property
    @abstractmethod
    def out_size(self):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__} (in_size: {self.in_size}, out_size: {self.out_size})"


class ThruEncoder(AbstractPosEncoder):
    """ Identity positional encoder. """
    def __init__(self, coord_size: int, **kwargs):
        super(ThruEncoder, self).__init__(coord_size, **kwargs)

    @property
    def out_size(self):
        return self.in_size

    def forward(self, coords):
        return coords


class BvecAngleEncoder(AbstractPosEncoder):
    """ Encodes the b vectors as angles in spherical coordinates. """
    def __init__(self, coord_size: int, vec_idx: tuple = (0, 1, 2), **kwargs):
        super(BvecAngleEncoder, self).__init__(coord_size, **kwargs)
        self._in_size = coord_size
        self.vec_idx = vec_idx


    @property
    def out_size(self):
        return self.in_size - 1

    def forward(self, coords):
        angles = xyz_to_angles(coords[..., self.vec_idx])
        remaining_coords = [
            coords[..., i:i+1] for i in range(coords.shape[-1]) 
            if i not in self.vec_idx
        ]
        
        return torch.cat(
            remaining_coords + [angles[0][..., None], angles[1][..., None]], dim=-1
        )


class NeRFPosEncoder(AbstractPosEncoder):
    """ Vectorized version of the positional encoder from the official NeRF paper [Mildenhall et al. 2020].
     Original implementation found above. """
    def __init__(self, coord_size: int, freq_num: Union[int, Tuple[int], List[int]], freq_scale: float = 1.0, **kwargs):
        super(NeRFPosEncoder, self).__init__(coord_size, **kwargs)
        if isinstance(freq_num, int):
            freq_num = [freq_num] * coord_size
        elif isinstance(freq_num, (list, tuple)):
            assert len(freq_num) == coord_size
        self.freq_num = freq_num
        self.freq_scale = freq_scale
        # Pre-initialize the frequency components. Each coordinate may have a different number of frequencies.
        self.exp_i_pi = torch.cat([2**torch.arange(i, dtype=torch.float32)[None] * self.freq_scale * np.pi
                                   for i in self.freq_num],
                                  dim=1)

    @property
    def out_size(self):
        return 2 * sum(self.freq_num)

    def forward(self, coords):
        coords_ = torch.cat([torch.tile(coords[..., j:j+1], (1, n)) for j, n in enumerate(self.freq_num)], dim=-1)
        exp_i_pi = torch.tile(self.exp_i_pi.to(coords.device), (coords_.shape[0], 1))
        prod = exp_i_pi * coords_
        out = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)
        return out


class FourierFeatPosEncoder(AbstractPosEncoder):
    """ Positional encoder from Fourite Features [Tancik et al. 2020]
     Implementation based on https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb """
    def __init__(self, coord_size: int, freq_num: int, freq_scale: float = 1.0, **kwargs):
        super(FourierFeatPosEncoder, self).__init__(coord_size, **kwargs)
        assert isinstance(freq_num, int)
        self.freq_num = freq_num
        self.freq_scale = freq_scale
        self.B_gauss = torch.normal(0.0, 1.0, size=(coord_size, self.freq_num), requires_grad=False) * self.freq_scale
        self.B_gauss_pi = 2. * np.pi * self.B_gauss

    @property
    def out_size(self):
        return 2 * self.freq_num

    def get_extra_state(self) -> Any:
        """ Required to store gaussian array into network state dict.
        Otherwise the positional encoder will not be the same when you load a saved checkpoint. """
        return {"B_gauss_pi": self.B_gauss_pi}

    def set_extra_state(self, state: Any):
        """ Required to store gaussian array into network state dict.
        Otherwise the positional encoder will not be the same when you load a saved checkpoint. """
        self.B_gauss_pi = state["B_gauss_pi"]

    def forward(self, coords):
        prod = coords @ self.B_gauss_pi.to(coords.device)
        out = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)
        return out



class SpatialOnlyEncoder(AbstractPosEncoder):
    """Encodes (x, y, z) using Gaussian Fourier features.

    Used for SHORE output mode where neither bval nor bvec are inputs.
    Output: [sin/cos(spatial)] = 2*freq_num dims.
    """

    def __init__(self, coord_size: int = 3, freq_num: int = 128, freq_scale: float = 1.0,
                 freq_learnable: bool = False, **kwargs):
        super().__init__(coord_size, **kwargs)
        self.freq_num = freq_num
        self.freq_learnable = bool(freq_learnable)

        B_gauss = torch.normal(0.0, 1.0, size=(3, freq_num),
                               requires_grad=False) * freq_scale
        B_pi = 2.0 * np.pi * B_gauss
        if self.freq_learnable:
            # MSMT-CSD-INR makes its frequency matrix an nn.Parameter
            # (models.py: `self.B = nn.Parameter(torch.randn([lpos, 3]) * sigma)`),
            # so the embedding bandwidth is LEARNED rather than fixed at init. Ours
            # has always been a frozen buffer. Exposed as a flag so the two can be
            # compared with nothing else changed.
            self.B_gauss_pi = torch.nn.Parameter(B_pi)
        else:
            self.register_buffer("B_gauss_pi", B_pi)

    @property
    def out_size(self):
        return 2 * self.freq_num

    def get_extra_state(self) -> Any:
        # A learnable B is already in state_dict as a parameter; duplicating it in
        # extra_state would make load_state_dict ambiguous.
        return {} if self.freq_learnable else {"B_gauss_pi": self.B_gauss_pi}

    def set_extra_state(self, state: Any):
        if not self.freq_learnable and state:
            self.B_gauss_pi.copy_(state["B_gauss_pi"])

    def forward(self, coords):
        # coords: [B, 3] = (x, y, z)
        spatial = coords[..., :3]
        prod = spatial @ self.B_gauss_pi
        return torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)


class SpatialBvalEncoder(AbstractPosEncoder):
    """Encodes (x, y, z, bval) using separate Gaussian Fourier features.

    Used for SH output mode where directions are not part of the input.
    Output: [sin/cos(spatial), sin/cos(bval)] = 2*freq_num + 2*bval_freq_num dims.
    """

    def __init__(self, coord_size: int = 4, freq_num: int = 128, freq_scale: float = 1.0,
                 bval_freq_num: int = 16, bval_freq_scale: float = 1.0, **kwargs):
        super().__init__(coord_size, **kwargs)
        self.freq_num = freq_num
        self.bval_freq_num = bval_freq_num

        B_gauss_spatial = torch.normal(0.0, 1.0, size=(3, freq_num),
                                       requires_grad=False) * freq_scale
        B_gauss_bval = torch.normal(0.0, 1.0, size=(1, bval_freq_num),
                                     requires_grad=False) * bval_freq_scale

        self.register_buffer("B_gauss_pi_spatial", 2.0 * np.pi * B_gauss_spatial)
        self.register_buffer("B_gauss_pi_bval", 2.0 * np.pi * B_gauss_bval)

    @property
    def out_size(self):
        return 2 * self.freq_num + 2 * self.bval_freq_num

    def get_extra_state(self) -> Any:
        return {
            "B_gauss_pi_spatial": self.B_gauss_pi_spatial,
            "B_gauss_pi_bval": self.B_gauss_pi_bval,
        }

    def set_extra_state(self, state: Any):
        self.B_gauss_pi_spatial.copy_(state["B_gauss_pi_spatial"])
        self.B_gauss_pi_bval.copy_(state["B_gauss_pi_bval"])

    def forward(self, coords):
        # coords: [B, 4] = (x, y, z, bval)
        spatial = coords[..., :3]
        bval = coords[..., 3:4]

        spatial_prod = spatial @ self.B_gauss_pi_spatial
        spatial_feat = torch.cat((torch.sin(spatial_prod), torch.cos(spatial_prod)), dim=-1)

        bval_prod = bval @ self.B_gauss_pi_bval
        bval_feat = torch.cat((torch.sin(bval_prod), torch.cos(bval_prod)), dim=-1)

        return torch.cat([spatial_feat, bval_feat], dim=-1)


class SphericalPosEncoder(AbstractPosEncoder):

    def __init__(self,
                coord_size: int,
                l_max: int,
                freq_num:int,
                freq_scale:int=1.0,
                antipodal: bool=True,
                bval_freq_num: int=None,
                bval_freq_scale: float=None,
                **kwargs):

        super().__init__(coord_size, **kwargs)

        self.l_max = l_max
        self.antipodal = antipodal
        self.coord_size = coord_size
        self.freq_num = freq_num
        self.freq_scale = freq_scale
        self.bval_freq_num = bval_freq_num
        self.bval_freq_scale = bval_freq_scale if bval_freq_scale is not None else freq_scale

        if bval_freq_num is not None:
            # Separate encoding: spatial [x,y,z] and bval [bval]
            spatial_size = coord_size - 1  # 3 for [x,y,z]
            B_gauss_spatial = torch.normal(0.0, 1.0,
                                           size=(spatial_size, freq_num),
                                           requires_grad=False) * self.freq_scale
            B_gauss_pi_spatial = 2. * np.pi * B_gauss_spatial

            B_gauss_bval = torch.normal(0.0, 1.0,
                                        size=(1, bval_freq_num),
                                        requires_grad=False) * self.bval_freq_scale
            B_gauss_pi_bval = 2. * np.pi * B_gauss_bval

            self.register_buffer("B_gauss_spatial", B_gauss_spatial)
            self.register_buffer("B_gauss_pi_spatial", B_gauss_pi_spatial)
            self.register_buffer("B_gauss_bval", B_gauss_bval)
            self.register_buffer("B_gauss_pi_bval", B_gauss_pi_bval)
        else:
            # Original behavior: encode all coords together
            B_gauss = torch.normal(0.0, 1.0,
                                        size=(coord_size, freq_num),
                                        requires_grad=False) * self.freq_scale
            B_gauss_pi = 2. * np.pi * B_gauss

            self.register_buffer("B_gauss", B_gauss)
            self.register_buffer("B_gauss_pi", B_gauss_pi)

        # init l
        if antipodal:
            self.l_values = list(range(0, l_max + 1, 2))
        else:
            self.l_values = list(range(l_max + 1))

    def get_extra_state(self) -> Any:
        if self.bval_freq_num is not None:
            return {
                "B_gauss_pi_spatial": self.B_gauss_pi_spatial,
                "B_gauss_pi_bval": self.B_gauss_pi_bval,
            }
        return {"B_gauss_pi": self.B_gauss_pi}

    def set_extra_state(self, state: Any):
        if "B_gauss_pi_spatial" in state:
            self.B_gauss_pi_spatial.copy_(state["B_gauss_pi_spatial"])
            self.B_gauss_pi_bval.copy_(state["B_gauss_pi_bval"])
        else:
            self.B_gauss_pi.copy_(state["B_gauss_pi"])

    @property
    def out_size(self):
        sph_size = sum([2 * l_ + 1 for l_ in self.l_values])
        if self.bval_freq_num is not None:
            return 2 * self.freq_num + 2 * self.bval_freq_num + sph_size
        return 2 * self.freq_num + sph_size


    def forward(self, coords):

        assert coords.shape[-1] == self.coord_size + 2  # coords + 2 angles

        # spherical coords (q values)
        pos, alpha, beta = (coords[..., :self.coord_size],
                            coords[..., self.coord_size],
                            coords[..., self.coord_size + 1])
        sph_coords = o3.spherical_harmonics_alpha_beta(self.l_values, alpha, beta)

        # positional coords (p and b values)
        if self.bval_freq_num is not None:
            spatial = pos[..., :3]
            bval = pos[..., 3:4]
            spatial_prod = spatial @ self.B_gauss_pi_spatial
            spatial_feat = torch.cat((torch.sin(spatial_prod), torch.cos(spatial_prod)), dim=-1)
            bval_prod = bval @ self.B_gauss_pi_bval
            bval_feat = torch.cat((torch.sin(bval_prod), torch.cos(bval_prod)), dim=-1)
            pos_coords = torch.cat([spatial_feat, bval_feat], dim=-1)
        else:
            prod = pos @ self.B_gauss_pi
            pos_coords = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)

        return torch.cat([pos_coords, sph_coords], dim=-1)
