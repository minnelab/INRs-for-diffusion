import torch
import numpy as np
from e3nn import o3
from e3nn.o3 import xyz_to_angles, angles_to_xyz
from torch.utils.data import Dataset

from inrdiff.networks.utils import xyz_to_angles as xyz_to_angles_custom


def encode_bval(bval, encoding="linear", max_bval=3000):
    """Encode b-value for network input.

    Args:
        bval: raw b-value tensor
        encoding: "linear" (b/max_bval) or "exp" (exp(-b/1000))
        max_bval: normalization constant for linear encoding
    """
    if encoding == "exp":
        return torch.exp(-bval / 1000.0)
    else:
        return bval / max_bval


def decode_bval(encoded, encoding="linear", max_bval=3000):
    """Inverse of encode_bval."""
    if encoding == "exp":
        return -1000.0 * torch.log(encoded.clamp(min=1e-8))
    else:
        return encoded * max_bval


class DWIReprDataset(Dataset):

    def __init__(self,
                 image,
                 mask,
                 bvecs,
                 bvals,
                 normalize_in_out=True,
                 normalize_0_1=False,
                 subsampl_step=1,
                 bval_selection='all',
                 max_bval=3000,
                 bval_encoding='linear',
                 cfg_precache_hash=None,
                 preload_to_gpu=False,
                 device='cpu',
                 is_train=True,
                 dwi_headers=None):

        super().__init__()
        self.device = device
        self.normalize_in_out = normalize_in_out
        self.normalize_0_1 = normalize_0_1
        self.bval_encoding = bval_encoding
        self.storage_precicion = torch.float16
        self.element_precicion = torch.float32
        self.eps = 1e-8
        self.y_min, self.y_max = image.min(), image.max()
        self.num_slices = image.shape[2]
        self.cfg_precache_hash = cfg_precache_hash
        self.max_bval = max_bval
        self.dwi_headers = dwi_headers
        image = self._subsample_spatial(image, subsampl_step)
        mask = self._subsample_spatial(mask, subsampl_step)

        bvecs, bvals, image = self._select_b(bvecs, bvals, image, bval_selection)        

        self.image = image.to(self.device, dtype=self.storage_precicion)
        self.mask = mask.to(self.device, dtype=torch.bool)

        self.bvecs = bvecs.to(self.device, dtype=self.storage_precicion)
        self.bvals = bvals.to(self.device, dtype=self.storage_precicion)
        # check if bvals are non-negative
        if (self.bvals<0).any():    
            print(f"bvals: {self.bvals}")
            raise ValueError("b-values are negative")
        self.bparams = torch.cat([self.bvals[:, None], self.bvecs], dim=-1)
        self.dim = self.image.shape
        self.num_samples = torch.prod(torch.tensor(self.dim)).item()
        
        print(f"dim: {self.dim}")
        self.xyz_grid = None 
        self.x, self.y = None, None
        # self.normalize_in_out = normalize_in_out
        self.x_mean, self.x_std = None, None
        self.y_mean, self.y_std = None, None
        self.y_min, self.y_max = self.image.min(), self.image.max()
        # self.normalize_0_1 = normalize_0_1
        # call initialisation 

        self._init_repr(self.image, self.mask, self.bvecs, self.bvals, is_train=is_train)

        # pre-compute normalization ()
        if self.normalize_in_out:
            # self.x_norm, self.y_norm = None, None
            if self.normalize_0_1:
                self.x_norm, self.y_norm = self.normalize_xy_01(self.x, self.y)
            else:
                self.x_norm, self.y_norm = self.normalize_xy_meanstd(self.x, self.y)
        else:
            self.x_norm, self.y_norm = self.x, self.y

        # shuffle once
        perm = torch.randperm(self.x_norm.shape[0])
        self.x_norm = self.x_norm[perm]
        self.y_norm = self.y_norm[perm]
        self._shuffle_perm = perm  # saved for subclasses that need aligned shuffling

        if preload_to_gpu:
            self.x_norm = self.x_norm.to('cuda')
            self.y_norm = self.y_norm.to('cuda')


    def _init_repr(self, image, mask, bvecs, bvals, is_train=True):
        image = image.to(self.device, dtype=self.storage_precicion)
        mask = mask.to(self.device, dtype=torch.bool)
        bvecs = bvecs.to(self.device, dtype=self.storage_precicion)
        bvals = bvals.to(self.device, dtype=self.storage_precicion)

        bparams = torch.cat([bvals[:, None], bvecs], dim=-1)
        dim = image.shape
        xyz_grid = None
        x, y = None, None

        x_unfold, y_unfold, xyz_grid = self._unfold_xyz(image, bparams)
        mask_unfold = self._unfold_mask(mask, image)

        # Mask values
        x_masked, y_masked = x_unfold[mask_unfold], y_unfold[mask_unfold]
        print(f"Loaded {x_masked.shape[0]} {'training' if is_train else 'validation'} points")

        # Set self.x and self.y regardless of train/val mode
        self.x, self.y = x_masked, y_masked
        self.dim, self.xyz_grid = dim, xyz_grid
        
        if is_train:
            self.image = image
            self.bvecs, self.bvals, self.bparams = bvecs, bvals, bparams
            self.dim, self.xyz_grid =  dim, xyz_grid
            if self.normalize_in_out:
                self.y_mean, self.y_std = self.y.mean(), self.y.std()
        else:
            self.image_val = image
            self.bvecs_val, self.bvals_val, self.bparams_val = bvecs, bvals, bparams
            self.dim_val, self.xyz_grid_val = dim, xyz_grid
            self.x_val, self.y_val = x_masked, y_masked

            if self.normalize_in_out:
                self.y_mean_val, self.y_std_val = self.y_val.mean(), self.y_val.std()
                

    def _unfold_xyz(self, image, bparams):
        dim = image.shape
        n_coord = dim[0] * dim[1] * dim[2]
        n_c = dim[-1]

        # reshape into [dimx x dimy x dimz, dimc]
        xyz_grid = torch.stack(torch.meshgrid(
                torch.arange(dim[0]),
                torch.arange(dim[1]),
                torch.arange(dim[2]), indexing='ij'), 
                dim=-1).to(self.device, dtype=torch.int16)


        xyz_coord = xyz_grid.reshape(-1, 3)

        # broadcast bval/bvec channel [n_coord, C]
        bparams = bparams[None,].expand(n_coord, -1, -1).reshape(-1, 4)  

        # repeat coords [n_coord x n_c, 3]
        xyz_coords_expanded = xyz_coord.repeat_interleave(n_c, dim=0)

        # collect indices 
        ret_x = torch.column_stack([xyz_coords_expanded, bparams])

        # reform also the y
        # ret_y = image.view(-1)
        ret_y = image.reshape(-1)
        # add assert that b-values are not negative
        assert torch.all(ret_x[:, 3] >= 0), 'b-values are negative'

        return ret_x, ret_y, xyz_grid

    @staticmethod
    def _subsample_spatial(img, subsampl_step):
        return img[::subsampl_step, ::subsampl_step, ::subsampl_step]

    @staticmethod
    def _select_b(bvecs, bvals, image, bval_selection, bval_tol=50):
        if bval_selection == 'all':
            return bvecs, bvals, image
        
        assert isinstance(bval_selection, list), "please provide a list of bvals to include or set to all"

        bval_sel = torch.tensor(bval_selection)
        diff = torch.abs(bvals.unsqueeze(1) - bval_sel)  
        mask = (diff <= bval_tol).any(dim=1)
        bval_idx = torch.nonzero(mask, as_tuple=True)[0]

        return bvecs[bval_idx], bvals[bval_idx], image[..., bval_idx]


    def normalize_xy_meanstd(self, x, y, inverse=False):
        # Upcast to element precision to prevent rounding errors
        x = x.to(self.element_precicion)
        y = y.to(self.element_precicion)

        x_n, y_n = torch.empty_like(x), torch.empty_like(y)

        if not inverse:
            # Normalize x
            x_n[..., :3] = x[..., :3] / torch.tensor(self.dim[:3], dtype=self.storage_precicion, device=x.device)
            x_n[..., 3] = encode_bval(x[..., 3], self.bval_encoding, self.max_bval)
            x_n[..., 4:] = x[..., 4:]

            # Normalize y
            y_n = (y - self.y_mean) / self.y_std

        else:
            # Denormalize x
            x_n[..., :3] = x[..., :3] * torch.tensor(self.dim[:3], dtype=self.storage_precicion, device=x.device)
            x_n[..., 3] = decode_bval(x[..., 3], self.bval_encoding, self.max_bval)
            x_n[..., 4:] = x[..., 4:]

            # Denormalize y
            y_n = (y * self.y_std) + self.y_mean

        # Downcast to storage precision for return
        return x_n.to(self.storage_precicion), y_n.to(self.storage_precicion)


    def normalize_xy_01(self, x, y, inverse=False):
        # Upcast to element precision to prevent rounding errors
        x = x.to(self.element_precicion)
        y = y.to(self.element_precicion)

        # normalize x between -1 and 1
            # Normalize x
        x_n, y_n = torch.empty_like(x), torch.empty_like(y)
        if not inverse:
            x_n[..., :3] = 2 * (x[..., :3] / torch.tensor(self.dim[:3], device=x.device)) - 1
            x_n[..., 3] = encode_bval(x[..., 3], self.bval_encoding, self.max_bval)
            x_n[..., 4:] = x[..., 4:]

        else:
            x_n[..., :3] = ((x[..., :3] + 1) / 2) * torch.tensor(self.dim[:3], device=x.device)
            x_n[..., 3] = decode_bval(x[..., 3], self.bval_encoding, self.max_bval)
            x_n[..., 4:] = x[..., 4:]

        y_n = (y - self.y_min) / (self.y_max - self.y_min) if not inverse else (y * (self.y_max - self.y_min)) + self.y_min

        # Downcast to storage precision for return
        return x_n, y_n
        # return x_n.to(self.storage_precicion), y_n.to(self.storage_precicion)


    @staticmethod
    def _unfold_bvals_bvecs(image, bvecs, bvals):
        pass

    def _fold_xyz(self, x, y, skip_checks=False, denormalize=False, force_int_coords=False):

        if denormalize:
            if self.normalize_0_1:
                x, y = self.normalize_xy_01(x, y, inverse=True)
            else:
                x, y = self.normalize_xy_meanstd(x, y, inverse=True)

        if force_int_coords:
            x[:, :3] = x[:, :3].round()

        if not skip_checks:
            assert self.xyz_grid is not None

            # test if all x,y,z values are integers
            assert torch.all(x[:, :3] == x[:, :3].int())

            # test if all x,y,z values are within the image grid
            assert self._vectors_in(x[:, :3], self.xyz_grid.reshape(-1, 3))

            # test if all bparams are within the bvecs, bvals
            assert self._vectors_in(x[:, 3:], self.bparams)

        # remove extra dim 
        y = y.squeeze()
        y = y.to(x.device)

        # Create image_unfolded on the same device and dtype as x
        image_unfolded = torch.zeros_like(self.image, device=x.device, dtype=x.dtype)

        # Convert y to the same dtype as image_unfolded
        y = y.to(image_unfolded.dtype)

        # recover the indices of the bparams 
        bparam_idx = self._map_to_bparam_idx(x[:, 3:])

        image_unfolded[x[:, 0].round().int(), 
                       x[:, 1].round().int(), 
                       x[:, 2].round().int(),
                       bparam_idx.int()] = y

        return image_unfolded


    def _unfold_mask(self, mask, image):

        mask_expanded = mask[..., None].repeat(1, 1, 1, image.shape[3])
        # ret_mask = mask_expanded.view(-1)
        ret_mask = mask_expanded.reshape(-1)

        return ret_mask


    @staticmethod
    def _vectors_in(A, B, atol=1e-6):
        # checks if all vectors in A are in B
        # Ensure both tensors are on same device
        if A.device != B.device:
            B = B.to(A.device)

        diff = A.unsqueeze(1) - B.unsqueeze(0)  # [N, M, 4]
        matches = torch.all(torch.abs(diff) <= atol, dim=2)  # [N, M]

        return torch.all(matches.any(dim=1))

    
    def _map_to_bparam_idx(self, A, atol=1e-6):
        # Make sure both tensors are on the same device
        if A.device != self.bparams.device:
            # Either move A to bparams device or vice versa
            # Since bparams is typically smaller, we'll move it to A's device
            bparams_device = self.bparams.to(A.device)
        else:
            bparams_device = self.bparams
        
        diff = A.unsqueeze(1) - bparams_device.unsqueeze(0)  # [N, M, 4]
        matches = torch.all(torch.abs(diff) <= atol, dim=2)  # [N, M]
        indices = torch.full((A.shape[0],), -1, dtype=torch.long, device=A.device)  
        for i in range(A.shape[0]):
            match_idx = torch.where(matches[i])[0]  
            if len(match_idx) > 0:
                indices[i] = match_idx[0]  

        return indices


    def __len__(self):
        assert self.x is not None, 'call DWIReprDataset._init_repr first'

        return self.x.shape[0]       


    def __getitem__(self, idx, train=True):
        assert self.x is not None, 'call DWIReprDataset._init_repr first'

        # Select train or validation data
        if train:
            x = self.x_norm[idx]
            y = self.y_norm[idx].unsqueeze(-1)

            return x, y, idx

        else:
            x = self.x_val[idx]
            y = self.y_val[idx].unsqueeze(-1)

            # Normalize
            if self.normalize_in_out:
                if self.normalize_0_1:
                    x_norm, y_norm = self.normalize_xy_01(x, y)
                else:
                    x_norm, y_norm = self.normalize_xy_meanstd(x, y)

                return x_norm, y_norm, idx

                # assert torch.all(x[:, 3] >= 0), 'b-values are negative'
            else:
                return x, y, idx



#         # 7) normalize
#         # 7a) de-mean
#         ch_all -= torch.tensor([dx, dy, dz, 0, torch.pi, self.bmax], device=self.device) / 2
#         ch_all *= torch.tensor([2/dx, 2/dy, 2/dz, 1/torch.pi, 2/torch.pi, 2/self.bmax], device=self.device) 

class PreBatchedDWIReprDataset(DWIReprDataset):
    """
    hacky method for fast indexing on gpu
    """

    def __init__(self, *args, train_batch_size=8192, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.x_norm is not None and self.y_norm is not None, "x_norm/y_norm must be precomputed"

        N = self.x_norm.shape[0]
        B = train_batch_size
        remainder = N % B

        if remainder != 0:
            pad_len = B - remainder
            self.x_norm = torch.cat([self.x_norm, self.x_norm[:pad_len]], dim=0)
            self.y_norm = torch.cat([self.y_norm, self.y_norm[:pad_len]], dim=0)        

        self.num_batches = self.x_norm.shape[0] // B
        self.x_batches = self.x_norm.view(self.num_batches, B, -1)
        self.y_batches = self.y_norm.view(self.num_batches, B, -1)

        # self.num_batches = N // B

        # self.x_batches = self.x_norm[:self.num_batches * B].view(self.num_batches, B, -1)
        # self.y_batches = self.y_norm[:self.num_batches * B].view(self.num_batches, B, -1)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        x, y = self.x_batches[idx], self.y_batches[idx]

        return x, y, idx


class SHOutputDWIReprDataset(DWIReprDataset):
    """Dataset for SH coefficient output mode.

    Each sample is (x, y, z, bval) -> [n_dirs] DWI signals for all directions
    in that shell. The network outputs SH coefficients, which are converted to
    signal via Y_matrix for loss computation.
    """

    def __init__(self, *args, sh_lmax=8, **kwargs):
        self.sh_lmax = sh_lmax
        self.l_values = list(range(0, sh_lmax + 1, 2))  # even orders, antipodal
        self.n_sh = sum(2 * l + 1 for l in self.l_values)  # 45 for lmax=8
        self._n_shells = None  # set during _unfold_xyz
        self.y_matrix = None   # [n_dirs, n_sh], set during _unfold_xyz
        self.shell_dir_counts = None
        self.shell_bvals = None
        super().__init__(*args, **kwargs)

    def _unfold_xyz(self, image, bparams):
        dim = image.shape  # [X, Y, Z, C]
        n_voxels = dim[0] * dim[1] * dim[2]
        bvals = bparams[:, 0]   # [C] raw b-values
        bvecs = bparams[:, 1:]  # [C, 3]

        # Group by shell (round to nearest 1000)
        shell_labels = torch.round(bvals / 1000) * 1000
        unique_shells = torch.unique(shell_labels).sort()[0]
        # Exclude B0 shells
        unique_shells = unique_shells[unique_shells > 50]
        self._n_shells = len(unique_shells)

        # Build xyz grid
        xyz_grid = torch.stack(torch.meshgrid(
            torch.arange(dim[0]),
            torch.arange(dim[1]),
            torch.arange(dim[2]), indexing='ij'),
            dim=-1).to(self.device, dtype=torch.int16)
        xyz_coord = xyz_grid.reshape(-1, 3)  # [n_voxels, 3]

        x_parts = []
        y_parts = []
        shell_bvecs_list = []
        n_dirs_per_shell = []

        for shell_bval in unique_shells:
            shell_mask = (shell_labels == shell_bval)
            shell_indices = torch.where(shell_mask)[0]
            n_dirs = len(shell_indices)
            n_dirs_per_shell.append(n_dirs)

            # x: [n_voxels, 4] = (x, y, z, actual_mean_bval)
            actual_bval = bvals[shell_indices].float().mean()
            bval_col = torch.full((n_voxels, 1), actual_bval.item(),
                                  dtype=bparams.dtype, device=self.device)
            x_shell = torch.cat([xyz_coord.to(bparams.dtype), bval_col], dim=-1)
            x_parts.append(x_shell)

            # y: [n_voxels, n_dirs]
            y_shell = image.reshape(n_voxels, -1)[:, shell_indices]
            y_parts.append(y_shell)

            shell_bvecs_list.append(bvecs[shell_indices])

        # Verify all shells have the same number of directions
        assert len(set(n_dirs_per_shell)) == 1, \
            f"All shells must have same number of directions, got {n_dirs_per_shell}"
        n_dirs = n_dirs_per_shell[0]
        self.shell_dir_counts = n_dirs
        self.shell_bvals = unique_shells

        # Verify bvecs are identical across shells
        for i in range(1, len(shell_bvecs_list)):
            if not torch.allclose(shell_bvecs_list[0].float(),
                                  shell_bvecs_list[i].float(), atol=1e-3):
                print("WARNING: bvecs differ across shells. Using first shell's bvecs.")
                break

        # Compute Y_matrix from first shell's bvecs
        reference_bvecs = shell_bvecs_list[0].float()
        alpha, beta = xyz_to_angles_custom(reference_bvecs)
        self.y_matrix = o3.spherical_harmonics_alpha_beta(
            self.l_values, alpha, beta)  # [n_dirs, n_sh]
        print(f"SH Y_matrix: {self.y_matrix.shape} (l_max={self.sh_lmax}, "
              f"{n_dirs} dirs, {self.n_sh} coeffs)")

        x_unfold = torch.cat(x_parts, dim=0)  # [n_voxels * n_shells, 4]
        y_unfold = torch.cat(y_parts, dim=0)  # [n_voxels * n_shells, n_dirs]

        assert x_unfold[:, 3].min() >= 0, 'b-values are negative'

        return x_unfold, y_unfold, xyz_grid

    def _unfold_mask(self, mask, image):
        assert self._n_shells is not None, '_unfold_xyz must be called before _unfold_mask'
        mask_expanded = mask[..., None].repeat(1, 1, 1, self._n_shells)
        ret_mask = mask_expanded.reshape(-1)
        return ret_mask


class PreBatchedSHOutputDWIReprDataset(SHOutputDWIReprDataset):
    """Pre-batched version of SHOutputDWIReprDataset for fast GPU training."""

    def __init__(self, *args, train_batch_size=8192, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.x_norm is not None and self.y_norm is not None

        N = self.x_norm.shape[0]
        B = train_batch_size
        remainder = N % B

        if remainder != 0:
            pad_len = B - remainder
            self.x_norm = torch.cat([self.x_norm, self.x_norm[:pad_len]], dim=0)
            self.y_norm = torch.cat([self.y_norm, self.y_norm[:pad_len]], dim=0)

        self.num_batches = self.x_norm.shape[0] // B
        self.x_batches = self.x_norm.view(self.num_batches, B, -1)
        self.y_batches = self.y_norm.view(self.num_batches, B, -1)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        return self.x_batches[idx], self.y_batches[idx], idx


class ShoreOutputDWIReprDataset(DWIReprDataset):
    """Dataset for SHORE coefficient output mode.

    Each sample is (x, y, z) -> [n_volumes] DWI signals for ALL directions
    and shells at that voxel. The network outputs SHORE coefficients, which
    are converted to signal via shore_matrix for loss computation.

    SHORE jointly encodes radial (b-value) and angular (direction) structure
    in a single basis, preventing the isotropic collapse seen with per-shell SH.
    """

    def __init__(self, *args, shore_radial_order=6, shore_zeta=700, shore_tau=None,
                 skip_y_normalize=False, b0_loss_mode=False, b0_image=None, **kwargs):
        self.shore_radial_order = shore_radial_order
        self.shore_zeta = shore_zeta
        self.shore_tau = shore_tau  # None = dipy default 1/(4*pi^2)
        self.shore_matrix = None  # [n_volumes, n_shore_coeffs], set during _unfold_xyz
        self.n_shore = None
        self._n_volumes_kept = None  # set during _unfold_xyz
        self._skip_y_normalize = skip_y_normalize
        self._b0_loss_mode = b0_loss_mode
        self._b0_image_raw = b0_image  # raw B0 image [X, Y, Z], normalized after dataset init
        self.b0_norm_per_voxel = None  # [N_masked], set during init if b0_loss_mode
        super().__init__(*args, **kwargs)

        # After parent init: compute normalized B0 per masked voxel.
        # Populated whenever a raw B0 image is provided; downstream consumers
        # (shore_b0_loss `predicted = signal * b0` reweighting AND the q=0
        # anchor supervision used by `b0_anchor_weight`) both read it.
        if self._b0_image_raw is not None:
            b0_flat = self._b0_image_raw.float().reshape(-1)
            mask_flat = self.mask.reshape(-1).cpu()
            b0_masked = b0_flat[mask_flat]
            # Normalize B0 the same way as y targets (on CPU)
            y_min = self.y_min.float().cpu()
            y_max = self.y_max.float().cpu()
            if self.normalize_0_1:
                b0_masked = (b0_masked - y_min) / (y_max - y_min)
            else:
                y_mean = self.y_mean.float().cpu()
                y_std = self.y_std.float().cpu()
                b0_masked = (b0_masked - y_mean) / y_std
            if self._b0_loss_mode:
                # shore_b0_loss divides by b0; clamp keeps it numerically safe.
                b0_masked = b0_masked.clamp(min=1e-3)
            # Apply same shuffle as x/y
            self.b0_norm_per_voxel = b0_masked[self._shuffle_perm].to(self.x_norm.dtype)
            print(f"B0 normalized per-voxel: range [{self.b0_norm_per_voxel.min():.4f}, {self.b0_norm_per_voxel.max():.4f}]")

    def _unfold_xyz(self, image, bparams):
        dim = image.shape  # [X, Y, Z, C]
        n_voxels = dim[0] * dim[1] * dim[2]
        bvals = bparams[:, 0]   # [C]
        bvecs = bparams[:, 1:]  # [C, 3]

        # Build xyz grid
        xyz_grid = torch.stack(torch.meshgrid(
            torch.arange(dim[0]),
            torch.arange(dim[1]),
            torch.arange(dim[2]), indexing='ij'),
            dim=-1).to(self.device, dtype=torch.int16)
        xyz_coord = xyz_grid.reshape(-1, 3)  # [n_voxels, 3]

        n_volumes = dim[-1]
        self._n_volumes_kept = n_volumes

        # Compute SHORE matrix using dipy
        from dipy.core.gradients import gradient_table
        from dipy.reconst.shore import shore_matrix as dipy_shore_matrix

        bvals_np = bvals.float().cpu().numpy()
        bvecs_np = bvecs.float().cpu().numpy()
        gtab = gradient_table(bvals_np, bvecs=bvecs_np)

        shore_kwargs = dict(radial_order=self.shore_radial_order, zeta=self.shore_zeta, gtab=gtab)
        if self.shore_tau is not None:
            shore_kwargs['tau'] = self.shore_tau
        M = dipy_shore_matrix(**shore_kwargs)  # [n_volumes, n_shore_coeffs]

        # Column-normalize for balanced gradients
        col_norms = np.linalg.norm(M, axis=0, keepdims=True)
        col_norms = np.maximum(col_norms, 1e-10)
        M_normalized = M / col_norms

        self.shore_matrix = torch.from_numpy(M_normalized).float()
        self.shore_col_norms = torch.from_numpy(col_norms.squeeze()).float()
        self.shore_pinv = torch.from_numpy(np.linalg.pinv(M_normalized)).float()  # [n_shore, n_volumes]
        self.n_shore = M.shape[1]
        print(f"SHORE matrix: {self.shore_matrix.shape} (radial_order={self.shore_radial_order}, "
              f"zeta={self.shore_zeta}, {self.n_shore} coefficients)")
        print(f"  Column norms after normalization: "
              f"[{self.shore_matrix.norm(dim=0).min():.4f}, {self.shore_matrix.norm(dim=0).max():.4f}]")

        # x: [n_voxels, 3] = (x, y, z) — no bval, no bvec
        x_unfold = xyz_coord.to(bparams.dtype)  # [n_voxels, 3]

        # y: [n_voxels, n_volumes] = all DWI signals at this voxel
        y_unfold = image.reshape(n_voxels, -1)  # [n_voxels, n_volumes]

        return x_unfold, y_unfold, xyz_grid

    def _unfold_mask(self, mask, image):
        # One entry per voxel (not per volume)
        return mask.reshape(-1)

    def normalize_xy_01(self, x, y, inverse=False):
        """Override: x is only (x,y,z) — no bval/bvec columns.
        When _skip_y_normalize is True, y is passed through unchanged
        (for shore_b0_normalize where E-space values are already in [0,1]).
        """
        x = x.to(self.element_precicion)
        y = y.to(self.element_precicion)

        x_n = torch.empty_like(x)
        if not inverse:
            x_n[..., :3] = 2 * (x[..., :3] / torch.tensor(self.dim[:3], device=x.device)) - 1
        else:
            x_n[..., :3] = ((x[..., :3] + 1) / 2) * torch.tensor(self.dim[:3], device=x.device)

        if self._skip_y_normalize:
            y_n = y
        else:
            y_n = (y - self.y_min) / (self.y_max - self.y_min) if not inverse else (y * (self.y_max - self.y_min)) + self.y_min
        return x_n, y_n

    def normalize_xy_meanstd(self, x, y, inverse=False):
        """Override: x is only (x,y,z) — no bval/bvec columns."""
        x = x.to(self.element_precicion)
        y = y.to(self.element_precicion)

        x_n = torch.empty_like(x)
        if not inverse:
            x_n[..., :3] = x[..., :3] / torch.tensor(self.dim[:3], dtype=self.storage_precicion, device=x.device)
            y_n = y if self._skip_y_normalize else (y - self.y_mean) / self.y_std
        else:
            x_n[..., :3] = x[..., :3] * torch.tensor(self.dim[:3], dtype=self.storage_precicion, device=x.device)
            y_n = y if self._skip_y_normalize else (y * self.y_std) + self.y_mean

        return x_n.to(self.storage_precicion), y_n.to(self.storage_precicion)


class PreBatchedShoreOutputDWIReprDataset(ShoreOutputDWIReprDataset):
    """Pre-batched version of ShoreOutputDWIReprDataset for fast GPU training."""

    def __init__(self, *args, train_batch_size=8192, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.x_norm is not None and self.y_norm is not None

        N = self.x_norm.shape[0]
        B = train_batch_size
        remainder = N % B

        if remainder != 0:
            pad_len = B - remainder
            self.x_norm = torch.cat([self.x_norm, self.x_norm[:pad_len]], dim=0)
            self.y_norm = torch.cat([self.y_norm, self.y_norm[:pad_len]], dim=0)
            if self.b0_norm_per_voxel is not None:
                self.b0_norm_per_voxel = torch.cat([self.b0_norm_per_voxel, self.b0_norm_per_voxel[:pad_len]], dim=0)

        self.num_batches = self.x_norm.shape[0] // B
        self.x_batches = self.x_norm.view(self.num_batches, B, -1)
        self.y_batches = self.y_norm.view(self.num_batches, B, -1)
        if self.b0_norm_per_voxel is not None:
            b0_batched = self.b0_norm_per_voxel.view(self.num_batches, B)
            # Move to same device as x/y batches
            self.b0_batches = b0_batched.to(self.x_batches.device)
        else:
            self.b0_batches = None

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        return self.x_batches[idx], self.y_batches[idx], idx


class BesselOutputDWIReprDataset(ShoreOutputDWIReprDataset):
    """Dataset for Bessel-Fourier coefficient output mode.

    Structurally identical to ShoreOutputDWIReprDataset: the model emits
    per-voxel coefficients that are multiplied by a fixed design matrix to
    reconstruct the signal. The only difference is that the design matrix is
    built from a Bessel-Fourier basis (see src/bessel_basis.py) instead of
    SHORE. Attribute names mirror the SHORE class (``shore_matrix``,
    ``shore_pinv``, ``n_shore``) so the downstream training and evaluation
    paths need no branching.
    """

    def __init__(
        self,
        *args,
        bessel_radial_order=2,
        bessel_lmax=8,
        bessel_q_max=None,
        **kwargs,
    ):
        self.bessel_radial_order = int(bessel_radial_order)
        self.bessel_lmax = int(bessel_lmax)
        self.bessel_q_max = None if bessel_q_max is None else float(bessel_q_max)
        self.bessel_q_max_used = None
        super().__init__(*args, **kwargs)

    def _unfold_xyz(self, image, bparams):
        from inrdiff.bessel_basis import bessel_fourier_matrix

        dim = image.shape
        n_voxels = dim[0] * dim[1] * dim[2]
        bvals = bparams[:, 0]
        bvecs = bparams[:, 1:]

        xyz_grid = torch.stack(torch.meshgrid(
            torch.arange(dim[0]),
            torch.arange(dim[1]),
            torch.arange(dim[2]), indexing='ij'),
            dim=-1).to(self.device, dtype=torch.int16)
        xyz_coord = xyz_grid.reshape(-1, 3)

        n_volumes = dim[-1]
        self._n_volumes_kept = n_volumes

        bvals_np = bvals.float().cpu().numpy()
        bvecs_np = bvecs.float().cpu().numpy()

        M, _coeff_info, q_max_used = bessel_fourier_matrix(
            bvals_np,
            bvecs_np,
            radial_order=self.bessel_radial_order,
            lmax=self.bessel_lmax,
            q_max=self.bessel_q_max,
        )

        col_norms = np.linalg.norm(M, axis=0, keepdims=True)
        col_norms = np.maximum(col_norms, 1e-10)
        M_normalized = M / col_norms

        self.shore_matrix = torch.from_numpy(M_normalized).float()
        self.shore_col_norms = torch.from_numpy(col_norms.squeeze()).float()
        self.shore_pinv = torch.from_numpy(np.linalg.pinv(M_normalized)).float()
        self.n_shore = int(M.shape[1])
        self.bessel_q_max_used = q_max_used
        print(
            f"Bessel-Fourier matrix: {tuple(self.shore_matrix.shape)} "
            f"(radial_order={self.bessel_radial_order}, lmax={self.bessel_lmax}, "
            f"q_max={q_max_used:.4f}, {self.n_shore} coefficients)"
        )
        print(
            f"  Column norms after normalization: "
            f"[{self.shore_matrix.norm(dim=0).min():.4f}, {self.shore_matrix.norm(dim=0).max():.4f}]"
        )

        x_unfold = xyz_coord.to(bparams.dtype)
        y_unfold = image.reshape(n_voxels, -1)
        return x_unfold, y_unfold, xyz_grid


class PreBatchedBesselOutputDWIReprDataset(BesselOutputDWIReprDataset):
    """Pre-batched version of BesselOutputDWIReprDataset."""

    def __init__(self, *args, train_batch_size=8192, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.x_norm is not None and self.y_norm is not None

        N = self.x_norm.shape[0]
        B = train_batch_size
        remainder = N % B

        if remainder != 0:
            pad_len = B - remainder
            self.x_norm = torch.cat([self.x_norm, self.x_norm[:pad_len]], dim=0)
            self.y_norm = torch.cat([self.y_norm, self.y_norm[:pad_len]], dim=0)
            if self.b0_norm_per_voxel is not None:
                self.b0_norm_per_voxel = torch.cat([self.b0_norm_per_voxel, self.b0_norm_per_voxel[:pad_len]], dim=0)

        self.num_batches = self.x_norm.shape[0] // B
        self.x_batches = self.x_norm.view(self.num_batches, B, -1)
        self.y_batches = self.y_norm.view(self.num_batches, B, -1)
        if self.b0_norm_per_voxel is not None:
            b0_batched = self.b0_norm_per_voxel.view(self.num_batches, B)
            self.b0_batches = b0_batched.to(self.x_batches.device)
        else:
            self.b0_batches = None

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        return self.x_batches[idx], self.y_batches[idx], idx


if __name__ == "__main__":

    from inrdiff.utils import load_data
    root = "../HCP_example"
    data = root + "/data.nii.gz"
    mask = root + "/brain_mask.nii.gz"
    bvec = root + "/bvecs.txt"
    bval = root + "/bvals.txt"
    config = {'data': data, 'mask': mask, 'bvec': bvec, 'bval': bval}
    dwi, mask, bvec, bval = load_data(config)

    volume_val_dataset = DWIReprDataset(dwi, mask, bvec, bval,
                                    normalize_0_1=True,
                                    normalize_in_out=True,
                                    subsampl_step=4)
