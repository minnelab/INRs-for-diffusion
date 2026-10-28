import torch 
from torch.utils.data import Dataset 
from inrdiff.dataset import DWIReprDataset
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


class ValidationSliceDataset(Dataset):
    def __init__(self, reference_dataset: DWIReprDataset, 
                 offset: int = 20, 
                 axis: int = 2,
                 b_indices: list = [0, 2, 6],
                 normalize_in_out=True, 
                 normalize_0_1=False, 
                 return_volumes=False):
        
        super().__init__()

        self.reference_dataset = reference_dataset
        self.offset = offset
        self.axis = axis
        self.b_indices = b_indices
        self.xyz_coord = None
        self.slice_idx = None
        self.normalize_in_out = normalize_in_out
        self.normalize_0_1 = normalize_0_1

        # Use validation data from the reference dataset
        self.bvecs_val = reference_dataset.bvecs
        self.bvals_val = reference_dataset.bvals
        self.image_val = reference_dataset.image
        self.num_slices = reference_dataset.num_slices
        self.return_volumes = return_volumes
        self._init_slice_grid()
        if not self.return_volumes:
            self.x, self.y = self._init_slice()

    def _init_slice_grid(self):
        # create slice to index x, y or z
        idx = [slice(None)] * len(self.reference_dataset.dim)  
        idx[self.axis] = self.offset  # could catch out of bounds error here
        self.slice_idx = idx

        self.xyz_coord = self.reference_dataset.xyz_grid[idx].reshape(-1, 3)

        # Print the slice index for validation

    def _get_slice(self, b_idx):
        assert self.xyz_coord is not None, 'call _init_slice_grid first'

        no_coords = self.xyz_coord.shape[0]
        # Ensure bvals_val and bvecs_val have the same number of dimensions
        bvals_val_expanded = self.bvals_val[None, b_idx].unsqueeze(-1)
        bvecs_val_expanded = self.bvecs_val[None, b_idx]
        
        # Create x for this b-index using validation bparams
        multi_b_param = torch.cat([bvals_val_expanded, bvecs_val_expanded], dim=-1).repeat(no_coords, 1)
        x = torch.cat([self.xyz_coord, multi_b_param], dim=-1)
        
        # Create y for this b-index using validation image
        y = self.image_val[self.xyz_coord[:, 0].round().int(), 
                                                self.xyz_coord[:, 1].round().int(), 
                                                self.xyz_coord[:, 2].round().int(), 
                                                b_idx]
        return x, y
        
    def _init_slice(self):
        assert self.xyz_coord is not None, 'call _init_slice_grid first'

        no_coords = self.xyz_coord.shape[0]
        
        # Initialize lists to store x and y for each b-index
        x_list = []
        y_list = []
        
        # Process each b-index
        for b_idx in self.b_indices:
            # Ensure bvals_val and bvecs_val have the same number of dimensions
            bvals_val_expanded = self.bvals_val[None, b_idx].unsqueeze(-1)
            bvecs_val_expanded = self.bvecs_val[None, b_idx]
            
            # Create x for this b-index using validation bparams
            multi_b_param = torch.cat([bvals_val_expanded, bvecs_val_expanded], dim=-1).repeat(no_coords, 1)
            x = torch.cat([self.xyz_coord, multi_b_param], dim=-1)
            x_list.append(x)
            
            # Create y for this b-index using validation image
            y = self.image_val[self.xyz_coord[:, 0].round().int(), 
                                                 self.xyz_coord[:, 1].round().int(), 
                                                 self.xyz_coord[:, 2].round().int(), 
                                                 b_idx]
            y_list.append(y)
        
        return x_list, y_list

    def _get_volume_slice(self, idx):
        """Get a slice from the entire volume for volume-based validation"""
        # Calculate which slice of the volume and which b-value we're accessing
        total_bvals = len(self.bvals_val)
        b_value_idx = idx // self.num_slices  # First determine which b-value
        slice_idx = idx % self.num_slices     # Then determine which slice
        
        
        # Safety check to prevent index out of bounds
        if b_value_idx >= total_bvals:
            print(f"Warning: b_value_idx {b_value_idx} out of range, using max available index {total_bvals-1}")
            b_value_idx = total_bvals - 1
        
        # Create a slice index that selects the appropriate slice along the specified axis
        vol_slice_idx = [slice(None)] * len(self.reference_dataset.dim)
        vol_slice_idx[self.axis] = slice_idx
        # Get xyz coordinates for this slice
        xyz_grid_slice = self.reference_dataset.xyz_grid[:, :, slice_idx, :].reshape(-1, 3)
        no_coords = xyz_grid_slice.shape[0]
        
        # Prepare b-values and b-vectors for this specific b-value index
        bvals_val_expanded = self.bvals_val[None, b_value_idx].unsqueeze(-1)
        bvecs_val_expanded = self.bvecs_val[None, b_value_idx]
        
        # Create x for this slice and b-value
        multi_b_param = torch.cat([bvals_val_expanded, bvecs_val_expanded], dim=-1).repeat(no_coords, 1)
        
        
        x = torch.cat([xyz_grid_slice, multi_b_param], dim=-1)
        
        # Create y for this slice and b-value
        y = self.image_val[xyz_grid_slice[:, 0].round().int(), 
                          xyz_grid_slice[:, 1].round().int(), 
                          xyz_grid_slice[:, 2].round().int(), 
                          b_value_idx]
        
        return x, y.unsqueeze(-1)

    def _fold_slice(self, x, y, b_idx_pos=0, denormalize=False, save_path=None):
        assert self.xyz_coord is not None, 'call _init_slice_grid first'
        if denormalize:
            if self.normalize_0_1:
                x, y = self.reference_dataset.normalize_xy_01(x, y, inverse=True)
            else:
                x, y = self.reference_dataset.normalize_xy_meanstd(x, y, inverse=True)
        
        # Make sure xyz_coord is on the same device as x
        if self.xyz_coord.device != x.device:
            xyz_coord_device = self.xyz_coord.to(x.device)
        else:
            xyz_coord_device = self.xyz_coord
            
        # Reshape y to match expected dimensions
        slice_ = y.reshape(self.reference_dataset.dim[:2])

        slice_idx = self.slice_idx
        
        # Determine b_idx based on mode
        if self.return_volumes:
            # In return_volumes mode, b_idx_pos is directly used as index
            # b_idx is the actual index into bvals_val
            if b_idx_pos < len(self.bvals_val):
                b_idx = b_idx_pos
            else:
                # Fallback to using the first b-value if b_idx_pos is out of range
                b_idx = 0
                print(f"Warning: b_idx_pos {b_idx_pos} out of range, using b_idx=0")
        else:
            # In standard mode, b_idx_pos is an index into self.b_indices
            if b_idx_pos < len(self.b_indices):
                b_idx = self.b_indices[b_idx_pos]
            else:
                # Fallback to using the first b-index if b_idx_pos is out of range
                b_idx = self.b_indices[0] if len(self.b_indices) > 0 else 0
                print(f"Warning: b_idx_pos {b_idx_pos} out of range, using first b-index")

        # Get mask slice and ensure it's on the same device as the data
        mask_slice = self.reference_dataset.mask[slice_idx[:3]].bool()
        if mask_slice.device != slice_.device:
            mask_slice = mask_slice.to(slice_.device)
        
        # Create a zero tensor for the normalized slice
        normalized_slice = torch.zeros_like(slice_)
        
        # Apply mask to both target and prediction
        if mask_slice.any():
            # Get masked data
            masked_data = slice_[mask_slice]
            
            # Get normalization parameters
            min_val = self.reference_dataset.y_min
            max_val = self.reference_dataset.y_max
            
            # Normalize only within the mask
            if max_val > min_val:
                normalized_slice[mask_slice] = (masked_data - min_val) / (max_val - min_val)
            else:
                normalized_slice[mask_slice] = masked_data

        # Save the slice as a PNG image if a save path is provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.imshow(normalized_slice.cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(save_path, f'slice_{b_idx}.png'), bbox_inches='tight', pad_inches=0)
            plt.close()

        return normalized_slice

    def __len__(self):
        if not self.return_volumes:
            assert self.x is not None, 'call DWIReprDataset._init_repr first'
            return len(self.x[0])  # Length of first x in the list
        else:
            # Pattern: b-value 0, all slices; b-value 1, all slices; etc.
            return len(self.bvals_val) * self.num_slices

    def __getitem__(self, idx):
        if not self.return_volumes:
            # Get x and y for all b-indices
            x_vals = [x_b[idx] for x_b in self.x]
            y_vals = [y_b[idx].unsqueeze(-1) for y_b in self.y]  # Add extra dimension to each y
            
            # normalize if required
            if self.normalize_in_out:
                x_norm = []
                y_norm = []
                for x, y in zip(x_vals, y_vals):
                    if self.normalize_0_1:
                        x_n, y_n = self.reference_dataset.normalize_xy_01(x, y)
                    else:
                        x_n, y_n = self.reference_dataset.normalize_xy_meanstd(x, y)
                    x_norm.append(x_n)
                    y_norm.append(y_n)
                
                return x_norm, y_norm, idx
            return x_vals, y_vals, idx
        else:
            # Handle volume-based access
            x, y = self._get_volume_slice(idx)
            # normalize if required
            if self.normalize_in_out:
                if self.normalize_0_1:
                    x, y = self.reference_dataset.normalize_xy_01(x, y)
                else:
                    x, y = self.reference_dataset.normalize_xy_meanstd(x, y)
            
            return x, y, idx


class ValidationCubeDataset(ValidationSliceDataset):

    def _get_volume(self, idx):
        """Get a slice from the entire volume for volume-based validation - not sliced!"""
        total_bvals = len(self.bvals_val)
        b_value_idx = idx 
        
        # Safety check to prevent index out of bounds
        if b_value_idx >= total_bvals:
            print(f"Warning: b_value_idx {b_value_idx} out of range, using max available index {total_bvals-1}")
            b_value_idx = total_bvals - 1

        # Create a slice index that selects the appropriate slice along the specified axis
        xyz_grid= self.reference_dataset.xyz_grid.reshape(-1, 3)
        no_coords = xyz_grid.shape[0]
        
        # Prepare b-values and b-vectors for this specific b-value index
        bvals_val_expanded = self.bvals_val[None, b_value_idx].unsqueeze(-1)
        bvecs_val_expanded = self.bvecs_val[None, b_value_idx]
        
        # Create x for this slice and b-value
        multi_b_param = torch.cat([bvals_val_expanded, bvecs_val_expanded], dim=-1).repeat(no_coords, 1)
        
        x = torch.cat([xyz_grid, multi_b_param], dim=-1)
        
        # Create y for this slice and b-value
        y = self.image_val[xyz_grid[:, 0].round().int(), 
                          xyz_grid[:, 1].round().int(), 
                          xyz_grid[:, 2].round().int(), 
                          b_value_idx]

        return x, y.unsqueeze(-1)

    def __getitem__(self, idx):
        # Handle volume-based access
        x, y = self._get_volume(idx)
        # normalize if required
        if self.normalize_in_out:
            if self.normalize_0_1:
                x, y = self.reference_dataset.normalize_xy_01(x, y)
            else:
                x, y = self.reference_dataset.normalize_xy_meanstd(x, y)
        
        return x, y, idx

    def __len__(self):
        # Pattern: b-value 0, all slices; b-value 1, all slices; etc.
        return len(self.bvals_val) 


if __name__ == "__main__":
    import torch
    import sys
    import os
    import yaml
    from inrdiff.dataset import DWIReprDataset
    from inrdiff.utils import load_data
    
    # Test with a sample config
    root = "../HCP_example"
    config = {
        'data': os.path.join(root, 'data.nii.gz'),
        'mask': os.path.join(root, 'brain_mask.nii.gz'),
        'bvec': os.path.join(root, 'bvecs.txt'),
        'bval': os.path.join(root, 'bvals.txt'),
        'normalize_in_out': True,
        'normalize_0_1': True,
    }
    
    # If command line arguments are provided, use them instead
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    
    try:
        # Load data
        print("Loading data...")
        dwi, mask, bvec, bval = load_data(config)
        
        # Create dataset
        print("Creating reference dataset...")
        reference_dataset = DWIReprDataset(
            dwi, mask, bvec, bval, 
            normalize_0_1=config.get('normalize_0_1', False), 
            normalize_in_out=config.get('normalize_in_out', True)
        )
        
        # Test ValidationCubeDataset
        print(f"\nTesting standard mode of ValidationCubeDataset")
        
        val_dataset = ValidationCubeDataset(
            reference_dataset,
            b_indices=[0, 2, 7],  # Test with a few b-values
            normalize_in_out=config.get('normalize_in_out', True),
            normalize_0_1=config.get('normalize_0_1', False),
            return_volumes=True # must be set here
        )
        # call
        val_dataset._get_volume(0)


        
        # Test ValidationSliceDataset in standard mode
        axis = 2  # Z-axis
        slice_idx = 20  # Middle slice
        print(f"\nTesting standard mode with slice {slice_idx} along axis {axis}")
        
        val_dataset = ValidationSliceDataset(
            reference_dataset,
            offset=slice_idx,
            axis=axis,
            b_indices=[0, 2, 7],  # Test with a few b-values
            normalize_in_out=config.get('normalize_in_out', True),
            normalize_0_1=config.get('normalize_0_1', False),
            return_volumes=False
        )
        
        print(f"Dataset length (standard mode): {len(val_dataset)}")
        
        # Loop through all items in the dataset
        print("Looping through standard mode dataset...")
        for i in tqdm(range(len(val_dataset))):
            x, y, idx = val_dataset[i]
            if i == 0:
                print(f"Sample item shapes: x[0]={x[0].shape}, y[0]={y[0].shape}")
        
        # Test with volume mode
        print("\nTesting volume mode")
        vol_dataset = ValidationSliceDataset(
            reference_dataset,
            offset=slice_idx,
            axis=axis,
            normalize_in_out=config.get('normalize_in_out', True),
            normalize_0_1=config.get('normalize_0_1', False),
            return_volumes=True
        )
        
        print(f"Dataset length (volume mode): {len(vol_dataset)}")
        
        # Loop through all items in the volume mode dataset
        print("Looping through volume mode dataset...")
        total_bvals = len(reference_dataset.bvals)
        
        for i in tqdm(range(len(vol_dataset))):
            try:
                x, y, idx = vol_dataset[i]
                if i % 10 == 0:  # Print info every 10 items
                    b_value_idx = i // vol_dataset.num_slices
                    slice_idx = i % vol_dataset.num_slices
                    print(f"Item {i}: slice={slice_idx}, b_value_idx={b_value_idx}, shapes: x={x.shape}, y={y.shape}")
            except Exception as e:
                print(f"Error on item {i}: {e}")
                break
                
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error in testing ValidationSliceDataset: {e}")
        import traceback
        traceback.print_exc()