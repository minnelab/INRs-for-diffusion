import torch
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(cfg, return_header=False):
    data = nib.load(cfg['data'])
    mask_data = nib.load(cfg['mask'])

    dwi = torch.tensor(data.get_fdata(), dtype=torch.float32)
    mask = torch.tensor(mask_data.get_fdata(), dtype=torch.float32)
    bvec = torch.tensor(np.loadtxt(cfg['bvec']).T, dtype=torch.float32)
    bval = torch.tensor(np.loadtxt(cfg['bval']), dtype=torch.float32)

    if not return_header:
        return dwi, mask, bvec, bval
    else:
        return dwi, mask, bvec, bval, data.header


def extract_b0_prior(dwi, bvec, bval, b0_threshold=10,
                     mask=None, normalize_first=False):
    """Extract mean B0 image and remove B0 volumes from data.

    Args:
        dwi: [X, Y, Z, C] DWI data
        bvec: [C, 3] gradient directions
        bval: [C] b-values
        b0_threshold: threshold to identify B0 volumes
        mask: [X, Y, Z] brain mask (used when normalize_first=True)
        normalize_first: if True, normalize all DWI to [0,1] before B0 subtraction

    Returns:
        b0_image: [X, Y, Z] mean B0 image
        dwi: [X, Y, Z, C'] DWI residuals with B0 volumes removed
        bvec: [C', 3] bvecs without B0
        bval: [C'] bvals without B0
        (if normalize_first): also returns dwi_min, dwi_max
    """
    b0_mask = bval <= b0_threshold

    if normalize_first:
        if mask is not None:
            brain = dwi[mask.bool()]
            dwi_min, dwi_max = brain.min(), brain.max()
        else:
            dwi_min, dwi_max = dwi.min(), dwi.max()
        dwi = (dwi - dwi_min) / (dwi_max - dwi_min + 1e-8)

    b0_image = dwi[..., b0_mask].mean(dim=-1)  # Average B0

    non_b0_mask = ~b0_mask
    dwi = dwi[..., non_b0_mask]
    bvec = bvec[non_b0_mask]
    bval = bval[non_b0_mask]

    # Subtract B0 from all remaining volumes (compute residuals)
    dwi = dwi - b0_image.unsqueeze(-1)

    if normalize_first:
        return b0_image, dwi, bvec, bval, dwi_min, dwi_max
    return b0_image, dwi, bvec, bval


def fix_checkpoint_keys(checkpoint_path, save_fixed=True, fixed_path=None):
    """
    Fix checkpoint keys by renaming 'model.' to 'network.'
    
    Args:
        checkpoint_path: Path to original checkpoint
        save_fixed: Whether to save the fixed checkpoint
        fixed_path: Custom path to save fixed checkpoint (if None, uses original_path + '.fixed')
    
    Returns:
        Fixed checkpoint dictionary
    """
    print(f"Loading and fixing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # Keep track of keys that were renamed
    renamed_count = 0
    
    # Create a new state dict
    new_state_dict = {}
    
    # Filter out LPIPS-related keys and rename 'model.' to 'network.' if needed
    for k, v in state_dict.items():
        # Skip LPIPS keys
        if k.startswith('_lpips_fn.'):
            continue
            
        if k.startswith('model.'):
            new_k = k.replace('model.', 'network.')
            new_state_dict[new_k] = v
            renamed_count += 1
        else:
            new_state_dict[k] = v
    
    if renamed_count > 0:
        print(f"Renamed {renamed_count} keys from 'model.' to 'network.'")
    
    checkpoint['state_dict'] = new_state_dict
    
    if save_fixed:
        if fixed_path is None:
            fixed_path = checkpoint_path + '.fixed'
        print(f"Saving fixed checkpoint to: {fixed_path}")
        torch.save(checkpoint, fixed_path)
    
    return checkpoint


def split_datasets(data, bvec, bval, train_split=0.8, val_split=0.2):
    """
    Split datasets into train and validation sets, stratified by rounded b-values.
    First samples validation data (for consistent validation across experiments),
    then samples training data if needed.
    
    Args:
        data: Input DWI data tensor (last dimension is the measurement dimension)
        bvec: B-vectors tensor (first dimension is the measurement dimension)
        bval: B-values tensor (first dimension is the measurement dimension)
        train_split: Fraction of data for training (default: 0.8)
        val_split: Fraction of data for validation (default: 0.2)
        
    Returns:
        Tuple of (train_data, train_bvec, train_bval), 
              (val_data, val_bvec, val_bval)
    """
    # Round b-values to nearest 100 for stratification
    strata = torch.round(bval / 100) * 100
    
    # Get indices for the split
    indices = np.arange(bval.shape[0])
    
    # First sample validation data
    if val_split <= 0:
        val_idx = np.array([])
        remaining_idx = indices
    else:
        try:
            # Sample validation data first
            remaining_idx, val_idx = train_test_split(
                indices, 
                test_size=val_split,
                random_state=42,  # for reproducibility
                stratify=strata.numpy()
            )
        except ValueError as e:
            # Handle case where stratification fails (not enough samples per class)
            print(f"Warning: Stratification failed for validation split: {e}")
            print("Falling back to non-stratified split for validation")
            remaining_idx, val_idx = train_test_split(
                indices, 
                test_size=val_split,
                random_state=42
            )
    
    # Now sample training data from remaining indices if needed
    if train_split >= (1.0 - val_split):
        # Use all remaining data for training
        train_idx = remaining_idx
    else:
        # Calculate the proportion of remaining data to use for training
        remaining_proportion = train_split / (1.0 - val_split)
        
        try:
            # Sample training data from remaining indices
            train_idx, _ = train_test_split(
                remaining_idx,
                train_size=remaining_proportion,
                random_state=42,
                stratify=strata.numpy()[remaining_idx] if len(remaining_idx) > 0 else None
            )
        except ValueError as e:
            # Handle case where stratification fails (not enough samples per class)
            print(f"Warning: Stratification failed for training split: {e}")
            print("Falling back to non-stratified split for training")
            train_idx, _ = train_test_split(
                remaining_idx,
                train_size=remaining_proportion,
                random_state=42
            )
    
    # Convert to tensors and sort
    train_idx = torch.tensor(sorted(train_idx))
    val_idx = torch.tensor(sorted(val_idx))
    
    # Split the data
    train_data = torch.index_select(data, -1, train_idx)
    val_data = torch.index_select(data, -1, val_idx)
    
    # Split bvecs and bvals
    train_bvec = torch.index_select(bvec, 0, train_idx)
    val_bvec = torch.index_select(bvec, 0, val_idx)
    
    train_bval = torch.index_select(bval, 0, train_idx)
    val_bval = torch.index_select(bval, 0, val_idx)
    
    return (train_data, train_bvec, train_bval), \
           (val_data, val_bvec, val_bval)
    
