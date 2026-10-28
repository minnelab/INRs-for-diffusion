import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
import torchvision.transforms as transforms


def normalize_within_mask(image, mask=None):
    """Normalize image values within mask to range [0,1].
    
    Args:
        image (numpy.ndarray): Image to normalize
        mask (numpy.ndarray, optional): Binary mask. If None, normalize entire image.
        
    Returns:
        numpy.ndarray: Normalized image
    """
    if mask is None:
        # Normalize the entire image
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        return np.zeros_like(image)
    
    # Normalize only within mask
    normalized = np.zeros_like(image)
    mask_indices = mask > 0
    if mask_indices.any():
        masked_data = image[mask_indices]
        min_val = np.min(masked_data)
        max_val = np.max(masked_data)
        if max_val > min_val:
            normalized[mask_indices] = (masked_data - min_val) / (max_val - min_val)
    return normalized


def compute_image_metrics(pred_image, target_image, mask=None, device='cuda', lpips_model=None, compute_lpips=True, psnr_metric=None, ssim_metric=None):
    """Compute PSNR, SSIM, and LPIPS for images.

    Args:
        pred_image (torch.Tensor or numpy.ndarray): Predicted image (can be 2D or 3D)
        target_image (torch.Tensor or numpy.ndarray): Target image (can be 2D or 3D)
        mask (torch.Tensor or numpy.ndarray, optional): Binary mask
        device (torch.device, optional): Device to compute metrics on
        lpips_model (lpips.LPIPS, optional): Pre-initialized LPIPS model
        psnr_metric (PeakSignalNoiseRatio, optional): Pre-initialized PSNR metric
        ssim_metric (StructuralSimilarityIndexMeasure, optional): Pre-initialized SSIM metric

    Returns:
        tuple: (PSNR value, SSIM value, LPIPS value)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert numpy arrays to tensors if needed
    if isinstance(pred_image, np.ndarray):
        pred_tensor = torch.from_numpy(pred_image).float()
    else:
        pred_tensor = pred_image.float()

    if isinstance(target_image, np.ndarray):
        target_tensor = torch.from_numpy(target_image).float()
    else:
        target_tensor = target_image.float()

    # Apply mask if provided
    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask).float()
        else:
            mask_tensor = mask.float()

        pred_tensor = pred_tensor * mask_tensor
        target_tensor = target_tensor * mask_tensor

    # Use passed model if available, otherwise create a new one
    if lpips_model is None and compute_lpips:
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(device)

    # Initialize metrics only if not provided
    if psnr_metric is None:
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    else:
        psnr = psnr_metric

    if ssim_metric is None:
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    else:
        ssim = ssim_metric
    to_3channel = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    
    # Move tensors to device
    pred_tensor = pred_tensor.to(device)
    target_tensor = target_tensor.to(device)
    
    # Handle 3D volumes
    is_3d = pred_tensor.dim() == 3
    if is_3d:
        # For PSNR, we can flatten the volume
        pred_flat = pred_tensor[mask].reshape(1, -1)
        target_flat = target_tensor[mask].reshape(1, -1)
        psnr_val = psnr(pred_flat, target_flat)
        
        # For SSIM, we need to maintain spatial dimensions
        # Reshape to BxCxDxHxW format (adding batch and channel dimensions)
        pred_3d = pred_tensor.unsqueeze(0).unsqueeze(0)  # BxCxDxHxW
        target_3d = target_tensor.unsqueeze(0).unsqueeze(0)  # BxCxDxHxW
        ssim_val = ssim(pred_3d, target_3d)
        
        # For LPIPS, we need to compute it slice-wise
        num_slices = pred_tensor.shape[-1]
        lpips_vals = []
        if compute_lpips:
            for slice_idx in range(num_slices):
                # Get current slice
                pred_slice = pred_tensor[..., slice_idx].unsqueeze(0).unsqueeze(0)
                target_slice = target_tensor[..., slice_idx].unsqueeze(0).unsqueeze(0)
                
                # Prepare tensors for LPIPS (requires 3 channels)
                pred_3ch = to_3channel(pred_slice.squeeze(0))
                target_3ch = to_3channel(target_slice.squeeze(0))
                
                # Compute LPIPS for this slice
                with torch.no_grad():
                    lpips_val = lpips_model(pred_3ch, target_3ch)
                    lpips_vals.append(lpips_val.item())
            
            # Average LPIPS across slices
            lpips_val = np.mean(lpips_vals)
        else:
            lpips_val = 0.0
    else:
        # Handle 2D images (original code)
        # Ensure tensors have batch and channel dimensions for metrics
        if pred_tensor.dim() == 2:
            pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)
        elif pred_tensor.dim() == 3:
            pred_tensor = pred_tensor.unsqueeze(0)
            
        if target_tensor.dim() == 2:
            target_tensor = target_tensor.unsqueeze(0).unsqueeze(0)
        elif target_tensor.dim() == 3:
            target_tensor = target_tensor.unsqueeze(0)
        
        # Compute PSNR and SSIM
        psnr_val = psnr(pred_tensor, target_tensor)
        ssim_val = ssim(pred_tensor, target_tensor)
        
        # Prepare tensors for LPIPS (requires 3 channels)
        pred_3ch = to_3channel(pred_tensor.squeeze(0))
        target_3ch = to_3channel(target_tensor.squeeze(0))
        
        # Compute LPIPS
        with torch.no_grad():
            lpips_val = lpips_model(pred_3ch, target_3ch)
            lpips_val = lpips_val.item()
    
    return psnr_val.item(), ssim_val.item(), lpips_val


def create_comparison_figure(target_image, pred_image, title=None):
    """Generate a visualization comparing the predicted and target slices.
    
    Args:
        target_image (numpy.ndarray or torch.Tensor): Target image data
        pred_image (numpy.ndarray or torch.Tensor): Predicted image data
        title (str, optional): Figure title
            
    Returns:
        matplotlib.figure.Figure: Figure object containing the visualization
    """
    # Convert to numpy if needed
    if isinstance(target_image, torch.Tensor):
        target_image = target_image.detach().cpu().numpy()
    if isinstance(pred_image, torch.Tensor):
        pred_image = pred_image.detach().cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    im1 = ax1.imshow(target_image, cmap='gray')
    ax1.set_title('Target')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(pred_image, cmap='gray')
    ax2.set_title('Prediction')
    plt.colorbar(im2, ax=ax2)
    
    if title:
        plt.suptitle(title)
    
    return fig


def create_difference_figure(pred_image, target_image, title=None):
    """Generate a visualization of the absolute difference between predicted and target.
    
    Args:
        pred_image (numpy.ndarray or torch.Tensor): Predicted image data
        target_image (numpy.ndarray or torch.Tensor): Target image data
        title (str, optional): Figure title
            
    Returns:
        matplotlib.figure.Figure: Figure object containing the visualization
    """
    # Convert to numpy if needed
    if isinstance(target_image, torch.Tensor):
        target_image = target_image.detach().cpu().numpy()
    if isinstance(pred_image, torch.Tensor):
        pred_image = pred_image.detach().cpu().numpy()
    
    # Calculate absolute difference
    diff_image = np.abs(pred_image - target_image)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(diff_image, cmap='hot')
    ax.set_title('Absolute Difference')
    plt.colorbar(im, ax=ax)
    
    if title:
        plt.suptitle(title)
    
    return fig


def log_metrics_and_images(logger, prefix, pred_image, target_image, mask_slice, 
                          device, step, slice_idx=None, gradient_idx=None, bval=None):
    """Compute and log metrics and visualizations.
    
    Args:
        logger: PyTorch Lightning logger
        prefix (str): Prefix for logged metrics
        pred_image (numpy.ndarray): Predicted image
        target_image (numpy.ndarray): Target image
        mask_slice (numpy.ndarray): Binary mask for the slice
        device (torch.device): Device to compute metrics on
        step (int): Global step for logging
        slice_idx (int, optional): Index of the slice
        gradient_idx (int, optional): Index of the gradient direction
        bval (float, optional): b-value for the gradient direction
    """
    # Normalize images
    pred_norm = normalize_within_mask(pred_image, mask_slice)
    target_norm = normalize_within_mask(target_image, mask_slice)
    
    # Compute metrics
    psnr_val, ssim_val, lpips_val = compute_image_metrics(
        pred_norm, target_norm, mask_slice, device
    )
    
    # Create metric name suffix
    suffix = ""
    if bval is not None:
        # Round b-value to nearest 100
        rounded_bval = round(bval / 100) * 100
        suffix = f"_b{rounded_bval:d}"
    
    # Log metrics
    logger.experiment.add_scalar(f'{prefix}/PSNR{suffix}', psnr_val, step)
    logger.experiment.add_scalar(f'{prefix}/SSIM{suffix}', ssim_val, step)
    logger.experiment.add_scalar(f'{prefix}/LPIPS{suffix}', lpips_val, step)
    
    # Log images
    gradient_str = f"grad_{gradient_idx}" if gradient_idx is not None else "slice"
    bval_str = f"_b{rounded_bval:d}" if bval is not None else ""
    
    logger.experiment.add_image(
        f'{prefix}/predictions/{gradient_str}{bval_str}',
        pred_norm[np.newaxis, :, :],
        global_step=step,
        dataformats='CHW'
    )
    
    logger.experiment.add_image(
        f'{prefix}/targets/{gradient_str}{bval_str}',
        target_norm[np.newaxis, :, :],
        global_step=step,
        dataformats='CHW'
    )
    
    # Log difference image
    diff_image = np.abs(pred_norm - target_norm)
    logger.experiment.add_image(
        f'{prefix}/differences/{gradient_str}{bval_str}',
        diff_image[np.newaxis, :, :],
        global_step=step,
        dataformats='CHW'
    )
    
    return psnr_val, ssim_val, lpips_val


def create_multirow_visualization(images_dict, title="B-Value Comparison", metrics_dict=None):
    """Create a figure with three rows (target, prediction, difference) for multiple b-values.
    
    Args:
        images_dict (dict): Dictionary where keys are b-values and values are tuples of 
                           (target_image, pred_image, mask)
        title (str): Title for the figure
        metrics_dict (dict, optional): Dictionary where keys are b-values and values are tuples of
                                      (psnr, ssim, lpips) metrics for each b-value
        
    Returns:
        matplotlib.figure.Figure: Combined figure with all b-values
    """
    num_bvals = len(images_dict)
    if num_bvals == 0:
        return None
        
    # Create figure with 3 rows (target, prediction, difference) and columns for each b-value
    fig, axes = plt.subplots(3, num_bvals, figsize=(4*num_bvals, 10))
    
    # If only one b-value, axes will not be a 2D array
    if num_bvals == 1:
        axes = axes.reshape(3, 1)
    
    # Sort b-values for consistent display order
    bvals = sorted(images_dict.keys())
    
    for col, bval in enumerate(bvals):
        target_img, pred_img, mask = images_dict[bval]
        
        # Get metrics for this b-value if available
        metrics_str = ""
        if metrics_dict and bval in metrics_dict:
            psnr, ssim, lpips = metrics_dict[bval]
            metrics_str = f"\nPSNR: {psnr:.2f}, SSIM: {ssim:.3f}, LPIPS: {lpips:.3f}"
        
        # Normalize images within mask
        target_norm = normalize_within_mask(target_img, mask)
        pred_norm = normalize_within_mask(pred_img, mask) 
        diff_norm = np.abs(pred_norm - target_norm)
        
        # Row 0: Target images
        im0 = axes[0, col].imshow(target_norm, cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f"Target (b={bval})")
        axes[0, col].axis('off')
        
        # Row 1: Prediction images
        im1 = axes[1, col].imshow(pred_norm, cmap='gray', vmin=0, vmax=1)
        axes[1, col].set_title(f"Prediction (b={bval}){metrics_str}")
        axes[1, col].axis('off')
        
        # Row 2: Difference images
        im2 = axes[2, col].imshow(diff_norm, cmap='hot', vmin=0, vmax=0.5)
        axes[2, col].set_title(f"Difference (b={bval})")
        axes[2, col].axis('off')
    
    # Add colorbars for each row
    for row in range(3):
        plt.colorbar(im0 if row == 0 else im1 if row == 1 else im2, 
                    ax=axes[row, :].tolist(), shrink=0.8)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def log_multirow_bvalue_visualization(logger, prefix, images_dict, step, gradient_idx=None, metrics_dict=None):
    """Create and log a multi-row visualization for different b-values.
    
    Args:
        logger: PyTorch Lightning logger
        prefix (str): Prefix for logged metrics
        images_dict (dict): Dictionary where keys are b-values and values are tuples of 
                           (target_image, pred_image, mask)
        step (int): Global step for logging
        gradient_idx (int, optional): Index of the gradient direction
        metrics_dict (dict, optional): Dictionary where keys are b-values and values are tuples of
                                      (psnr, ssim, lpips) metrics for each b-value
    """
    # Create the figure
    title = f"B-Value Comparison (Gradient {gradient_idx})" if gradient_idx is not None else "B-Value Comparison"
    fig = create_multirow_visualization(images_dict, title=title, metrics_dict=metrics_dict)
    if fig is None:
        return
    
    # Convert figure to numpy array
    fig.canvas.draw()
    img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Log the figure
    gradient_str = f"grad_{gradient_idx}" if gradient_idx is not None else "all"
    logger.experiment.add_image(
        f'{prefix}/bvalue_comparison/{gradient_str}',
        img_data.transpose(2, 0, 1),  # Convert to CHW format
        global_step=step
    )
    
    plt.close(fig)


def init_lpips_model(device='cuda'):
    """Initialize LPIPS model once to avoid repeated loading"""
    import lpips
    lpips_model = lpips.LPIPS(net='alex').to(device)
    return lpips_model