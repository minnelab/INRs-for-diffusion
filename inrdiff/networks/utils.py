import torch

def xyz_to_angles(xyz):
    """adapted from https://github.com/e3nn/e3nn/blob/c370f49a9006230c1f428d00fe6826042b84bf35/e3nn/o3/_rotation.py#L672-L698"""

    xyz = torch.nn.functional.normalize(xyz, p=2.0, dim=-1)  # forward 0's instead of nan for zero-radius
    xyz = xyz.clamp(-1, 1)

    beta = torch.acos(xyz[..., 1])
    alpha = torch.atan2(xyz[..., 0], xyz[..., 2])
    
    return alpha, beta


def angles_to_xyz(alpha, beta) -> torch.Tensor:
    """adapted from https://github.com/e3nn/e3nn/blob/c370f49a9006230c1f428d00fe6826042b84bf35/e3nn/o3/_rotation.py#L615"""
    
    alpha, beta = torch.broadcast_tensors(alpha, beta)
    x = torch.sin(beta) * torch.sin(alpha)
    y = torch.cos(beta)
    z = torch.sin(beta) * torch.cos(alpha)
    
    return torch.stack([x, y, z], dim=-1)
