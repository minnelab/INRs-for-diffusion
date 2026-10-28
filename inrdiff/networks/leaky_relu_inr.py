import torch
import torch.nn as nn
import numpy as np
from scipy.special import sph_harm

# (change here to omnit legacy warning)
# torch.set_default_tensor_type(torch.FloatTensor)


class PositionalEncoding(nn.Module):
    """
    Applies positional encoding to input coordinates using sinusoidal functions.
    
    This encoding maps input coordinates to a higher-dimensional space using
    a series of sine and cosine functions at different frequencies, which
    helps neural networks better represent high-frequency functions.
    
    Args:
        num_frequencies (int): Number of frequency bands to use for encoding.
                              Higher values enable capturing finer details.
    """
    def __init__(self, num_frequencies=6):
        super().__init__()
        self.num_frequencies = num_frequencies
    
    def forward(self, x):
        """
        Apply positional encoding to input coordinates.
        
        Args:
            x (torch.Tensor): Input coordinates of shape [..., D]
                             where D is the dimensionality of each point.
                             
        Returns:
            torch.Tensor: Encoded coordinates of shape [..., D + 2*D*num_frequencies]
                         containing original coordinates and their encodings.
        """
        frequencies = 2.0 ** torch.arange(self.num_frequencies, device=x.device)
        angles = x[..., None] * frequencies
        
        # Compute sines and cosines
        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)
        
        # Flatten and concatenate
        encoded = torch.cat([x, sin_features.flatten(-2), cos_features.flatten(-2)], dim=-1)
        return encoded


class SphericalHarmonicsEncoding(nn.Module):
    """
    Encodes 3D vectors using spherical harmonics basis functions.
    
    This module converts 3D vectors to spherical coordinates and computes
    their representation in the spherical harmonics basis, which is useful
    for encoding directional information in a rotation-invariant manner.
    
    Args:
        max_degree (int): Maximum degree of spherical harmonics to compute.
                         Higher values capture more angular detail.
    """
    def __init__(self, max_degree=4):
        super().__init__()
        self.max_degree = max_degree
    
    def cart2sph(self, x, y, z):
        """
        Convert from Cartesian (x,y,z) to spherical (r,theta,phi) coordinates.
        
        Args:
            x, y, z (torch.Tensor): Cartesian coordinates
            
        Returns:
            tuple: (r, theta, phi) where:
                r: radial distance
                theta: polar angle in [0, pi]
                phi: azimuthal angle in [-pi, pi]
        """
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.arccos(z / (r + 1e-8))  # polar angle in [0, pi]
        phi = torch.atan2(y, x)              # azimuth in [-pi, pi]
        return r, theta, phi
    
    def compute_sh(self, theta, phi):
        """
        Computes spherical harmonic basis up to self.max_degree.
        
        Args:
            theta (torch.Tensor): Polar angles in [0, pi]
            phi (torch.Tensor): Azimuthal angles in [-pi, pi]
            
        Returns:
            torch.Tensor: Spherical harmonic coefficients with shape [..., (max_degree+1)²]
            
        Note:
            For higher degrees, you may want a more robust SH library 
            or direct associated Legendre polynomial expansions.
        """
        coeffs = []
        for l in range(0, self.max_degree + 1):
            for m in range(-l, l + 1):
                x = torch.cos(theta)
                m_abs = abs(m)
                
                if l == 0:
                    p = torch.ones_like(x)
                elif l == 1:
                    if m_abs == 0:
                        p = x
                    else:
                        p = -torch.sqrt(1 - x**2)
                elif l == 2:
                    if m_abs == 0:
                        p = 0.5 * (3 * x**2 - 1)
                    elif m_abs == 1:
                        p = -3 * x * torch.sqrt(1 - x**2)
                    else:
                        p = 3 * (1 - x**2)
                else:
                    p = torch.cos(l * theta)

                # Normalization
                norm = torch.sqrt(torch.tensor((2*l + 1)/(4*np.pi), device=theta.device))
                
                if m < 0:
                    sh = norm * p * torch.sin(m_abs * phi)
                elif m == 0:
                    sh = norm * p
                else:
                    sh = norm * p * torch.cos(m * phi)
                
                coeffs.append(sh)
        
        return torch.stack(coeffs, dim=-1)
    
    def forward(self, qvec):
        """
        Compute spherical harmonic encoding for 3D vectors.
        
        Args:
            qvec (torch.Tensor): Input vectors of shape [..., 3], representing
                                q-vectors = sqrt(b)*bvec in diffusion MRI
                                
        Returns:
            torch.Tensor: Spherical harmonic coefficients with shape [..., (max_degree+1)²]
        """
        x, y, z = qvec[..., 0], qvec[..., 1], qvec[..., 2]
        r, theta, phi = self.cart2sph(x, y, z)
        
        # Compute spherical harmonics based on direction
        sh_coeffs = self.compute_sh(theta, phi)
        return sh_coeffs


class LeakyReLUMLP(nn.Module):
    """
    Implicit Neural Representation (INR) model for diffusion MRI signal reconstruction.
    
    This network takes spatial coordinates (x,y,z) and diffusion encoding parameters
    (bvec, bval) as input and predicts the diffusion MRI signal at those locations.
    The model uses positional encoding for spatial coordinates and spherical harmonics
    encoding for the diffusion directions to effectively represent the complex signal.
    
    Args:
        in_size (int): Input dimension (usually 7 for x,y,z,bvec_x,bvec_y,bvec_z,bval)
        out_size (int): Output dimension (usually 1 for signal prediction)
        hidden_size (int): Dimension of hidden layers
        num_layers (int): Number of hidden layers in the network
        position_frequencies (int): Number of frequency bands for positional encoding
        sh_max_degree (int): Maximum degree of spherical harmonics for direction encoding
        skip_connections (bool): Whether to use skip connections from input to all layers
    """
    def __init__(self, 
                 in_size,
                 out_size=1,
                 hidden_size=256, 
                 num_layers=3,
                 position_frequencies=6,
                 sh_max_degree=6,
                 skip_connections=True):
        super().__init__()
        
        self.in_size = in_size
        self.out_size = out_size
        self.skip_connections = skip_connections
        self.TWOPI = 2 * np.pi
        
        # Position encoding for spatial coordinates (x,y,z)
        self.position_encoding = PositionalEncoding(num_frequencies=position_frequencies)
        
        # Spherical harmonics encoding for q-space (direction)
        self.sh_encoding = SphericalHarmonicsEncoding(max_degree=sh_max_degree)
        
        # -- Compute input dimension --
        # 1) Spatial channels after encoding
        position_channels = 3 * (1 + 2 * position_frequencies)
        # 2) Spherical Harmonic channels for direction
        sh_channels = (sh_max_degree + 1)**2
        # 3) We'll include 1 channel for the radial magnitude ||q||
        #    If you want the raw b-value as well, add another channel.
        r_channels = 1
        
        input_dim = position_channels + sh_channels + r_channels
        
        self.input_layer = nn.Linear(input_dim, hidden_size)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_size + input_dim if skip_connections else hidden_size
            layer = nn.Sequential(
                nn.Linear(in_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.LeakyReLU()
            )
            self.hidden_layers.append(layer)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, out_size),
        )
        
    def forward(self, x):
        """
        Forward pass of the diffusion MRI INR model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [..., 7] where:
                - x[..., :3] -> spatial coordinates (x,y,z)
                - x[..., 3:4] -> b-value (diffusion weighting)
                - x[..., 4:7] -> diffusion gradient direction (unit vector)
                
        Returns:
            torch.Tensor: Predicted diffusion MRI signal of shape [..., out_size]
        """
        
        position = x[..., 0:3]
        bval     = x[..., 3:4]   # shape: [..., 1]
        bvec     = x[..., 4:7]
        

        # check if bval has nans
        if torch.isnan(bval).any():
            print(f"bval: {bval}")
            raise ValueError("bval contains NaNs")
        # check if bval is non-negative
        if (bval<0).any():
            print(f"bval: {bval}")
            raise ValueError("bval contains negative values")

        # check if bvec has nans
        if torch.isnan(bvec).any():
            print(f"bvec: {bvec}")
            raise ValueError("bvec contains NaNs")
        # Form the full q-vector by multiplying direction with sqrt(b)
        # (Ensure bval is non-negative, or clamp at zero if needed.)
        qvec = bvec * torch.sqrt(bval+1e-5)/self.TWOPI # shape: [..., 3]
        
        # check if nans
        if torch.isnan(qvec).any():
            print(f"qvec: {qvec}")
            print(f"bvec: {bvec}")
            print(f"bval: {(bval<0).any()}")
            raise ValueError("qvec contains NaNs")
        
        # Encode the spatial position
        encoded_position = self.position_encoding(position)
        
        # Encode the q-vector direction in spherical harmonics
        encoded_qdir = self.sh_encoding(qvec)
        
        # check if nans
        if torch.isnan(encoded_qdir).any():
            print(f"encoded_qdir: {encoded_qdir}")
            print(f"qvec: {qvec}")
            print(f"bvec: {bvec}")
            print(f"bval: {bval}")
            raise ValueError("encoded_qdir contains NaNs")
        
        # Also feed the radial magnitude ||q||
        r = torch.norm(qvec, dim=-1, keepdim=True)    # shape: [..., 1]
        
        # Combine all features (position encoding + SH direction + radial magnitude)
        encoded_input = torch.cat([encoded_position, encoded_qdir, bval], dim=-1).float()
        
        # check if nans
        if torch.isnan(encoded_input).any():
            print(f"encoded_input: {encoded_input}")
            print(f"encoded_position: {encoded_position}")
            print(f"encoded_qdir: {encoded_qdir}")
            print(f"bval: {bval}")
            raise ValueError("encoded_input contains NaNs")
        
        # Pass through the network
        features = self.input_layer(encoded_input)
        
        for layer in self.hidden_layers:
            if self.skip_connections:
                features = layer(torch.cat([features, encoded_input], dim=-1))
            else:
                features = layer(features)
        
        output = self.output_layer(features)
        return output


def get_model(config=None):
    """
    Helper function to create a LeakyReLUMLP model with default or custom configuration.
    
    Args:
        config (dict, optional): Configuration dictionary with model parameters.
                                If None, default parameters will be used.
                                
    Returns:
        LeakyReLUMLP: Instantiated model
        
    Example:
        >>> model = get_model({'in_size': 7, 'out_size': 1, 'hidden_size': 128, 'num_layers': 4})
    """
    if config is None:
        config = {
            'in_size': 7,
            'out_size': 1,
            'hidden_size': 256,
            'num_layers': 6,
            'position_frequencies': 6,
            'sh_max_degree': 4,
            'skip_connections': True
        }
    return LeakyReLUMLP(**config)
