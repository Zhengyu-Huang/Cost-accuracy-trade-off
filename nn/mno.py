"""
Multiscale Neural Operator (MNO) Implementation for 1D and 2D Problems

This module implements the Multiscale Neural Operator architecture as described in 
"Multiscale Neural Operator for Parametric Partial Differential Equations" by Li et al.

Key concepts:
- Spectral convolution in Fourier space
- Learnable weight matrices for Fourier modes
- Combined with local convolution for better expressivity
"""


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.adam import Adam
from utility.losses import LpLoss
from utility.normalizer import UnitGaussianNormalizer
from timeit import default_timer
from typing import Literal

# ============================================================================
# Utility Functions
# ============================================================================

def add_padding(x, pad_nums):
    """
    Add zero padding to the spatial dimensions of the input tensor.
    
    Args:
        x: Input tensor of shape (batch, channels, spatial_dims...)
        pad_nums: List of padding amounts for each spatial dimension
                  (ordered from last dimension to first in F.pad convention)
    
    Returns:
        Padded tensor with same number of dimensions
    """
    
    if x.ndim == 3:  # fourier1d
        res = F.pad(x, [0, pad_nums[0]], "constant", 0)
    elif x.ndim == 4:  # fourier2d
        res = F.pad(x, [0, pad_nums[1], 0, pad_nums[0]], "constant", 0)
    elif x.ndim == 5:  # fourier3d
        res = F.pad(x, [0, pad_nums[2], 0, pad_nums[1], 0, pad_nums[0]], "constant", 0)
    elif x.ndim == 6:  # fourier4d
        res = F.pad(
            x,
            [0, pad_nums[3], 0, pad_nums[2], 0, pad_nums[1], 0, pad_nums[0]],
            "constant",
            0,
        )
    else:
        print("error : x.ndim = ", x.ndim)

    return res


def remove_padding(x, pad_nums):
    """
    Remove zero padding from the spatial dimensions of the input tensor.
    
    Args:
        x: Input tensor of shape (batch, channels, spatial_dims...)
        pad_nums: List of padding amounts for each spatial dimension
    
    Returns:
        Tensor with padding removed
    """
    if x.ndim == 3:  # fourier1d
        res = x[..., : (None if pad_nums[0] == 0 else -pad_nums[0])]

    elif x.ndim == 4:  # fourier2d
        res = x[
            ...,
            : (None if pad_nums[0] == 0 else -pad_nums[0]),
            : (None if pad_nums[1] == 0 else -pad_nums[1]),
        ]

    elif x.ndim == 5:  # fourier3d
        res = x[
            ...,
            : (None if pad_nums[0] == 0 else -pad_nums[0]),
            : (None if pad_nums[1] == 0 else -pad_nums[1]),
            : (None if pad_nums[2] == 0 else -pad_nums[2]),
        ]

    elif x.ndim == 6:  # fourier4d
        res = x[
            ...,
            : (None if pad_nums[0] == 0 else -pad_nums[0]),
            : (None if pad_nums[1] == 0 else -pad_nums[1]),
            : (None if pad_nums[2] == 0 else -pad_nums[2]),
            : (None if pad_nums[3] == 0 else -pad_nums[3]),
        ]

    else:
        print("error : x.ndim = ", x.ndim)

    return res


def _get_act(act):
    """
    Get activation function by name.
    
    Args:
        act: String name of activation function. Options:
             'tanh', 'gelu', 'relu', 'elu', 'leaky_relu', 'none'
    
    Returns:
        Activation function callable or None
    """
    if act == "tanh":
        func = F.tanh
    elif act == "gelu":
        func = F.gelu
    elif act == "relu":
        func = F.relu_
    elif act == "elu":
        func = F.elu_
    elif act == "leaky_relu":
        func = F.leaky_relu_
    elif act == "softsign":
        func = F.softsign
    elif act == "none":
        func = None
    else:
        raise ValueError(f"{act} is not supported")
    return func



@torch.jit.script
def compl_mul1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Complex-valued matrix multiplication for 1D Fourier modes.
    
    Args:
        a: Input tensor of shape (batch, in_channels, modes)
        b: Weight tensor of shape (in_channels, out_channels, modes)
    
    Returns:
        Output tensor of shape (batch, out_channels, modes)
    
    Note:
        The multiplication is element-wise along the mode dimension,
        which corresponds to the convolution theorem in Fourier space.
    """
    # Einstein summation: bix, iox -> box
    # b: batch, i: in_channels, x: modes
    # i: in_channels, o: out_channels, x: modes
    res = torch.einsum("bix,iox->box", a, b)
    return res


@torch.jit.script
def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Complex-valued matrix multiplication for 2D Fourier modes.
    
    Args:
        a: Input tensor of shape (batch, in_channels, modes1, modes2)
        b: Weight tensor of shape (in_channels, out_channels, modes1, modes2)
    
    Returns:
        Output tensor of shape (batch, out_channels, modes1, modes2)
    """
    # Einstein summation: bixy, ioxy -> boxy
    res = torch.einsum("bixy,ioxy->boxy", a, b)
    return res


# ============================================================================
# Spectral Convolution Layers
# ============================================================================

class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution Layer (Fourier layer)
    
    This layer performs:
    1. Real FFT of the input to Fourier space
    2. Linear transformation of Fourier coefficients (learnable)
    3. Inverse FFT back to physical space
    
    The linear transformation in Fourier space corresponds to convolution
    in physical space via the Convolution Theorem.
    """
    def __init__(self, in_channels, out_channels, modes1):
        """
        Initialize 1D Spectral Convolution layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1: Number of Fourier modes to retain (k_max)
                   Should be <= floor(N/2) + 1 where N is spatial resolution
        """
        
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        # Initialize weights with scaling to maintain variance
        # Shape: (in_channels, out_channels, modes1) - complex-valued
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def forward(self, x):
        """
        Forward pass of 1D spectral convolution.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, spatial_dim)
        
        Returns:
            Output tensor of shape (batch_size, out_channels, spatial_dim)
        """
        batchsize = x.shape[0]
        
        # Step 1: Real FFT along the spatial dimension
        # rfftn computes FFT for real inputs, returns only positive frequencies
        # Shape: (batch, in_channels, spatial_dim//2 + 1)
        x_ft = torch.fft.rfftn(x, dim=[2])

        # Step 2: Apply learnable weights to Fourier coefficients
        # Initialize output Fourier tensor with zeros
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,    # Number of positive frequencies
            device=x.device,
            dtype=torch.cfloat,
        )
        
        # Multiply the first `modes1` Fourier coefficients with weights
        # This is the core spectral operation
        out_ft[:, :, : self.modes1] = compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )
        # Note: Higher frequencies (modes1 to end) are set to zero
        # This acts as a low-pass filter

        # Step 3: Inverse FFT back to physical space
        # irfftn expects the full set of positive frequencies
        x = torch.fft.irfftn(out_ft, s=[x.size(-1)], dim=[2])
        return x


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution Layer (Fourier layer)
    
    This layer performs convolution via multiplication in Fourier space.
    For 2D, we keep modes in both dimensions and use two sets of weights
    for symmetric handling of positive and negative frequencies.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Initialize 2D Spectral Convolution layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1: Number of Fourier modes to retain in first dimension (height)
            modes2: Number of Fourier modes to retain in second dimension (width)
        """
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        
        # Initialize two sets of weights:
        # - weights1: for positive frequencies (0 to modes1, 0 to modes2)
        # - weights2: for negative frequencies (-modes1 to -1, 0 to modes2)
        # This accounts for the symmetry of real-valued FFT
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    def forward(self, x):
        """
        Forward pass of 2D spectral convolution.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        batchsize = x.shape[0]
                
        # Step 1: Real FFT along spatial dimensions (height, width)
        # Shape: (batch, in_channels, height, width//2 + 1)
        x_ft = torch.fft.rfftn(x, dim=[2, 3])

        # Step 2: Apply learnable weights
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),           # height dimension
            x.size(-1) // 2 + 1,  # width dimension (positive frequencies)
            device=x.device,
            dtype=torch.cfloat,
        )
        
        # Handle positive frequencies: indices [0:modes1, 0:modes2]
        out_ft[:, :, : self.modes1, : self.modes2] = compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        # Handle negative frequencies: indices [-modes1:, 0:modes2]
        # These correspond to frequencies with negative wavenumber in the first dimension
        out_ft[:, :, -self.modes1 :, : self.modes2] = compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Step 3: Inverse FFT back to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[2, 3])
        return x
    


class GradientLayer1d(nn.Module):
    """
    Gradient Operator Approximation.
    
    This module computes the nodal gradients of a field using the finite difference 
    approximation and maps them to the output channel space.

    box_neighbor_average is applied after the gradient computation to remove oscillations
    
    Logic: Output = W_grad1( geo_act( W_grad2(compute_gradient(x) ) ))
    """
    def __init__(self, in_channels, out_channels, dx1, geo_act='softsign'):
        super(GradientLayer1d, self).__init__()
        self.gw1 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.gw2 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.register_buffer("dx1", torch.tensor(float(dx1)))
        self.geo_act = _get_act(geo_act)

    def compute_gradient(self, x):
        '''
        Assume periodic boundary condition, compute finite difference gradient
        Input:
            x: Input tensor of shape (batch_size, in_channels, spatial_dim)
        Returns:
            Output tensor of shape (batch_size, in_channels, spatial_dim)
        '''
        x_forward = torch.roll(x, shifts=-1, dims=-1)
        x_backward = torch.roll(x, shifts=1, dims=-1)

        return (x_forward - x_backward) / (2.0 * self.dx1)

    def forward(self, x):
        '''
        Input:
            x: Input tensor of shape (batch_size, in_channels, spatial_dim)
        Returns:
            Output tensor of shape (batch_size, out_channels, spatial_dim)
        '''
        return self.gw1(self.geo_act(box_neighbor_average(self.gw2(self.compute_gradient(x)))))
    


class GradientLayer2d(nn.Module):
    """
    Gradient Operator Approximation.
    
    This module computes the nodal gradients of a field using the finite difference 
    approximation and maps them to the output channel space.

    box_neighbor_average is applied after the gradient computation to remove oscillations
    
    Logic: Output = W_grad1( geo_act( W_grad2(compute_gradient(x) ) ))
    """
    def __init__(self, in_channels, out_channels, dx1, dx2, geo_act='softsign'):
        super(GradientLayer2d, self).__init__()
        self.gw1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.gw2 = nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False)
        self.register_buffer("dx1", torch.tensor(float(dx1)))
        self.register_buffer("dx2", torch.tensor(float(dx2)))
        self.geo_act = _get_act(geo_act)


    def compute_gradient(self, x):
        # Derivative along x-direction, dimension -2.
        grad_x1 = (
            torch.roll(x, shifts=-1, dims=-2)
            - torch.roll(x, shifts=1, dims=-2)
        ) / (2.0 * self.dx1)

        # Derivative along y-direction, dimension -1.
        grad_x2 = (
            torch.roll(x, shifts=-1, dims=-1)
            - torch.roll(x, shifts=1, dims=-1)
        ) / (2.0 * self.dx2)

        # Concatenate gradients along the channel dimension.
        # Shape: (batch_size, 2 * in_channels, nx, ny)
        grad = torch.cat([grad_x1, grad_x2], dim=1)

        return grad
    
    def forward(self, x):
        '''
        Input:
            x: float[batch_size, in_channels, nx, ny]
        Return:
            float[batch_size, out_channels, nx, ny]
        '''
        return self.gw1(self.geo_act(box_neighbor_average(self.gw2(self.compute_gradient(x)))))
# ============================================================================
# Multiscale Neural Operator (MNO) Models
# ============================================================================


def box_neighbor_average(
    x: torch.Tensor,
    iterations: int = 1
) -> torch.Tensor:
    """
    Repeated self-plus-neighbor box averaging on a structured mesh
    with periodic boundary conditions.

    Each iteration replaces every node by the average over its
    3^ndim local box neighborhood, including itself.


    Parameters
    ----------
    x : Tensor[batch_size, n_channels, nx, ny, ...]
        Input node features.

    iterations : int, default=1
        Number of averaging iterations.

    Returns
    -------
    Tensor[batch_size, n_channels, nx, ny, ...]
        Smoothed node features.

    Notes
    -----
    This is a special case of graph smoothing with uniform weights and implicit self-inclusion.
    """

    ndim = x.dim() - 2

    pool_fn = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}[ndim]
    pad_width = [1] * (2 * ndim)
    out = x
    for _ in range(iterations):
        # Apply circular padding so that neighbours wrap around
        # padding = 1 in each spatial dimension (both sides)
         
        out = F.pad(out, pad=pad_width, mode='circular')

        # Perform convolution (valid region corresponds to original size)
        # out = conv_fn(out, kernel, groups=n_channels)
        out = pool_fn(out, kernel_size=3, stride=1)

    return out




class MNO1d(nn.Module):
    """
    1D Multiscale Neural Operator
    
    Architecture:
    - Input lifting: Linear layer to expand to channel dimension
    - Multiple Fourier layers: Each combines spectral convolution + local convolution
    - Output projection: Linear layers to map to desired output dimension
    
    The operator learns mappings between infinite-dimensional function spaces
    by parameterizing the integral kernel in Fourier space.
    """
    def __init__(self,
                 layers=[64,64,64,64],
                 fc_dim=64,
                 in_dim=2, out_dim=1,
                 act='gelu',
                 modes=[16,16,16],
                 grad_layer=True,
                 geo_act='softsign', dx1=0.5, 
                 pad_ratio=0, 
                 cnn_kernel_size=1,
                 incremental=False):
        """
        Initialize 1D MNO.
        
        Args:
            modes: Number of Fourier modes to retain
            width: Channel width (used if layers is None)
            layers: List of channel sizes for each Fourier layer.
                   If None, uses [width] * 4 (4 layers of same width)
            fc_dim: Dimension of fully connected hidden layer. If <=0, skip FC layer.
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            act: Activation function name
            pad_ratio: Fraction of input size to pad (helps with periodic boundary approximation)
            cnn_kernel_size: Kernel size for local (W) convolution
        """
        super(MNO1d, self).__init__()

        # Set up layer sizes
        self.layers = layers
        self.pad_ratio = pad_ratio
        self.fc_dim = fc_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grad_layer = grad_layer
        # When incremental is true, x + NN(x), do not use normalizer
        self.incremental = incremental
        # Layer 1: Lift input to higher-dimensional channel space
        # Input shape: (batch, spatial_points, in_dim)
        # Output shape: (batch, spatial_points, layers[0])
        self.fc0 = nn.Linear(in_dim, layers[0])  # input channel is 2: (a(x), x)

        # Fourier layers: Each contains a spectral convolution and local convolution
        # The combination (W + K) allows both global (spectral) and local patterns
        self.modes1 = modes
        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, num_modes) for in_size, out_size, num_modes in zip(layers, layers[1:], self.modes1)])

        # Local convolution layers (W operator) - captures local features
        # Using padding='same' via manual padding to preserve spatial dimensions
        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, kernel_size=cnn_kernel_size, padding=(cnn_kernel_size//2))
                                 for in_size, out_size in zip(layers, layers[1:])])
        
        # Gradient layers - captures local features
        self.grad_layers = nn.ModuleList(
            [
                GradientLayer1d(in_size, out_size, dx1=dx1, geo_act=geo_act)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        ) if grad_layer else [None]*len(self.layers[1:])

        # Output projection layers
        # if fc_dim = 0, we do not have nonlinear layer
        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim) 
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)
            
        self.act = _get_act(act)

    def forward(self, x):
        """
        Forward pass of 1D MNO.
        
        Args:
            x: Input tensor of shape (batch_size, nx_in, in_dim)
               Typically contains (function values, coordinates)
        
        Returns:
            Output tensor of shape (batch_size, nx_out, out_dim)
        """        
        if self.incremental:
            x0 = x[...,0:self.out_dim]
        
        length = len(self.ws)
                
        # Step 1: Lift input to channel space
        # Shape: (batch, nx, in_dim) -> (batch, nx, layers[0])
        x = self.fc0(x)
            
        # Step 2: Permute to channel-first format for convolution
        # (batch, nx, channels) -> (batch, channels, nx)
        x = x.permute(0, 2, 1)
        
        # Step 3: Apply padding (if specified)
        # Padding helps mitigate boundary effects when domain isn't perfectly periodic
        pad_nums = [math.floor(self.pad_ratio * x.shape[-1])]
        x = add_padding(x, pad_nums=pad_nums)

        # Step 4: Apply Fourier layers
        for i, (speconv, w, grad_layer) in enumerate(zip(self.sp_convs, self.ws, self.grad_layers)):
            # Spectral convolution (K operator) - captures global patterns
            x1 = speconv(x)
            # Local convolution (W operator) - captures local patterns
            x2 = w(x)
            # Gradient layer
            x3 = grad_layer(x) if self.grad_layer else 0

            # Apply activation (except after last layer)
            if self.act is not None and i != length - 1:
                x = x + self.act(x1 + x2 + x3)
            else:
                x = x1 + x2 + x3
                
        # Step 5: Remove padding
        x = remove_padding(x, pad_nums=pad_nums)
        
        # Step 6: Permute back to channel-last format
        # (batch, channels, nx) -> (batch, nx, channels)
        x = x.permute(0, 2, 1)

        # Step 7: Project to output dimension
        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)
            
        x = self.fc2(x)
        
        if self.incremental:
            x += x0
        return x



class MNO2d(nn.Module):
    """
    2D Multiscale Neural Operator
    
    Architecture similar to MNO1d but for 2D spatial domains.
    Handles input of the form (a(x,y), x, y) and outputs solution u(x,y).
    """
    def __init__(
        self,
        layers=[64,64,64,64],
        fc_dim=64,
        in_dim=3,
        out_dim=1,
        act="gelu",
        modes1 = [16,16,16], modes2 = [16,16,16],
        grad_layer=True,
        geo_act='softsign', dx1=0.5, dx2=0.5,
        pad_ratio=0,
        cnn_kernel_size=1,
        incremental=False):
        
        super(MNO2d, self).__init__()

        """
        Initialize 2D MNO.
        
        Args:
            modes1: Number of Fourier modes in height dimension
            modes2: Number of Fourier modes in width dimension
            width: Channel width (used if layers is None)
            layers: List of channel sizes for each Fourier layer
            fc_dim: Dimension of fully connected hidden layer. If <=0, skip FC layer.
            in_dim: Input feature dimension (typically 3: coefficient + x + y)
            out_dim: Output feature dimension
            act: Activation function name
            pad_ratio: Fraction of input size to pad (helps with boundary approximation)
            cnn_kernel_size: Kernel size for local (W) convolution
        """

        
        # Set up layer sizes
        self.layers = layers
        self.pad_ratio = pad_ratio
        self.fc_dim = fc_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grad_layer = grad_layer
        # When incremental is true, x + NN(x), do not use normalizer
        self.incremental = incremental
        
        # Layer 1: Lift input to channel space
        # Input shape: (batch, height, width, in_dim)
        self.fc0 = nn.Linear(in_dim, layers[0])

        # Fourier layers
        self.modes1 = modes1
        self.modes2 = modes2
        self.sp_convs = nn.ModuleList(
            [
                SpectralConv2d(in_size, out_size, mode1_num, mode2_num)
                for in_size, out_size, mode1_num, mode2_num in zip(
                    self.layers, self.layers[1:], self.modes1, self.modes2
                )
            ]
        )
        # Local convolution layers (2D Conv2d)
        self.ws = nn.ModuleList(
            [
                nn.Conv2d(in_size, out_size, kernel_size=(cnn_kernel_size,cnn_kernel_size), padding=(cnn_kernel_size//2,cnn_kernel_size//2))
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        # Gradient layers - captures local features
        self.grad_layers = nn.ModuleList(
            [
                GradientLayer2d(in_size, out_size, dx1=dx1, dx2=dx2, geo_act=geo_act)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        ) if grad_layer else [None]*len(self.layers[1:])

        # Output projection
        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)

        self.act = _get_act(act)

    def forward(self, x):
        """
        Forward pass of 2D MNO.
        
        Args:
            x: Input tensor of shape (batch_size, height, width, in_dim)
               Typically contains (coefficient function a(x,y), x, y)
        
        Returns:
            Output tensor of shape (batch_size, height, width, out_dim)
        """
        if self.incremental:
            x0 = x[...,0:self.out_dim]
            
                    
        length = len(self.ws)

        # Step 1: Lift input to channel space
        # (batch, height, width, in_dim) -> (batch, height, width, layers[0])
        x = self.fc0(x)
        
        # Step 2: Permute to channel-first format
        # (batch, height, width, channels) -> (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        # Step 3: Apply padding
        pad_nums = [
            math.floor(self.pad_ratio * x.shape[-2]),
            math.floor(self.pad_ratio * x.shape[-1]),
        ]
        x = add_padding(x, pad_nums=pad_nums)

        # Step 4: Apply Fourier layers

        for i, (speconv, w, grad_layer) in enumerate(zip(self.sp_convs, self.ws, self.grad_layers)):
            x1 = speconv(x)
            x2 = w(x)
            x3 = grad_layer(x) if self.grad_layer else 0

            # Apply activation (except after last layer)
            if self.act is not None and i != length - 1:
                x = x + self.act(x1 + x2 + x3)
            else:
                x = x1 + x2 + x3


        # Step 5: Remove padding
        x = remove_padding(x, pad_nums=pad_nums)

        # Step 6: Permute back to channel-last format
        # (batch, channels, height, width) -> (batch, height, width, channels)
        x = x.permute(0, 2, 3, 1)

        # Step 7: Project to output
        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)
        
        if self.incremental:
            x += x0
        return x




# ============================================================================
# Training Function
# ============================================================================
def MNO_train(x_train, y_train, x_test, y_test, config, model, save_model_name="./MNO_model"):
    """
    Training function for MNO models.
    
    This function handles:
    - Data normalization
    - Training loop with configurable optimizer and scheduler
    - Evaluation on test set
    - Model checkpointing
    
    Args:
        x_train: Training input data (n_train, n_x, in_dims)
        y_train: Training target data (n_train, n_x, out_dims)
        x_test: Test input data (n_test, n_x, in_dims)
        y_test: Test target data (n_test, n_x, out_dims)
        config: Dictionary containing training configuration with keys:
            - train: sub-dictionary with:
                - normalization_x, normalization_y: bool for data normalization
                - normalization_dim_x, normalization_dim_y: dimensions to normalize
                - non_normalized_dim_x, non_normalized_dim_y: dimensions to skip
                - batch_size: Training batch size
                - base_lr: Learning rate
                - weight_decay: Weight decay for optimizer
                - scheduler: Learning rate scheduler (currently supports 'OneCycleLR')
                - epochs: Number of training epochs
        model: MNO model instance to train
        save_model_name: Path to save model checkpoints (without extension)
    
    Returns:
        train_rel_lp_losses: List of relative Lp training losses per epoch
        test_rel_lp_losses: List of relative Lp test losses per epoch
        test_lp_losses: List of absolute Lp test losses per epoch
    """
    
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_lp_losses = []
    test_rel_lp_losses = []
    test_lp_losses = []
    normalization_x, normalization_y = config["train"]["normalization_x"], config["train"]["normalization_y"]
    normalization_dim_x, normalization_dim_y = config["train"]["normalization_dim_x"], config["train"]["normalization_dim_y"]
    non_normalized_dim_x, non_normalized_dim_y = config["train"]["non_normalized_dim_x"], config["train"]["non_normalized_dim_y"]
    loss_p = config["train"]["loss_p"]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data normalization (important for stable training)
    if normalization_x:
        print("normalization x")
        x_normalizer = UnitGaussianNormalizer(x_train, non_normalized_dim = non_normalized_dim_x, normalization_dim=normalization_dim_x)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)
        
    if normalization_y:
        print("normalization y")
        y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim = non_normalized_dim_y, normalization_dim=normalization_dim_y)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                               batch_size=config['train']['batch_size'], shuffle=False)
    
    
    # Setup optimizer
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'], weight_decay=config['train']['weight_decay'])
    
    # Setup learning rate scheduler
    if config["train"]["scheduler"] == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config['train']['base_lr'], 
            div_factor=2, final_div_factor=100,pct_start=0.2,
            steps_per_epoch=len(train_loader), epochs=config['train']['epochs'])
    else:
        print("Scheduler ", config['train']['scheduler'], " has not implemented.")

    model.train()
    
    # Loss function: Relative Lp loss
    myloss = LpLoss(d=1, p=loss_p, size_average=False)
    epochs = config['train']['epochs']

    # Training loop
    for ep in range(epochs):
        t1 = default_timer()
        train_rel_lp = 0

        # Training phase
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x) 
            # Decode if normalized
            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
            loss.backward()

            optimizer.step()
            scheduler.step()
            train_rel_lp += loss.item()

        # Evaluation phase
        test_lp = 0
        test_rel_lp = 0
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                batch_size_ = x.shape[0]
                out = model(x) 

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                test_rel_lp += myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
                test_lp += myloss.abs(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()




        
        # Average losses over dataset size
        train_rel_lp/= n_train
        test_lp /= n_test
        test_rel_lp/= n_test
        
        # Store losses
        train_rel_lp_losses.append(train_rel_lp)
        test_rel_lp_losses.append(test_rel_lp)
        test_lp_losses.append(test_lp)
    
        t2 = default_timer()
        
        # Print progress
        print("Epoch : ", ep, " Time: ", round(t2-t1,3), " Rel. Train L" + str(int(loss_p)) + " Loss : ", train_rel_lp, " Rel. Test L" + str(int(loss_p)) + " Loss : ", test_rel_lp, " Test L" + str(int(loss_p)) + " Loss : ", test_lp, flush=True)
        
        # Save checkpoint every 100 epochs and at final epoch
        if (ep %100 == 99) or (ep == epochs -1):    
            if save_model_name:
                torch.save(model.state_dict(), save_model_name + ".pth")
                if normalization_x:
                    torch.save(x_normalizer.state_dict(), save_model_name + "_normalization_x.pth")
                if normalization_y:
                    torch.save(y_normalizer.state_dict(), save_model_name + "_normalization_y.pth")
    
    
    return train_rel_lp_losses, test_rel_lp_losses, test_lp_losses


# ============================================================================
# Training Function
# ============================================================================
def MNO_recurrent_train(x_train, y_train, x_test, y_test, n_step, config, model, save_model_name="./MNO_model"):
    """
    Training function for MNO models for time dependent problems.
    
    This function handles:
    - Data normalization
    - Training loop with configurable optimizer and scheduler
    - Evaluation on test set
    - Model checkpointing
    
    Args:
        x_train: Training input data (n_train, n_x, in_dims)
        y_train: Training target data (n_train, n_x, out_dims * n_step)
        x_test: Test input data (n_test, n_x, in_dims)
        y_test: Test target data (n_test, n_x, out_dims * n_step)
        n_step: int recurrent training for n_steps
        config: Dictionary containing training configuration with keys:
            - train: sub-dictionary with:
                - normalization_x, normalization_y: bool for data normalization
                - normalization_dim_x, normalization_dim_y: dimensions to normalize
                - non_normalized_dim_x, non_normalized_dim_y: dimensions to skip
                - batch_size: Training batch size
                - base_lr: Learning rate
                - weight_decay: Weight decay for optimizer
                - scheduler: Learning rate scheduler (currently supports 'OneCycleLR')
                - epochs: Number of training epochs
        model: MNO model instance to train
        save_model_name: Path to save model checkpoints (without extension)
    
    Returns:
        train_rel_lp_losses: List of relative Lp training losses per epoch
        test_rel_lp_losses: List of relative Lp test losses per epoch
        test_lp_losses: List of absolute Lp test losses per epoch
    """
    in_dim, out_dim = x_train.shape[-1], y_train.shape[-1] // n_step
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_lp_losses = []
    test_rel_lp_losses = []
    test_lp_losses = []
    normalization_x, normalization_y = config["train"]["normalization_x"], config["train"]["normalization_y"]
    normalization_dim_x, normalization_dim_y = config["train"]["normalization_dim_x"], config["train"]["normalization_dim_y"]
    non_normalized_dim_x, non_normalized_dim_y = config["train"]["non_normalized_dim_x"], config["train"]["non_normalized_dim_y"]
    loss_p = config["train"]["loss_p"]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    if normalization_x:
        print("normalization x")
        x_normalizer = UnitGaussianNormalizer(x_train, non_normalized_dim = non_normalized_dim_x, normalization_dim=normalization_dim_x)
        x_normalizer.to(device)
        
    if normalization_y:
        print("normalization y")
        y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim = non_normalized_dim_y, normalization_dim=normalization_dim_y)
        y_normalizer.to(device)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                               batch_size=config['train']['batch_size'], shuffle=False)
    
    
    # Setup optimizer
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'], weight_decay=config['train']['weight_decay'])
    
    # Setup learning rate scheduler
    if config["train"]["scheduler"] == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config['train']['base_lr'], 
            div_factor=2, final_div_factor=100,pct_start=0.2,
            steps_per_epoch=len(train_loader), epochs=config['train']['epochs'])
    else:
        print("Scheduler ", config['train']['scheduler'], " has not implemented.")
    
    # Loss function: Relative Lp loss
    myloss = LpLoss(d=1, p=loss_p, size_average=False)
    epochs = config['train']['epochs']


    # Training loop
    for ep in range(epochs):
        t1 = default_timer()
        train_rel_lp = 0

        # Training phase
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            
            x_pred = x.clone()
            loss = 0.0
            for i in range(n_step):
                y_pred = model(x=x_normalizer(x_pred) if normalization_x else x_pred)
                if normalization_y:
                    y_pred = y_normalizer.decode(y_pred)
                y_target = y[..., i * out_dim:(i + 1) * out_dim]
                loss = loss + myloss(
                    y_pred.reshape(batch_size_, -1),
                    y_target.reshape(batch_size_, -1),
                ) / n_step
                x_pred = torch.cat([y_pred, x[..., out_dim:]], dim=-1) if in_dim > out_dim else y_pred
            loss.backward()

            optimizer.step()
            scheduler.step()
            train_rel_lp += loss.item()

        # Evaluation phase
        test_lp = 0
        test_rel_lp = 0
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                batch_size_ = x.shape[0]
                
                x_pred = x.clone()
                for i in range(n_step):
                    y_pred = model(x = x_normalizer(x_pred) if normalization_x else x_pred) 
                    if normalization_y:
                        y_pred = y_normalizer.decode(y_pred)
                    y_target = y[..., i * out_dim:(i + 1) * out_dim]
                    test_rel_lp += myloss(
                        y_pred.reshape(batch_size_, -1),
                        y_target.reshape(batch_size_, -1),
                    ).item() / n_step
                    test_lp += myloss.abs(
                        y_pred.reshape(batch_size_, -1),
                        y_target.reshape(batch_size_, -1),
                    ).item() / n_step
                    x_pred = torch.cat([y_pred, x[..., out_dim:]], dim=-1) if in_dim > out_dim else y_pred




        
        # Average losses over dataset size
        train_rel_lp/= n_train
        test_lp /= n_test
        test_rel_lp/= n_test
        
        # Store losses
        train_rel_lp_losses.append(train_rel_lp)
        test_rel_lp_losses.append(test_rel_lp)
        test_lp_losses.append(test_lp)
    
        t2 = default_timer()
        
        # Print progress
        print("Epoch : ", ep, " Time: ", round(t2-t1,3), " Rel. Train L" + str(int(loss_p)) + " Loss : ", train_rel_lp, " Rel. Test L" + str(int(loss_p)) + " Loss : ", test_rel_lp, " Test L" + str(int(loss_p)) + " Loss : ", test_lp, flush=True)
        
        # Save checkpoint every 100 epochs and at final epoch
        if (ep %100 == 99) or (ep == epochs -1):    
            if save_model_name:
                torch.save(model.state_dict(), save_model_name + ".pth")
                if normalization_x:
                    torch.save(x_normalizer.state_dict(), save_model_name + "_normalization_x.pth")
                if normalization_y:
                    torch.save(y_normalizer.state_dict(), save_model_name + "_normalization_y.pth")
    
    
    return train_rel_lp_losses, test_rel_lp_losses, test_lp_losses


def setup_model(in_dim, out_dim, fc_dim, k_max, n_layer, 
                grad_layer, dxs, dx_scale, pad_ratio, incremental=False, checkpoint_path = None):
    """
    Instantiate a MNO (Multi-channel Neural Operator) model for 1D or 2D problems,
    optionally loading pre-trained weights from a checkpoint.
    
    Args:
        in_dim: Number of input channels
        out_dim: Number of output channels
        fc_dim: Hidden dimension of fully-connected layers
        k_max: Number of Fourier modes to keep (same for all layers and dimensions)
        n_layer: Number of MNO layers
        dxs: List/tuple of grid spacings (e.g., [dx1] for 1D, [dx1, dx2] for 2D)
        dx_scale: Scaling factor applied to each dx
        pad_ratio: Padding ratio for spectral convolution
        incremental: Whether the output adds the input
        checkpoint_path: Optional path to a .pth file containing model state_dict
    
    Returns:
        model: Instantiated MNO1d or MNO2d model (or None if dim > 2)
    """
    dim = len(dxs)
    if dim == 1:
        model = MNO1d(
               layers=[fc_dim]*(n_layer+1),
               fc_dim=fc_dim,
               in_dim=in_dim, out_dim=out_dim,
               act='gelu',
               modes = [k_max]*n_layer,
               grad_layer = grad_layer,
               geo_act='softsign', dx1=dxs[0]*dx_scale,
               pad_ratio=pad_ratio, 
               cnn_kernel_size=1,
               incremental=incremental
               )
    elif dim == 2:
        model = MNO2d(
                layers=[fc_dim]*(n_layer+1),
                fc_dim=fc_dim,
                in_dim=in_dim, out_dim=out_dim,
                act='gelu',
                modes1 = [k_max]*n_layer, modes2 = [k_max]*n_layer,
                grad_layer = grad_layer,
                geo_act='softsign', dx1=dxs[0]*dx_scale, dx2=dxs[1]*dx_scale,
                pad_ratio=pad_ratio, 
                cnn_kernel_size=1,
                incremental=incremental
                )
    else:
        raise ValueError(f"Only 1D and 2D are supported, got dim={dim}")
    
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    return model


def mno_floating_point_cost(
    dim, in_dim, out_dim, k_max, fc_dim, nlayer, ne,
    grad_layer=True, pad_ratio=0.0,
):
    """Estimate inference FLOPs on an isotropic structured grid.

    Lifting and projection use the original grid size ``ne``. When padding is
    enabled, the operator layers use the padded grid size, matching the
    one-sided integer padding applied in ``MNO1d`` and ``MNO2d``.
    """
    if pad_ratio < 0:
        raise ValueError(f"pad_ratio must be nonnegative, got {pad_ratio}")

    ne_layer = ne
    if pad_ratio > 0:
        n = round(ne ** (1.0 / dim))
        n_padded = n + math.floor(pad_ratio * n)
        ne_layer = n_padded**dim

    c_sigma = 1.0
    K = (2*k_max+1)**dim
    C_lift = 2*ne*in_dim*fc_dim
    C_proj = 2*ne*fc_dim*fc_dim + 2*ne*fc_dim*out_dim + c_sigma*ne*fc_dim

    # C_layer = 10*fc_dim*ne*np.log2(ne) + K*(8*fc_dim*fc_dim - 2*fc_dim) + ((2*dim+4)*fc_dim*fc_dim + (2*dim+4+c_sigma)*fc_dim)*ne 

    #             FFT                         mode mixing                     affine term    add global/local     activation    residual    
    C_layer = 10*fc_dim*ne_layer*np.log2(ne_layer) + K*(8*fc_dim*fc_dim - 2*fc_dim) + 2*fc_dim*fc_dim*ne_layer + fc_dim*ne_layer + c_sigma*fc_dim*ne_layer + fc_dim*ne_layer

    
    if grad_layer:
        C_layer += ((2*dim+2)*fc_dim*fc_dim + (2*dim+2)*fc_dim)*ne_layer

    return C_lift + C_proj + nlayer*C_layer
