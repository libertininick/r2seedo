"""Base layers for Deep RL Agents."""

import math
from typing import Any

import torch
from torch import Tensor, nn


class ChannelNorm(nn.Module):
    """Normalization along channel dimension."""

    def __init__(
        self,
        num_channels: int,
        *,
        eps: float = 0.00001,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> None:
        """Initialize layer.

        Parameters
        ----------
        num_channels: int
            Number of channels.
        eps: float, optional
            Small constant for safe divide.
            (default = 0.00001)
        elementwise_affine: bool, optional
            Include learnable elementwise affine parameters.
            (default = True)
        bias: bool, optional
            Include learnable bias parameter.
            (default = True)
        device: Any | None, optional
            Compute device.
            (default = None)
        dtype: Any | None, optional
            Parameter data type.
            (default = None)
        """
        super().__init__()
        self.normalizer = nn.LayerNorm(
            normalized_shape=num_channels,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Parameters
        ----------
        x: Tensor[B, C, H, W]

        Returns
        -------
        Tensor[B, C, H, W]
        """
        return self.normalizer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvBlock(nn.Module):
    """Two stage convolutional block.

    `features >| 1x1 bottleneck -> group-wise cov |> features`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] | int,
        *,
        stride: tuple[int, int] | int = 1,
        padding: tuple[int, int] | int = 0,
        dropout_p: float | None = None,
    ) -> None:
        """Initialize block.

        Parameters
        ----------
        in_channels: int
            Number of channels into block.
        out_channels: int
            Number of channels out of block.
        kernel_size: tuple[int, int] | int
            Group-wise convolution kernel size.
        stride: tuple[int, int] | int, optional
            Group-wise convolution stride.
            (default = 1)
        padding: tuple[int, int] | int, optional
            Group-wise convolution padding.
            (default = 0)
        dropout_p: float | None, optional
            Channel dropout probability.
            (default = None)
        """
        super().__init__()

        # Define block layers
        layers = [
            # Normalize along channel dimension
            ChannelNorm(num_channels=in_channels),
            # ELU activation
            nn.ELU(),
            # Reduce feature dimension
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            # Normalize along spatial dimension
            nn.GroupNorm(num_groups=out_channels, num_channels=out_channels),
            # ELU activation
            nn.ELU(),
            # Conv per channel
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=out_channels,
            ),
        ]

        # Add dropout layer
        if dropout_p:
            layers.append(nn.Dropout2d(p=dropout_p))

        # Wrap in sequential container
        self.block = nn.Sequential(*layers)

        # Initialize weights
        self.apply(init_conv_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Parameters
        ----------
        x: Tensor[B, C_in, H_in, W_in]

        Returns
        -------
        Tensor[B, C_out, H_out, W_out]
        """
        return self.block(x)


class ParallelMLP(nn.Module):
    """Parallel multi layer perceptron (MLP) network."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int,
        num_sub_networks: int,
        depth: int,
    ) -> None:
        """Initialize a network of parallel sub-policies.

        Parameters
        ----------
        in_features: int
            Number of input features.
        out_features: int
            Number of output features.
        hidden_dim: int
            Dimension of the hidden layers for each sub-network.
        num_sub_networks: int
            Number of parallel sub-networks.
        depth: int, >= 1
            Depth of each sub-network.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_sub_networks = num_sub_networks

        # Number of total channels based on number of sub-networks
        in_channels = num_sub_networks * in_features
        hidden_channels = num_sub_networks * hidden_dim
        out_channels = num_sub_networks * out_features

        # Make sequential blocks based on network depth
        def make_block(
            in_ch: int, out_ch: int, *, activation: bool = True
        ) -> nn.Sequential:
            layers = [
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=1,
                    groups=num_sub_networks,
                ),
                nn.GroupNorm(num_groups=num_sub_networks, num_channels=out_ch),
            ]
            if activation:
                layers.append(nn.ELU())

            return nn.Sequential(*layers)

        blocks: list[nn.Sequential] = []
        if depth == 1:
            blocks.append(make_block(in_channels, out_channels))
        elif depth > 1:
            blocks.append(make_block(in_channels, hidden_channels))
            for _ in range(depth - 2):
                blocks.append(make_block(hidden_channels, hidden_channels))
            blocks.append(make_block(hidden_channels, out_channels, activation=False))
        else:
            raise ValueError("Depth must be a positive integer")

        # Combine blocks into a sequential module
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        """Get logits for action space for each sub-policy given an observation.

        Parameters
        ----------
        x: Tensor[..., in_features]
            Input features.

        Returns
        -------
        Tensor[..., num_sub_networks, out_features]
            Outputs for each sub-policy.
        """
        # Ensure input is at least 2D
        x = torch.atleast_2d(x)

        # Reshape inputs for parallel sub-networks
        *other_dims, feat_dim = x.shape
        if feat_dim != self.in_features:
            raise ValueError("Features dimension does not match network input")
        batch_size = math.prod(other_dims)
        x = (
            # Collapse other dimensions
            x.view(-1, 1, self.in_features)
            # Duplicate observation for each sub-network
            .expand(batch_size, self.num_sub_networks, self.in_features)
            # Reshape to input to 4D for network
            .reshape(batch_size, self.num_sub_networks * self.in_features, 1, 1)
        )

        # Get sub-network outputs
        x = self.blocks.forward(x)

        # Reshape to original dimensions
        x = x.reshape(*other_dims, self.num_sub_networks, self.out_features)

        return x


def init_conv_weights(module: nn.Module) -> None:
    """Initialize the weights of convolutional layers.

    Examples
    --------
    >>> _ = torch.manual_seed(1234)
    >>> net = torch.nn.Conv2d(5,1,3)
    >>> net.weight.data = torch.nn.init.normal_(net.weight.data)

    >>> x = torch.randn(1, 5, 3, 3)
    >>> with torch.no_grad():
    ...     y = net(x).squeeze()
    >>> y
    tensor(5.0570)

    >>> net = net.apply(init_conv_weights)
    >>> with torch.no_grad():
    ...     y = net(x).squeeze()
    >>> y
    tensor(0.2762)
    """
    if isinstance(module, nn.Conv2d):
        module.weight.data = nn.init.kaiming_normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data = nn.init.constant_(module.bias.data, 0)
