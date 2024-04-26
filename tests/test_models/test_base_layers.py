"""Tests for base layers."""

import pytest
import torch
from pytest_check import check

from r2seedo.models.base_layers import (
    ChannelNorm,
    ConvBlock,
    ParallelMLP,
    init_conv_weights,
)


def test_channel_norm() -> None:
    """Test that input is normalized along channel dimension."""
    torch.manual_seed(1234)

    num_channels = 64
    norm_layer = ChannelNorm(num_channels=num_channels)

    # Generate random input
    x = torch.randn(2, num_channels, 4, 4)
    x *= torch.arange(1, num_channels * 2 + 1).reshape(2, num_channels, 1, 1)

    # Apply channel normalization layer
    with torch.no_grad():
        y = norm_layer(x)

    # Normalize x along channel
    channel_mean = x.mean(dim=1, keepdim=True)
    channel_std = x.std(dim=1, keepdim=True)
    x_norm = (x - channel_mean) / (channel_std + 1e-6)

    # Check difference is small
    diffs = torch.abs(x_norm - y)
    assert diffs.mean() / torch.abs(y).mean() < 0.01


def test_conv_block() -> None:
    """Test conv block output shape."""
    # Define conv block
    conv_block = ConvBlock(
        in_channels=4,
        out_channels=16,
        kernel_size=2,
        stride=2,
    )

    # Generate random input
    x = torch.randn(10, 4, 84, 84)

    # Apply conv block
    with torch.no_grad():
        y = conv_block.forward(x)

    # Check output shape
    assert y.shape == (10, 16, 42, 42)


def test_pmlp_raises_on_invalid_depth() -> None:
    """Test that exception is raised when depth is invalid."""
    with pytest.raises(ValueError):
        ParallelMLP(
            in_features=8,
            out_features=4,
            hidden_dim=16,
            num_sub_networks=5,
            depth=0,
        )


@pytest.mark.parametrize("num_sub_networks", [1, 2, 3, 4])
@pytest.mark.parametrize("act_dim", [2, 3, 4])
@pytest.mark.parametrize("input_dim", [(8,), (3, 8), (4, 1), (2, 12, 8)])
def test_pmlp_output_shape(
    input_dim: tuple[int, ...], act_dim: int, num_sub_networks: int
) -> None:
    """Test output shape of ParallelMLP."""
    *other_dims, obs_dim = input_dim
    if not other_dims:
        other_dims = [1]

    psp = ParallelMLP(
        in_features=obs_dim,
        out_features=act_dim,
        hidden_dim=16,
        num_sub_networks=num_sub_networks,
        depth=2,
    ).eval()

    x = torch.randn(*input_dim)
    with torch.no_grad():
        y = psp.forward(x)

    assert y.shape == (*other_dims, num_sub_networks, act_dim)


def test_initialize_conv_weights() -> None:
    """Test initializing conv2d weights produces correct output on forward pass."""
    torch.manual_seed(1234)

    # Define a network and initialize weights ~N(0,1)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(5, 10, 1), torch.nn.ELU(), torch.nn.Conv2d(10, 1, 3, bias=False)
    )
    net[0].weight.data = torch.nn.init.normal_(net[0].weight.data)
    net[-1].weight.data = torch.nn.init.normal_(net[-1].weight.data)

    # Check size of output on forward pass
    x = torch.randn(1, 5, 3, 3)
    with torch.no_grad():
        y = net.forward(x).squeeze()
    with check:
        assert torch.allclose(y, torch.tensor(-29.4642), atol=1e-3)

    # Initialize weights
    net.apply(init_conv_weights)

    # Re-check size of output on forward pass
    with torch.no_grad():
        y = net.forward(x).squeeze()
    with check:
        assert torch.allclose(y, torch.tensor(-0.5675), atol=1e-3)
