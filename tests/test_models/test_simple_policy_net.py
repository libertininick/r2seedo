"""Tests for the SimplePolicyNet class."""

import torch
from pytest_check import check

from r2seedo.models.simple_policy_net import (
    SimplePolicyNet,
    compute_loss,
    get_reward_to_go_weights,
)


def test_simple_policy_net() -> None:
    """Test SimplePolicyNet."""
    policy = SimplePolicyNet(
        obs_dim=8,
        act_dim=4,
        hidden_dim=16,
        num_sub_policies=3,
        depth=2,
    )

    # 15 time steps, 5 environments:
    obs = torch.randn(15, 5, 8)
    with torch.no_grad():
        # Sample actions
        actions = policy.get_action(obs)
        check.equal(actions.shape, (15, 5))

        # Get the action distribution
        action_dist = policy.get_policy(obs)

        # Get the log probability of the actions taken
        logp: torch.Tensor = action_dist.log_prob(value=actions)
        check.equal(logp.shape, (15, 5))


def test_get_reward_to_go_weights() -> None:
    """Test get_reward_to_go_weights."""
    rewards = torch.tensor([[2.4491, 1.2703], [-0.9930, -0.8285], [3.4893, -5.6373]])
    weights = get_reward_to_go_weights(rewards)
    expected = torch.tensor([[4.9454, -5.1955], [2.4963, -6.4658], [3.4893, -5.6373]])
    assert torch.allclose(weights, expected)


def test_compute_loss() -> None:
    """Test compute_loss."""
    policy = SimplePolicyNet(
        obs_dim=8,
        act_dim=4,
        hidden_dim=16,
        num_sub_policies=3,
        depth=2,
    )

    # 15 time steps, 5 environments:
    obs = torch.randn(15, 5, 8)
    actions = torch.randint(0, 4, size=(15, 5))
    rewards = torch.randn(15, 5)

    # Compute the loss
    loss = compute_loss(policy, obs, actions, rewards)
    check.equal(loss.shape, ())
    check.is_instance(loss, torch.Tensor)
    assert torch.isfinite(loss)
