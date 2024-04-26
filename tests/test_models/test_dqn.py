"""Tests for DQN model."""

import pytest
import torch
from pytest_check import check
from stable_baselines3.common.buffers import ReplayBufferSamples

from r2seedo.models.dqn import DQN, DQNConfig, calculate_loss, get_action_value_estimate

# Constants
BATCH_SIZE = 16
OBS_SHAPE = (1, 64, 64)
NUM_ACTIONS = 2
HIDDEN_DIM = 8
NUM_VALUE_SUB_NETWORKS = 4
VALUE_NETWORK_DEPTH = 1


@pytest.fixture(scope="module")
def dqn_config() -> DQNConfig:
    """Return a DQNConfig test fixture."""
    return DQNConfig(
        observation_shape=OBS_SHAPE,
        num_actions=NUM_ACTIONS,
        hidden_dim=HIDDEN_DIM,
        num_value_sub_networks=NUM_VALUE_SUB_NETWORKS,
        value_network_depth=VALUE_NETWORK_DEPTH,
    )


@pytest.fixture(scope="module")
def online_net(dqn_config: DQNConfig) -> DQN:
    """Return a DQN test fixture."""
    torch.manual_seed(1234)
    return DQN(dqn_config)


@pytest.fixture(scope="module")
def target_net(dqn_config: DQNConfig) -> DQN:
    """Return a DQN test fixture."""
    torch.manual_seed(4567)
    return DQN(dqn_config)


@pytest.fixture(scope="module")
def samples() -> ReplayBufferSamples:
    """Return a ReplayBufferSamples test fixture."""
    torch.manual_seed(8910)
    return ReplayBufferSamples(
        observations=torch.randint(0, 256, (BATCH_SIZE, *OBS_SHAPE)),
        actions=torch.randint(0, NUM_ACTIONS, (BATCH_SIZE, 1)),
        rewards=torch.rand((BATCH_SIZE, 1)).round(),  # rewards are {0., 1.}
        next_observations=torch.randint(0, 256, (BATCH_SIZE, *OBS_SHAPE)),
        dones=torch.randint(0, 2, (BATCH_SIZE, 1)),  # dones are {0, 1}
    )


def test_forward_pass(online_net: DQN, samples: ReplayBufferSamples) -> None:
    """Test forward pass of DQN model."""
    value_est = online_net.forward(samples.observations)
    assert value_est.shape == (BATCH_SIZE, NUM_ACTIONS)


def test_get_action(online_net: DQN, samples: ReplayBufferSamples) -> None:
    """Test DQN get_action method."""
    action = online_net.get_action(samples.observations, epsilon=0.5)
    assert action.shape == (BATCH_SIZE,)
    assert torch.all(action >= 0) and torch.all(action < NUM_ACTIONS)


@pytest.mark.parametrize("num_actions", [1, 2, 3, 4])
@pytest.mark.parametrize("observation_shape", [(1,), (2,), (3, 4), (5, 6, 7)])
def test_get_action_value_estimate(
    observation_shape: tuple[int, ...], num_actions: int
) -> None:
    """Test get_action_value_estimate function."""
    # Random value estimates and actions
    value_estimates = torch.rand(*observation_shape, num_actions)
    actions = torch.randint(0, num_actions, observation_shape)

    # Get action value estimates
    action_value_estimates = get_action_value_estimate(value_estimates, actions)

    # Check if output shape is correct
    with check:
        assert torch.atleast_1d(action_value_estimates).shape == observation_shape

    # Check if output values are correct
    value_estimates = value_estimates.reshape(-1, num_actions)
    expected = torch.stack(
        [value_estimates[i, a] for i, a in enumerate(actions.reshape(-1, 1))],
        dim=0,
    )
    assert torch.allclose(action_value_estimates, expected.reshape(observation_shape))


def test_calculate_loss_double_q(
    online_net: DQN, target_net: DQN, samples: ReplayBufferSamples
) -> None:
    """Test calculate_loss function with `double_q=True`."""
    gamma = 0.5

    # Calculate loss
    loss = calculate_loss(
        online_network=online_net,
        target_network=target_net,
        samples=samples,
        gamma=gamma,
        double_q=True,
    )

    # Calculate expected loss
    with torch.no_grad():
        # Get optimal actions from online network
        next_actions = online_net.get_action(samples.next_observations, epsilon=0.0)

        # Get value estimate of actions from target network
        value_est = target_net.forward(samples.next_observations)
        target_value_est = value_est.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)

    # Calculate temporal difference target
    discounted_future_rewards = (
        gamma  # discount factor
        * target_value_est  # value estimate of next observation
        * (1 - samples.dones.squeeze(-1))  # zero out terminal states
    )
    td_target = samples.rewards.squeeze(-1) + discounted_future_rewards

    # Get online network's value estimate of actions taken during sampling
    online_value_est = online_net.forward(samples.observations)
    online_value_est = online_value_est.gather(-1, samples.actions).squeeze(-1)

    # Mean squared error loss
    expected_loss = (online_value_est - td_target).pow(2).mean()

    assert torch.allclose(loss, expected_loss)
