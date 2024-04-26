"""Tests for Q Table model."""

import gymnasium as gym
import pytest
import torch
from pytest_check import check

from r2seedo.models.q_table import QTable, train_for_one_episode


@pytest.fixture
def agent() -> QTable:
    """Return a QTable instance."""
    return QTable(num_states=3, num_actions=2, discount_factor=0.9)


def test_q_table_init(agent: QTable) -> None:
    """Test QTable initialization."""
    check.equal(agent.q_table.shape, (3, 2))
    check.equal(agent.to_dataframe().shape, (3, 2))


def test_q_table_update(agent: QTable) -> None:
    """Test QTable update method."""
    # (s, a, r, s')
    state = 0
    action = 0
    reward = 1.0
    next_state = 1
    learning_rate = 0.5

    # Compute the maximum Q-value for next state
    max_q_value = agent.q_table[next_state].max(dim=-1).values

    # Compute the temporal difference target
    td_target = reward + agent.discount_factor * max_q_value

    # Compute update for state-action pair
    q_value = agent.q_table[state, action]
    updated_q_value = q_value * (1 - learning_rate) + learning_rate * td_target

    # Update Q-value
    agent.update(state, action, reward, next_state, learning_rate)

    # Check if Q-value was updated correctly
    assert torch.allclose(agent.q_table[state, action], updated_q_value)


@pytest.mark.parametrize("shape", [(1,), (2,), (3, 4), (5, 6, 7)])
def test_q_table_get_action_shape(agent: QTable, shape: tuple[int, ...]) -> None:
    """Test QTable get_action method returns the correct action shape."""
    state = torch.randint(0, 3, shape)
    epsilon = 0.1
    action = agent.get_action(state, epsilon)
    check.equal(action.shape, state.shape)


@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_q_table_get_action(agent: QTable, epsilon: float) -> None:
    """Test QTable get_action method random vs greedy."""
    state = torch.zeros(10_000).to(torch.int64)
    action = agent.get_action(state, epsilon)
    if epsilon == 0.0:
        # Assert greedy action
        assert torch.equal(action, agent.q_table[state].argmax(dim=-1))
    else:
        # Assert random actions were chosen
        check.equal(action.unique().tolist(), [0, 1])


def test_train_for_one_episode() -> None:
    """Test train_for_one_episode function."""
    # Define environment
    env = gym.make("FrozenLake-v1", is_slippery=False)

    # Initialize agent
    agent = QTable(
        num_states=env.observation_space.n,  # type: ignore
        num_actions=env.action_space.n,  # type: ignore
        discount_factor=0.99,
    )

    total_reward = train_for_one_episode(
        agent, env, max_steps=100, learning_rate=0.5, epsilon=1, seed=1234
    )
    assert 0.0 <= total_reward <= 1.0
