"""A Q-table model and utilities for the Q-learning algorithm."""

import gymnasium as gym
import pandas as pd
import torch
from torch import Tensor


class QTable:
    """Quality-value table for Q-learning algorithm."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        discount_factor: float,
        state_names: list[str] | None = None,
        action_names: list[str] | None = None,
    ) -> None:
        """Initialize the Q-table with zeros for all (state, action) pairs.

        Parameters
        ----------
        num_states: int
            Number of states in the environment.
        num_actions: int
            Number of actions in the environment.
        discount_factor: float
            Discount factor (gamma) for future rewards.
        state_names: list[str], optional
            Names of the states in the environment.
        action_names: list[str], optional
            Names of the actions in the environment.

        Attributes
        ----------
        q_table: Tensor[num_states, num_actions, dtype=float32]
            Learned table of Q-values for each (state, action) pair.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.state_names = state_names
        self.action_names = action_names

        # Initialize Q-table with zeros
        self.q_table = torch.zeros(num_states, num_actions)

    def __repr__(self) -> str:
        """Return a string representation of the Q-table."""
        return (
            f"{self.__class__.__name__}("
            f"num_states={self.num_states}, "
            f"num_actions={self.num_actions}, "
            f"discount_factor={self.discount_factor:.2f})"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert Q-table to a pandas DataFrame.

        Returns
        -------
        df: pd.DataFrame
            Q-table as a DataFrame with state and action names.
        """
        # Convert Q-table to a DataFrame
        df = pd.DataFrame(self.q_table.numpy())

        # Add state and action names to the DataFrame
        if self.state_names is not None:
            df.index = self.state_names
        if self.action_names is not None:
            df.columns = self.action_names

        return df

    def get_action(self, state: Tensor, epsilon: float) -> Tensor:
        """Select an action using an epsilon-greedy strategy.

        Parameters
        ----------
        state: Tensor[..., dtype=int64]
            Environment state(s) to choose an action for.
        epsilon: float
            Probability of choosing a random action (exploration).

        Returns
        -------
        action: Tensor[..., dtype=int64]
            Action(s) to take in the environment.
        """
        # For exploration, choose a random action
        exploration_action = torch.randint(
            low=0, high=self.num_actions, size=state.shape
        )

        # For exploitation, choose the best action in the Q-table given the state
        exploitation_action = self.q_table[state].argmax(dim=-1)

        action = torch.where(
            torch.rand(state.shape) < epsilon,
            exploration_action,
            exploitation_action,
        )

        return action

    def update(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        learning_rate: float,
    ) -> None:
        """Update Q-table (state, action) pair.

        Parameters
        ----------
        state: Tensor[..., dtype=int64]
            Current state of agent.
        action: Tensor[..., dtype=int64]
            Action taken by agent.
        reward: Tensor[..., dtype=float32]
            Reward received by agent for taking `action` at `state`.
        next_state: Tensor[..., dtype=int64]
            The next state given current state and action taken.
        learning_rate : float
            Learning rate.
        """
        # Compute the maximum Q-value for next state
        max_q_value = self.q_table[next_state].max(dim=-1).values

        # Compute the temporal difference target
        td_target = reward + self.discount_factor * max_q_value

        # Compute update for state-action pair
        self.q_table[state, action] = (
            self.q_table[state, action] * (1 - learning_rate)
            + learning_rate * td_target
        )


def train_for_one_episode(
    agent: QTable,
    env: gym.Env,
    max_steps: int,
    learning_rate: float,
    epsilon: float,
    *,
    seed: int | None = None,
    verbose: bool = False,
) -> float:
    """Train an agent for one episode in the environment.

    Parameters
    ----------
    agent: QTable
        Q-table agent to train in the environment.
    env: gym.Env
        Gym environment to train the agent in.
    max_steps: int
        Maximum number of steps per episode.
    learning_rate: float
        Learning rate for updating Q-values.
    epsilon: float
        Probability of choosing a random action (exploration).
    seed: int, optional
        Random environment seed for reproducibility.
    verbose: bool, optional
        Whether to print sequence of steps over episode.
        (default = False)

    Returns
    -------
    total_reward: float
        Total reward accumulated during the episode.
    """
    # Reset the environment and get the initial state
    state, _ = env.reset(seed=seed)
    state = torch.tensor(state)
    total_reward: float = 0

    # Iterate through the environment for a maximum number of steps
    for step_i in range(max_steps):
        # Get the action from the agent
        action = agent.get_action(state, epsilon)

        # Take a step in the environment given the action
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        next_state = torch.tensor(next_state)

        # Update the agent from the reward given by the action
        agent.update(state, action, torch.tensor(reward), next_state, learning_rate)
        state = next_state
        total_reward += float(reward)

        if verbose:
            print(
                f"{step_i}: "
                f"state={state.item():>2}, "
                f"action={action.item():>2}, "
                f"reward={reward:>4.0f}, "
                f"total_reward={total_reward:>5.0f}"
            )

        if terminated or truncated:
            break

    return total_reward
