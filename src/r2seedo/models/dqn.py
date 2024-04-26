"""Deep Q-Learning Network."""

import numpy as np
import numpy.typing as npt
import torch
from stable_baselines3.common.buffers import ReplayBufferSamples
from torch import Tensor, nn

from r2seedo.models.base_layers import ConvBlock, ParallelMLP
from r2seedo.utils.core import Config


# Configurations
class DQNConfig(Config, frozen=True):
    """Configuration for a DQN.

    Attributes
    ----------
    observation_shape: tuple[int, int, int]
        Shape (num_frames, height, width) of an environment observation.
    num_actions: int
        Number of actions in the environment.
    hidden_dim: int
        Dimension of the hidden layers.
        (default = 256)
    num_value_sub_networks: int
        Number of parallel sub-networks for estimating action values.
        (default = 32)
    value_network_depth: int
        Depth of each sub-network for estimating action values.
        (default = 3)
    """

    observation_shape: tuple[int, int, int]
    num_actions: int
    hidden_dim: int = 256
    num_value_sub_networks: int = 32
    value_network_depth: int = 3


class DQNTrainingConfig(Config, frozen=True):
    """Configuration for training a DQN.

    Attributes
    ----------
    gamma: float
        Future reward discount factor.
        Usually between 0.9 and 0.99.
    double_q: bool
        If True, decouple the selection of the action and evaluation of
        the action's value in the temporal difference target.
    target_inertia: float
        Inertia for target network's current weights when updating from
        online network.
    """

    gamma: float
    double_q: bool
    target_inertia: float


# Model
class DQN(nn.Module):
    """Deep Q-Learning Network."""

    def __init__(self, dqn_config: DQNConfig) -> None:
        """Initialize a Deep Q-Learning Network from configuration."""
        super().__init__()

        # Store configuration
        self.config = dqn_config

        # Submodule for featurizing environment observations
        in_channels = self.config.observation_shape[0]
        out_channels = np.linspace(in_channels * 8, self.config.hidden_dim, 3).astype(
            int
        )
        self.env_featurizer = nn.Sequential(
            ConvBlock(in_channels, out_channels[0], kernel_size=8, stride=4),
            ConvBlock(out_channels[0], out_channels[1], kernel_size=4, stride=2),
            ConvBlock(out_channels[1], out_channels[2], kernel_size=4, stride=2),
            nn.AdaptiveAvgPool2d(output_size=(2, 2)),
            nn.Flatten(),
            # nn.Linear(hidden_dim * 4, num_actions),
        )

        # Submodule for parallel estimating action values across multiple sub-networks
        self.value_networks = ParallelMLP(
            in_features=self.config.hidden_dim * 4,  # 2x2 hidden_dim
            out_features=self.config.hidden_dim // 4,
            hidden_dim=self.config.hidden_dim,
            num_sub_networks=self.config.num_value_sub_networks,
            depth=self.config.value_network_depth,
        )

        # linear layer to map value latent space to # of actions
        self.to_action_space = nn.Linear(
            self.config.hidden_dim // 4, self.config.num_actions
        )

        # Value selector across sub-networks
        self.value_selector = nn.Sequential(
            nn.Linear(
                in_features=self.config.hidden_dim * 4,
                out_features=2 * self.config.num_value_sub_networks,
            ),
            nn.ELU(),
            nn.Linear(
                in_features=2 * self.config.num_value_sub_networks,
                out_features=self.config.num_value_sub_networks,
            ),
            nn.Softmax(dim=-1),
        )

    def forward(self, observation: Tensor) -> Tensor:
        """Estimate value of each action given the observation.

        Parameters
        ----------
        observation: Tensor[..., num_frames, H, W, dtype=uint8]
            Environment observation(s) to estimate value of each action for.

        Returns
        -------
        value_estimates: Tensor[..., num_actions, dtype=float32]
            Estimated value of each action for the observation(s).
        """
        # Featurize environment observation
        features: Tensor = self.env_featurizer.forward(observation.float() / 255.0)

        # Estimate value of each action across multiple sub-networks
        value_latent = self.value_networks.forward(features)

        # Select value estimate from each sub-network
        value_weights = self.value_selector.forward(features)

        # Weighted sum of value estimates from sub-networks
        value_estimates = torch.einsum("bni,bn->bi", value_latent, value_weights)

        # Map value latent space to action space
        value_estimates = self.to_action_space.forward(value_estimates)

        return value_estimates

    def get_action(self, observation: Tensor, epsilon: float) -> Tensor:
        """Select an action using an epsilon-greedy strategy.

        Parameters
        ----------
        observation: Tensor[..., num_frames, H, W, dtype=uint8]
            Environment observation(s) to choose an action for.
        epsilon: float
            Probability of choosing a random action (exploration).

        Returns
        -------
        action: Tensor[..., dtype=int64]
            Action(s) to take in the environment.
        """
        if observation.ndim == 3:  # noqa: PLR2004
            # Add batch dimension if missing
            observation = observation.unsqueeze(0)

        # For exploration, choose a random action
        exploration_action = torch.randint(
            low=0,
            high=self.config.num_actions,
            size=observation.shape[:-3],
            device=self.device,
        )
        if epsilon == 1:
            # 100% exploration
            return exploration_action

        # For exploitation, use network's estimate of best action given the observation
        exploitation_action = self.forward(observation).argmax(dim=-1)
        if epsilon == 0:
            # 100% exploitation
            return exploitation_action

        # Choose between exploration and exploitation
        z = torch.rand(exploration_action.shape, device=self.device)
        action = torch.where(z < epsilon, exploration_action, exploitation_action)

        return action

    def predict(
        self,
        observation: npt.NDArray[np.float32],
        state: None = None,
        episode_start: npt.NDArray[np.float32] | None = None,  # noqa: ARG002
        deterministic: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[npt.NDArray[np.float32], None]:
        """Predict action given an observation."""
        with torch.no_grad():
            action = self.get_action(
                torch.as_tensor(observation),
                epsilon=0.0 if deterministic else 0.01,
            )

        return action.cpu().numpy(), state

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device


# Functions
def calculate_loss(
    online_network: DQN,
    target_network: DQN,
    samples: ReplayBufferSamples,
    *,
    gamma: float = 0.99,
    double_q: bool = True,
) -> Tensor:
    """Calculate training loss for a DQN.

    Parameters
    ----------
    online_network: DQN
        Online network.
    target_network: DQN
        Target network.
    samples: ReplayBufferSamples
        Sample batch from replay buffer.
            samples.observations: Tensor[batch_size, num_frames, H, W, dtype=uint8]
            samples.next_observations: Tensor[batch_size, num_frames, H, W, dtype=uint8]
            samples.actions: Tensor[batch_size, 1, dtype=int64]
            samples.rewards: Tensor[batch_size, 1, dtype=float32]
    gamma: float
        Discount factor for future rewards.
        (default = 0.99)
    double_q: bool
        If True, decouple the selection of the action and evaluation of
        the action's value in the temporal difference target (double q-learning).
        (default = True)

    Returns
    -------
    Tensor
        Mean squared difference between online value estimate and TD target.
    """
    # Apply greedy policy to get value estimates of next observation
    with torch.no_grad():
        if double_q:
            # Online network's "optimal" action for next observation
            next_actions = online_network.get_action(
                samples.next_observations, epsilon=0.0
            )

            # Get value estimate of (next observation, action) from target network
            target_value_est = get_action_value_estimate(
                value_estimates=target_network.forward(samples.next_observations),
                actions=next_actions,
            )
        else:
            # Get value estimate of best action in next observation from target network
            target_value_est = (
                target_network.forward(samples.next_observations).max(dim=-1).values
            )

    # Calculate temporal difference target
    discounted_future_rewards = (
        gamma  # discount factor
        * target_value_est  # value estimate of next observation
        * (1 - samples.dones.squeeze(-1))  # zero out terminal states
    )
    td_target = samples.rewards.squeeze(-1) + discounted_future_rewards

    # Get online network's value estimate of actions taken during sampling
    online_value_est = get_action_value_estimate(
        value_estimates=online_network.forward(samples.observations),
        actions=samples.actions.squeeze(-1),
    )

    # loss := mean squared difference between online value estimate and TD target
    loss = nn.functional.mse_loss(td_target, online_value_est)

    return loss


def get_action_value_estimate(
    value_estimates: Tensor,
    actions: Tensor,
) -> Tensor:
    """Get value estimate of specified actions.

    Parameters
    ----------
    value_estimates: Tensor[..., num_actions]
        Value estimates of all actions.
    actions: Tensor[..., dtype=int64]
        Actions to get value estimates for.

    Returns
    -------
    Tensor[..., dtype=float32]
        Value estimates of specified actions.

    Examples
    --------
    >>> import torch

    >>> value_estimates = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    >>> actions = torch.tensor([0, 2])
    >>> get_action_value_estimate(value_estimates, actions)
    tensor([1., 6.])
    """
    return value_estimates.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def update_target_network(
    target_network: DQN,
    online_network: DQN,
    target_inertia: float,
) -> None:
    """Update target network weights from online network.

    `target = target * inertia + online * (1 - inertia)`

    Parameters
    ----------
    target_network: DQN
        The target network.
    online_network: DQN
        The online network.
    target_inertia: float
        Inertia for target network's current weights.

    Raises
    ------
    ValueError
        If `target_inertia` is not in [0, 1].
    """
    if not 0 <= target_inertia <= 1:
        raise ValueError("Target inertia must be in [0, 1]")

    if target_inertia == 0:
        # No inertia; copy online network's weights to target network
        target_network.load_state_dict(online_network.state_dict())
    else:
        for target_param, online_param in zip(
            target_network.parameters(), online_network.parameters(), strict=True
        ):
            target_param.data.mul_(target_inertia)
            target_param.data.add_(online_param.data * (1 - target_inertia))
