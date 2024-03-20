"""A simple policy network."""

import math

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from torch.distributions import Categorical


class ParallelSubPolicies(nn.Module):
    """A network of parallel MLP sub-policies."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        num_sub_policies: int,
        depth: int,
    ) -> None:
        """Initialize a network of parallel sub-policies.

        Parameters
        ----------
        obs_dim: int
            Dimension of the observation space.
        act_dim: int
            Dimension of the action space.
        hidden_dim: int
            Dimension of the hidden layers for each sub-policy.
        num_sub_policies: int
            Number of parallel sub-policies.
        depth: int, >= 1
            Depth of the network for each sub-policy.
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_sub_policies = num_sub_policies

        # Number of total channels based on number of sub-policies
        in_channels = num_sub_policies * obs_dim
        hidden_channels = num_sub_policies * hidden_dim
        out_channels = num_sub_policies * act_dim

        # Make sequential blocks based on network depth
        def make_block(
            in_ch: int, out_ch: int, *, activation: bool = True
        ) -> nn.Sequential:
            layers = [
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=1,
                    groups=num_sub_policies,
                ),
                nn.GroupNorm(num_groups=num_sub_policies, num_channels=out_ch),
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

    def forward(self, observation: Tensor) -> Tensor:
        """Get logits for action space for each sub-policy given an observation.

        Parameters
        ----------
        observation: Tensor[..., obs_dim, dtype=float32]
            Observation tensor.

        Returns
        -------
        action_logits: Tensor[..., num_sub_policies, act_dim, dtype=float32]
            Logits for action space for each sub-policy.
        """
        # Ensure observation is at least 2D
        observation = torch.atleast_2d(observation)

        # Reshape observation for input to network
        *other_dims, obs_dim = observation.shape
        if obs_dim != self.obs_dim:
            raise ValueError("Observation dimension does not match network input")
        batch_size = math.prod(other_dims)
        observation = (
            # Collapse other dimensions
            observation.view(-1, 1, self.obs_dim)
            # Duplicate observation for each sub-policy
            .expand(batch_size, self.num_sub_policies, self.obs_dim)
            # Reshape to input to 4D for network
            .reshape(batch_size, self.num_sub_policies * self.obs_dim, 1, 1)
        )

        # Get sub-policy action logits
        action_logits: Tensor = self.blocks.forward(observation)

        # Reshape to original dimensions
        action_logits = action_logits.reshape(
            *other_dims, self.num_sub_policies, self.act_dim
        )

        return action_logits


class SimplePolicyNet(nn.Module):
    """Simple policy network for learning a discrete action space."""

    def __init__(  # noqa: PLR0913
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        num_sub_policies: int,
        depth: int,
        obs_mean: Tensor | None = None,
        obs_std: Tensor | None = None,
    ) -> None:
        """Initialize a simple policy network.

        Parameters
        ----------
        obs_dim: int
            Dimension of the observation space.
        act_dim: int
            Dimension of the action space.
        hidden_dim: int
            Dimension of the hidden layers for each sub-policy.
        num_sub_policies: int
            Number of parallel sub-policies.
        depth: int, >= 1
            Depth of the network for each sub-policy.
        obs_mean: Tensor[obs_dim, dtype=float32] | None
            Observation variable means for normalization.
            If `None`, no normalization is applied.
            (default = None)
        obs_std: Tensor[obs_dim, dtype=float32] | None
            Observation variable stds for normalization.
            If `None`, no normalization is applied.
            (default = None)
        """
        super().__init__()

        # Observation normalization parameters
        self.obs_mean = nn.Parameter(obs_mean) if obs_mean is not None else None
        self.obs_std = nn.Parameter(obs_std) if obs_std is not None else None

        # Define a set of parallel sub-policies
        self.parallel_sub_policies = ParallelSubPolicies(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            num_sub_policies=num_sub_policies,
            depth=depth,
        )

        # Define a policy selector
        self.policy_selector = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=2 * num_sub_policies),
            nn.ELU(),
            nn.Linear(in_features=2 * num_sub_policies, out_features=num_sub_policies),
            nn.Softmax(dim=-1),
        )

    def forward(self, observation: Tensor) -> Tensor:
        """Get logits for action space given an observation.

        Parameters
        ----------
        observation: Tensor[..., obs_dim, dtype=float32]
            Observation tensor.

        Returns
        -------
        action_logits: Tensor[..., act_dim, dtype=float32]
            Logits for action space.
        """
        # Ensure observation is at least 2D
        observation = torch.atleast_2d(observation)

        # Normalize observation
        if self.obs_mean is not None and self.obs_std is not None:
            observation = (observation - self.obs_mean) / (self.obs_std + 1e-6)

        # Get sub-policy action logits
        sub_logits = self.parallel_sub_policies.forward(observation)

        # Get sub-policy selector weights
        sub_policy_weights: Tensor = self.policy_selector.forward(observation)

        # Get weighted action logits
        action_logits = (sub_logits * sub_policy_weights.unsqueeze(-1)).sum(dim=-2)

        return action_logits

    def get_policy(self, observation: Tensor) -> Categorical:
        """Compute probability distribution over actions given an observation.

        Parameters
        ----------
        observation: Tensor[..., obs_dim, dtype=float32]
            Observation tensor.

        Returns
        -------
        policy: Categorical
            Action distribution given the observation.
        """
        logits = self.forward(observation)
        return Categorical(logits=logits)

    def get_action(self, observation: Tensor) -> Tensor:
        """Sample an action from the policy given an observation.

        Parameters
        ----------
        observation : Tensor[..., obs_dim, dtype=float32]
            Observation tensor.

        Returns
        -------
        action: Tensor[..., dtype=int64]
            Sampled action.
        """
        return self.get_policy(observation).sample()

    def predict(
        self,
        observation: npt.NDArray[np.float32],
        state: None = None,
        episode_start: npt.NDArray[np.float32] | None = None,  # noqa: ARG002
        deterministic: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[npt.NDArray[np.float32], None]:
        """Predict action given an observation."""
        with torch.no_grad():
            policy = self.get_policy(
                observation=torch.as_tensor(observation, dtype=torch.float32),
            )

        action: Tensor = policy.mode if deterministic else policy.sample()

        return action.cpu().numpy(), state


def compute_loss(
    model: SimplePolicyNet, observations: Tensor, actions: Tensor, weights: Tensor
) -> Tensor:
    """Compute the policy gradient loss from a batch of `(obs, action, weight)` tuples.

    - the `(obs, action, weight)` tuples are collected while acting according
    to the current policy over the course of an episode (or several episodes).
    - the weight for a observation-action pair is the return from the episode
    to which it belongs.

    Parameters
    ----------
    model: SimplePolicyNet
        The policy network
    observations: Tensor[..., obs_dim, dtype=float32]
        A batch of observations
    actions: Tensor[..., dtype=int64]
        A batch of actions
    weights: Tensor[..., dtype=float32]
        Episode return for each state-action pair.

    Returns
    -------
    Tensor[1]
        The policy gradient loss
    """
    # Validate the shape of inputs
    if observations.shape[:-1] != actions.shape:
        raise ValueError("# Observations must match # actions")
    if observations.shape[:-1] != weights.shape:
        raise ValueError("# Observations must match # weights")

    # Get action probability distribution given observation
    action_dist = model.get_policy(observations)

    # Compute log probability of the action
    logp: Tensor = action_dist.log_prob(value=actions)

    # Compute the policy gradient loss
    return -(logp * weights).mean()


def get_reward_to_go_weights(rewards: Tensor) -> Tensor:
    """Compute the reward-to-go weights from a batch of rewards.

    Rewards obtained before taking an action have no bearing on how good
    that action was; only rewards that come after.

    Parameters
    ----------
    rewards: Tensor[n_steps, n_envs, dtype=float32]
        A sequence of rewards from an episode (per environment).

    Returns
    -------
    Tensor[..., dtype=float32]
        The reward-to-go weights for each step in the sequence.

    Example
    -------
    >>> import torch
    >>> rewards = torch.tensor(
    ...     [[ 2.4491,  1.2703], [-0.9930, -0.8285], [3.4893, -5.6373]]
    ... )
    >>> get_reward_to_go_weights(rewards)
    tensor([[ 4.9454, -5.1955],
            [ 2.4963, -6.4658],
            [ 3.4893, -5.6373]])
    """
    # Compute the reward-to-go weights
    return rewards.flip(dims=(0,)).cumsum(dim=0).flip(dims=(0,))
