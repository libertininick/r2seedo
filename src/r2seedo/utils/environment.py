"""Environment utilities."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.env_util import VecEnv, make_vec_env
from torch import Tensor, device

from r2seedo.utils.core import Config


class AtariEnvConfig(Config, frozen=True):
    """Configuration for Atari Environment.

    Attributes
    ----------
    env_id: str
        The environment id of the Atari game.
    clip_reward: bool
        If True, clip reward to {-1, 0, 1} depending on its sign.
        (default = True)
    frame_skip: int
        Frequency at which the agent experiences the game.
        (default = 4)
    frame_stack: int
        Number of frames to stack for each observation.
        (default = 4)
    grayscale: bool
        If True, convert RGB image to grayscale.
        (default = True)
    noop_max: int
        Maximum number of no-op actions at the beginning of an episode
        to obtain initial state.
        (default = 30)
    screen_size: tuple[int, int] | int
        Size (height, width) to resize screen to.
        (default = 84)
    terminal_on_life_loss: bool
        If True, the episode ends when a life is lost.
        (default = True)
    """

    env_id: str
    clip_reward: bool = True
    frame_skip: int = 4
    frame_stack: int = 4
    grayscale: bool = True
    noop_max: int = 30
    screen_size: tuple[int, int] | int = (84, 84)
    terminal_on_life_loss: bool = True

    def make_env(
        self,
        *,
        num_envs: int = 1,
        video_folder: Path | str | None = None,
        seed: int | None = None,
    ) -> VecEnv:
        """Create a vectorized environment from configuration.

        Parameters
        ----------
        num_envs: int
            Number of environments to create.
            (default = 1)
        video_folder: Path | str | None
            Folder to store recorded videos.
            (default = None)
        seed: int | None
            Seed for environment.
            (default = None)

        Returns
        -------
        VecEnv
            Vectorized environment.
        """
        return make_vec_env(
            env_id=self.env_id,
            n_envs=num_envs,
            seed=seed,
            wrapper_class=partial(
                wrap_atari_env,
                clip_reward=self.clip_reward,
                frame_skip=self.frame_skip,
                frame_stack=self.frame_stack,
                grayscale=self.grayscale,
                noop_max=self.noop_max,
                screen_size=self.screen_size,
                terminal_on_life_loss=self.terminal_on_life_loss,
                video_folder=video_folder,
            ),
        )


def capture_replay(
    env: gym.Env,
    action_func: Callable[[Tensor], Tensor],
    video_folder: str,
    max_steps: int | None = None,
) -> list[float]:
    """Capture a replay of the model's interaction with the environment.

    Parameters
    ----------
    env: gym.Env
        An RL environment instance.
    action_func: Callable[[Tensor], Tensor]
        A function that takes observation tensor and returns action tensor.
    video_folder: str
        Folder to store the recorded video.
    max_steps: int | None, optional
        Maximum number of steps to record.
        If None, record until the end of the episode.
        (default = None)

    Returns
    -------
    list[float]
        List of rewards obtained during the replay.
    """
    # Add video recording wrapper
    env = gym.wrappers.RecordVideo(env, video_folder)

    # Reset the environment
    obs, *_ = env.reset()

    # Capture the replay
    rewards = []
    step = 0
    while True:
        # convert observation to tensor
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        else:
            obs = torch.tensor(obs)

        # get action | observation
        action = action_func(obs).item()

        # Perform the actions
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)

        # Increment the step count
        step += 1

        # Check if the environment has reached end of episode or max steps exceeded
        if terminated or truncated or (max_steps is not None and step >= max_steps):
            break

    # Close the environment
    env.close()

    return rewards


def copy_n_adj_next_observation(
    next_obs: np.ndarray,
    termination: np.ndarray,
    infos: list[dict],
) -> np.ndarray:
    """Copy the next observation and adjust for termination (if any).

    Parameters
    ----------
    next_obs : np.ndarray[num_envs, ...]
        The next observations.
    termination : np.ndarray[num_envs, dtype=bool]
        Termination flag for each environment.
    infos: list[dict]
        Information for each environment.

    Returns
    -------
    np.ndarray[num_envs, ...]
        A copy of next observation adjusted for termination for each env.
    """
    # Copy the next observations
    next_obs_copy = next_obs.copy()

    # Check if any of the episodes terminated
    if termination.any():
        for i, info in enumerate(infos):
            if termination[i]:
                # If the episode terminated, use the terminal observation
                next_obs_copy[i] = info["terminal_observation"]

    return next_obs_copy


def get_replay_buffer(
    env: VecEnv,
    buffer_size: int,
    *,
    device: device | str = "auto",
    optimize_memory_usage: bool = True,
    handle_timeout_termination: bool = False,
) -> ReplayBuffer:
    """Initialize a replay buffer for a specific environment."""
    buffer = ReplayBuffer(
        buffer_size=buffer_size,
        observation_space=env.observation_space,
        action_space=env.action_space,
        n_envs=env.num_envs,
        device=device,
        optimize_memory_usage=optimize_memory_usage,
        handle_timeout_termination=handle_timeout_termination,
    )

    return buffer


def wrap_atari_env(  # noqa: C901, PLR0913
    env: gym.Env,
    *,
    clip_reward: bool,
    frame_skip: int,
    frame_stack: int,
    grayscale: bool,
    noop_max: int,
    screen_size: tuple[int, int] | int,
    terminal_on_life_loss: bool,
    video_folder: Path | str | None,
) -> gym.Env:
    """Wrap Atari environment with specified modifications.

    Parameters
    ----------
    clip_reward: bool
        If True, clip reward to `{-1, 0, 1}` depending on its sign.
    frame_skip: int
        Frequency at which the agent experiences the game.
    frame_stack: int
        Number of frames to stack for each observation.
    grayscale: bool
        If True, convert RGB image to grayscale.
    noop_max: int
        Maximum number of no-op actions at the beginning of an episode
        to obtain initial state.
    screen_size: tuple[int, int] | int
        Size (height, width) to resize screen to.
    terminal_on_life_loss: bool
        If True, the episode ends when a life is lost.
    video_folder: Path | str | None
        Folder to store recorded videos. If None, no recording is done.

    Returns
    -------
    gym.Env
        Wrapped environment.
    """
    # Input wrappers
    env = gym.wrappers.ResizeObservation(env, shape=screen_size)
    if grayscale:
        env = gym.wrappers.GrayScaleObservation(env)
    if frame_skip > 1:
        env = MaxAndSkipEnv(env, skip=frame_skip)
    if frame_stack > 1:
        env = gym.wrappers.FrameStack(env, num_stack=frame_stack)

    # Reward wrappers
    if clip_reward:
        env = ClipRewardEnv(env)

    # Reset wrappers
    if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
        env = FireResetEnv(env)
    if noop_max > 0:
        env = NoopResetEnv(env, noop_max=noop_max)
    if terminal_on_life_loss:
        env = EpisodicLifeEnv(env)

    # Output wrappers
    if video_folder:
        env = gym.wrappers.RecordVideo(env, str(video_folder))

    # Return wrapped environment
    return env
