"""Training utilities."""

from enum import StrEnum
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch
from torch import optim

from r2seedo.utils.core import Config


class ScheduleMode(StrEnum):
    """Ramp mode for schedule."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class ScheduleParams(NamedTuple):
    """Parameters for a specific timestep in a schedule."""

    timestep: int
    exploration_rate: float
    learning_rate: float


class LearningRateConfig(Config, frozen=True):
    """Configuration for learning rate evolution over training.

    Attributes
    ----------
    lr_start: float
        The initial learning rate (post-ramp).
        (default = 0.001)
    lr_end: float
        The final learning rate.
        (default = 1e-5)
    ramp_fraction: float
        The fraction of total timesteps to ramp the learning rate over.
        (default = 0.1)
    constant_fraction: float
        Fraction of total timesteps to keep the learning rate constant after the ramp.
        (default = 0.25)
    ramp_mode: ScheduleMode
        The ramp mode for the learning rate.
        (default = ScheduleMode.LINEAR)
    anneal_mode: ScheduleMode
        The anneal mode for the learning rate.
        (default = ScheduleMode.EXPONENTIAL)
    """

    lr_start: float = 0.001
    lr_end: float = 1e-5
    ramp_fraction: float = 0.1
    constant_fraction: float = 0.25
    ramp_mode: ScheduleMode = ScheduleMode.LINEAR
    anneal_mode: ScheduleMode = ScheduleMode.EXPONENTIAL

    def __post_init__(self) -> None:
        """Post initialization checks."""
        # Validate learning rate values
        if not (0 < self.lr_end < self.lr_start):
            raise ValueError(
                "Learning rate values must satisfy: `0 < lr_end < lr_start`"
            )

        # Validate the ramp and constant fractions
        if not 0 <= self.ramp_fraction <= 1:
            raise ValueError("Ramp fraction must be between 0 and 1.")
        if not 0 <= self.constant_fraction <= 1:
            raise ValueError("Constant fraction must be between 0 and 1.")
        if self.ramp_fraction + self.constant_fraction > 1:
            raise ValueError("Ramp and constant fractions must sum to less than 1.")

    def get_learning_rate_schedule(
        self, total_timesteps: int
    ) -> npt.NDArray[np.float_]:
        """Generate a learning rate schedule.

        Parameters
        ----------
        total_timesteps: int
            The total number of timesteps.

        Returns
        -------
        npt.NDArray[np.float_]
            The learning rate schedule.

        Examples
        --------
        >>> lr_config = LearningRateConfig(0.1, 0.01, 0.1, 0.4)
        >>> lr_config.get_learning_rate_schedule(10).round(2)
        array([0.01, 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.06, 0.03, 0.02, 0.01])
        """
        # Generate ramp schedule
        ramp_timesteps = int(total_timesteps * self.ramp_fraction)
        ramp_schedule = get_schedule(
            self.lr_end, self.lr_start, ramp_timesteps, self.ramp_mode
        )

        # Generate constant schedule post ramp
        constant_timesteps = int(total_timesteps * self.constant_fraction)
        constant_schedule = np.full(constant_timesteps, self.lr_start)

        # Generate annealing schedule
        annealing_timesteps = total_timesteps - ramp_timesteps - constant_timesteps
        anneal_schedule = get_schedule(
            self.lr_start, self.lr_end, annealing_timesteps, self.anneal_mode
        )

        # Concatenate ramp, constant, and annealing schedules
        schedule = np.concatenate((ramp_schedule, constant_schedule, anneal_schedule))

        return schedule


class ExplorationConfig(Config, frozen=True):
    """Configuration for exploration rate evolution over training.

    Attributes
    ----------
    epsilon_start: float
        The initial exploration rate.
        (default = 1.0)
    epsilon_end: float
        The final exploration rate.
        (default = 0.01)
    exploration_fraction: float
        The fraction of total timesteps to anneal epsilon over.
        (default = 0.2)
    anneal_mode: ScheduleMode
        The anneal mode for the exploration rate.
        (default = ScheduleMode.LINEAR)
    """

    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    exploration_fraction: float = 0.2
    anneal_mode: ScheduleMode = ScheduleMode.LINEAR

    def __post_init__(self) -> None:
        """Post initialization checks."""
        # Validate epsilon values
        if not (0 <= self.epsilon_end <= self.epsilon_start <= 1):
            raise ValueError(
                "Epsilon values must satisfy: `0 <= epsilon_end <= epsilon_start <= 1`"
            )

        # Validate the exploration fraction
        if not 0 <= self.exploration_fraction <= 1:
            raise ValueError("Exploration fraction must be between 0 and 1.")

        # Validate the anneal mode
        # object.__setattr__(self, "anneal_mode", ScheduleMode(self.anneal_mode))
        if self.anneal_mode is ScheduleMode.EXPONENTIAL and self.epsilon_end == 0:
            raise ValueError(
                "Epsilon values must be greater than 0 for exponential annealing."
            )

    def get_exploration_schedule(self, total_timesteps: int) -> npt.NDArray[np.float_]:
        """Generate an exploration schedule.

        Parameters
        ----------
        total_timesteps: int
            The total number of timesteps.

        Returns
        -------
        npt.NDArray[np.float_]
            The exploration schedule.

        Examples
        --------
        >>> exploration_config = ExplorationConfig(1.0, 0.1, 0.7)
        >>> exploration_config.get_exploration_schedule(10).round(2)
        array([1.  , 0.85, 0.7 , 0.55, 0.4 , 0.25, 0.1 , 0.1 , 0.1 , 0.1 ])

        >>> exploration_config = ExplorationConfig(
        ...     1.0, 0.1, 0.7, anneal_mode="exponential"
        ... )
        >>> exploration_config.get_exploration_schedule(10).round(2)
        array([1.  , 0.68, 0.46, 0.32, 0.22, 0.15, 0.1 , 0.1 , 0.1 , 0.1 ])
        """
        # Generate annealing schedule
        anneal_timesteps = int(total_timesteps * self.exploration_fraction)
        anneal_schedule = get_schedule(
            self.epsilon_start, self.epsilon_end, anneal_timesteps, self.anneal_mode
        )

        # Generate constant schedule post annealing
        constant_schedule = np.full(
            total_timesteps - anneal_timesteps, self.epsilon_end
        )

        # Concatenate the annealing and constant schedules
        schedule = np.concatenate((anneal_schedule, constant_schedule))

        return schedule


class ReplayBufferConfig(Config, frozen=True):
    """Configuration for replay buffer.

    Attributes
    ----------
    buffer_fraction: float
        Maximum size of buffer relative to total timesteps.
    batch_size: int
        Number of samples to return in a batch.
        (default = 32)
    """

    buffer_fraction: float
    batch_size: int = 32

    def get_buffer_size(self, total_timesteps: int) -> int:
        """Get the buffer size based on total timesteps."""
        return max(self.batch_size, int(total_timesteps * self.buffer_fraction))


class TrainingConfig(Config, frozen=True):
    """Configuration for training.

    Attributes
    ----------
    total_timesteps: int
        The total number of timesteps (per environment) to train for.
    num_envs: int
        The number of parallel environments to use for training.
        (default = 1)
    train_freq: int
        The frequency (# timesteps) at which to train the agent.
        (default = 1)
    evaluation_rate: float
        The rate (% of total timesteps) at which to evaluate the agent.
        (default = 0.1)
    optimizer_name: str
        The optimizer to use for training.
        (default = "Adam")
    learning_rate_config: LearningRateConfig
        The learning rate configuration.
        (default = LearningRateConfig())
    exploration_config: ExplorationConfig | None
        The exploration configuration.
        (default = ExplorationConfig())
    replay_buffer_config: ReplayBufferConfig | None
        The replay buffer configuration.
        (default = None)
    other: Config | None
        Any other configuration parameters.
        (default = None)
    """

    total_timesteps: int
    num_envs: int = 1
    train_freq: int = 1
    evaluation_rate: float = 0.1
    optimizer_name: str = "Adam"
    learning_rate_config: LearningRateConfig = LearningRateConfig()
    exploration_config: ExplorationConfig | None = ExplorationConfig()
    replay_buffer_config: ReplayBufferConfig | None = None
    other: Config | None = None

    @property
    def evaluation_freq(self) -> int:
        """Get the evaluation frequency (# timesteps)."""
        return int(self.total_timesteps * self.evaluation_rate)

    @property
    def optimizer_cls(self) -> type[optim.Optimizer]:
        """Get optimizer class based on name."""
        return getattr(torch.optim, self.optimizer_name)

    def get_parameter_schedule(self) -> list[ScheduleParams]:
        """Generate learning rate & exploration rate schedules given total timesteps."""
        learning_rate_schedule = self.learning_rate_config.get_learning_rate_schedule(
            self.total_timesteps
        )

        exploration_schedule = (
            self.exploration_config.get_exploration_schedule(self.total_timesteps)
            if self.exploration_config
            else np.full(self.total_timesteps, 0.0)
        )

        return [
            ScheduleParams(timestep, exploration_rate, learning_rate)
            for timestep, (exploration_rate, learning_rate) in enumerate(
                zip(exploration_schedule, learning_rate_schedule, strict=True)
            )
        ]


def get_schedule(
    start: float, stop: float, num: int, schedule_mode: ScheduleMode | str
) -> npt.NDArray[np.float_]:
    """Generate a schedule."""
    match schedule_mode:
        case ScheduleMode.LINEAR:
            schedule = np.linspace(start, stop, num)
        case ScheduleMode.EXPONENTIAL:
            schedule = np.exp(np.linspace(np.log(start), np.log(stop), num))
        case _:
            raise ValueError("Invalid schedule mode.")
    return schedule


def set_learning_rate(optimizer: optim.Optimizer, lr: float | list[float]) -> None:
    """Update learning rate of an optimizer (in place).

    Parameters
    ----------
    optimizer: Optimizer
        Optimizer instance to update learning rates for.
    lr: Union[float, List[float]]
        Learning rate to update to, OR learning rate for each parameter group.
    """
    if isinstance(lr, float):
        lr = [lr] * len(optimizer.param_groups)

    for param_group, grp_lr in zip(optimizer.param_groups, lr, strict=True):
        param_group["lr"] = grp_lr
