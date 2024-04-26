"""Core utilities for r2seedo library."""

import json
from pathlib import Path
from typing import Any

import msgspec
import torch


class Config(msgspec.Struct, frozen=True):
    """Base configuration class.

    Methods
    -------
    to_dict()
        Convert configuration to serializable Python dictionary.
    dump(path)
        Save configuration to file.

    Class Methods
    -------------
    load(path)
        Load configuration from file.
    """

    def __str__(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to serializable Python dictionary."""
        return msgspec.to_builtins(self)

    def dump(self, path: Path | str) -> None:
        """Save configuration to file."""
        with open(path, "wb") as f:
            f.write(msgspec.json.encode(self))

    @classmethod
    def load(cls, path: Path | str) -> "Config":
        """Load configuration from file."""
        with open(path, "rb") as f:
            return msgspec.json.decode(f.read(), type=cls)


def get_device(device_id: int = 0) -> torch.device:
    """Get compute device by id."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    elif device_id < torch.cuda.device_count():
        return torch.device(device_id)
    else:
        raise ValueError(f"{device_id} is not valid.")
