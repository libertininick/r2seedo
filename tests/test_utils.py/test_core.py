"""Tests for core utilities."""

from enum import StrEnum
from tempfile import TemporaryDirectory
from typing import NamedTuple

import pytest

from r2seedo.utils.core import Config


# Mock configuration classes
class MockEnum(StrEnum):
    """Mock enumeration."""

    A = "a"
    B = "b"


class MockTuple(NamedTuple):
    """Mock named tuple."""

    v0: str
    v1: float
    v3: list[int]


class MockChildConfig(Config, frozen=True):
    """Mock child configuration class."""

    param0: MockTuple


class MockConfig(Config, frozen=True):
    """Mock configuration class."""

    param0: int
    param1: MockEnum
    param2: MockChildConfig


@pytest.fixture
def config() -> MockConfig:
    """Create a configuration object."""
    return MockConfig(
        param0=0,
        param1=MockEnum.A,
        param2=MockChildConfig(
            param0=MockTuple("v-0", 0.0, [0, 1, 2]),
        ),
    )


def test_config_to_dict(config: MockConfig) -> None:
    """Test configuration to dictionary conversion."""
    assert config.to_dict() == {
        "param0": 0,
        "param1": "a",
        "param2": {"param0": ("v-0", 0.0, [0, 1, 2])},
    }


def test_config_dump_load(config: Config) -> None:
    """Test configuration dump and load."""
    with TemporaryDirectory() as tempdir:
        path = f"{tempdir}/config.json"
        config.dump(path)
        loaded = MockConfig.load(path)
        assert loaded == config
