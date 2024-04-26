"""Test for model I/O functions."""

from tempfile import TemporaryDirectory

import pytest
import torch
from pytest_check import check

from r2seedo.io import (
    AssetFileNames,
    generate_keypair,
    load_keypair,
    load_n_verify_model,
    save_private_key,
    sign_n_save_model,
)


@pytest.mark.parametrize("password", [None, b"password-bytes", "password-str"])
def test_generate_n_save_keypair(password: bytes | str | None) -> None:
    """Test generating and saving a keypair."""
    # Test message to sign
    test_message = b"test message"

    # Generate keypair
    keypair = generate_keypair()

    # Sign test message with private key
    signature = keypair.private.sign(test_message)

    # Save and reload keypair
    with TemporaryDirectory() as temp_dir:

        # Save private key
        private_key_path = f"{temp_dir}/{AssetFileNames.PRIVATE_KEY}"
        save_private_key(keypair.private, private_key_path, password)

        # Load keypair from private key
        loaded_keypair = load_keypair(private_key_path, password)

    # Sign test message with loaded private key
    loaded_signature = loaded_keypair.private.sign(test_message)

    # Check that bytes are equal, and public key can verify message with signature
    check.equal(signature, loaded_signature)
    check.is_none(loaded_keypair.public.verify(signature, test_message))
    check.is_none(keypair.public.verify(loaded_signature, test_message))


def test_save_load_model() -> None:
    """Test saving and reloading a simple model."""
    # Define a simple model
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
    )

    # Generate keypair for signing
    keypair = generate_keypair()

    with TemporaryDirectory() as temp_dir:
        # Save model
        sign_n_save_model(model, temp_dir, keypair)

        # Load model
        loaded_model = load_n_verify_model(source=temp_dir)

    check.equal(len(model), len(loaded_model))
    with check:
        assert torch.equal(model[0].weight, loaded_model[0].weight)
        assert torch.equal(model[2].weight, loaded_model[2].weight)


def test_save_load_model_separate_public_key() -> None:
    """Test saving and reloading a simple model keeping the public key separate."""
    # Define a simple model
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
    )

    # Generate keypair for signing
    keypair = generate_keypair()

    with TemporaryDirectory() as temp_dir:
        # Save model
        sign_n_save_model(model, temp_dir, keypair, save_public_key=False)

        # Load model
        loaded_model = load_n_verify_model(source=temp_dir, public_key=keypair.public)

    check.equal(len(model), len(loaded_model))
    with check:
        assert torch.equal(model[0].weight, loaded_model[0].weight)
        assert torch.equal(model[2].weight, loaded_model[2].weight)


def test_overwrite_raises() -> None:
    """Test that trying overwriting a model dir raises an error."""
    # Define a simple model
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
    )

    # Generate keypair for signing
    keypair = generate_keypair()

    with TemporaryDirectory() as temp_dir:
        # Save model
        sign_n_save_model(model, temp_dir, keypair)

        # Try to save model again
        with pytest.raises(FileExistsError):
            sign_n_save_model(model, temp_dir, keypair)


def test_overwrite() -> None:
    """Test that overwriting model dir works."""
    # Define two simple model
    torch.manual_seed(0)
    model0 = torch.nn.Sequential(
        torch.nn.Linear(1, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
    )
    model1 = torch.nn.Sequential(
        torch.nn.Linear(1, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )

    # Generate keypair for signing
    keypair = generate_keypair()

    with TemporaryDirectory() as temp_dir:
        # Save model 0
        sign_n_save_model(model0, temp_dir, keypair)

        # Overwrite model 0 with model 1
        sign_n_save_model(model1, temp_dir, keypair, overwrite=True)

        # Load model
        loaded_model1 = load_n_verify_model(source=temp_dir, public_key=keypair.public)

    check.equal(len(model1), len(loaded_model1))
    with check:
        assert torch.equal(model1[0].weight, loaded_model1[0].weight)
        assert torch.equal(model1[2].weight, loaded_model1[2].weight)
        assert torch.equal(model1[4].weight, loaded_model1[4].weight)
