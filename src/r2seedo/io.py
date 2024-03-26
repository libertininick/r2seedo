"""Utilities for loading and saving models."""

import io
from pathlib import Path
from typing import Any, NamedTuple

import joblib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


class LoadVerification(NamedTuple):
    """Verification key and signature for verifying model bytes before load."""

    key: ed25519.Ed25519PublicKey
    signature: bytes


class SaveReceipt(NamedTuple):
    """Path and signature for saving model bytes to disk."""

    path: Path
    signature: bytes | None


def load_model(source: Path | str, verification: LoadVerification | None) -> Any:
    """Load model from disk.

    This function reads the model bytes from disk, verifies the authenticity
    of the model using the provided verification key and signature, and loads
    the model from the bytes using `joblib.load`.

    Parameters
    ----------
    source: Path | str
        Model filepath.
    verification: Verification, optional
        Verification key and signature for model.
        If provided, the authenticity of model bytes is verified using
        the key and signature.

    Returns
    -------
    Any
        Model object.
    """
    # Read model into buffer
    with open(source, "rb") as f:
        buffer = io.BytesIO(f.read())

    # Verify model
    if verification:
        # Raises InvalidSignature if verification fails
        verification.key.verify(verification.signature, buffer.getvalue())

    # Load model from buffer
    return joblib.load(buffer)


def save_model(
    model: Any,
    destination: Path | str,
    private_key: ed25519.Ed25519PrivateKey | None,
) -> SaveReceipt:
    """Save model to disk.

    This function saves the model bytes to disk using `joblib.dump` and
    signs the model bytes using the provided private key.

    Parameters
    ----------
    model: Any
        Model object.
    destination: Path | str
        Destination filepath.
    private_key: Ed25519PrivateKey, optional
        Private key for signing model bytes.
        If provided, the model bytes are signed using the key.

    Returns
    -------
    SaveReceipt
        Path and signature for saving model bytes to disk.
    """
    # Dump model to buffer
    buffer = io.BytesIO()
    joblib.dump(model, buffer)

    # Sign model
    signature = None
    if private_key:
        signature = private_key.sign(buffer.getvalue())

    # Write model to disk
    with open(destination, "wb") as f:
        f.write(buffer.getvalue())

    return SaveReceipt(Path(destination), signature)
