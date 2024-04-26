"""Utilities for loading and saving models.

This module utilizes the `joblib` library for serializing and deserializing
model objects.
https://joblib.readthedocs.io/en/latest/persistence.html


WARNING: `joblib.load` relies on the pickle module and can therefore execute
arbitrary Python code. It should never be used to load files from untrusted sources.

To mitigate **some** of the risks associated with deserializing binary files,
this module provides functionality for verifying the authenticity of model
bytes before loading them. This is achieved by signing the model bytes
using an Ed25519 private key before saving them to disk, and verifying the
signature using the corresponding public key before deserializing the model.

This functionality is achieved using the `cryptography` library.
https://cryptography.io/en/latest/

WARNING: This implementation is not foolproof; to quote the `cryptography`
documentation: "You should ONLY use it if you're 100% absolutely sure that
you know what you're doing because this module is full of land mines,
dragons, and dinosaurs with laser guns."
"""

import io
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple

import joblib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

__all__ = [
    "KeyPair",
    "generate_keypair",
    "load_keypair",
    "load_n_verify_model",
    "save_private_key",
    "sign_n_save_model",
]


class AssetFileNames(StrEnum):
    """Asset file names."""

    MODEL = "model.joblib"
    SIGNATURE = "signature.txt"
    PRIVATE_KEY = "private.pem"
    PUBLIC_KEY = "public.pem"


class KeyPair(NamedTuple):
    """Ed25519 keypair.

    Attributes
    ----------
    private: Ed25519PrivateKey
        Private key.
    public: Ed25519PublicKey
        Public key.
    """

    private: ed25519.Ed25519PrivateKey
    public: ed25519.Ed25519PublicKey


class Verification(NamedTuple):
    """Verification key and signature for verifying model bytes before load."""

    key: ed25519.Ed25519PublicKey
    signature: bytes


class SaveReceipt(NamedTuple):
    """Path and signature for saving model bytes to disk."""

    path: Path
    signature: bytes | None


def generate_keypair() -> KeyPair:
    """Generate Ed25519 keypair.

    Returns
    -------
    KeyPair
        Private and public keypair.
    """
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    return KeyPair(private_key, public_key)


def load_n_verify_model(
    source: Path | str,
    public_key: ed25519.Ed25519PublicKey | Path | str | None = None,
) -> Any:
    """Load model from directory and verify authenticity.

    This function:
    - Reads the model bytes, signature, and public key from the directory.
    - Verifies the authenticity of the model bytes.
    - Deserializes the model bytes using `joblib.load`.

    Parameters
    ----------
    source: Path | str
        Directory containing model files.
    public_key: Ed25519PublicKey | Path | str | None, optional
        Public key (or path to `public.pem`) for verifying model bytes.
        If provided, the model bytes are verified using this key (instead
        of reading the public key from disk).
        (default = None)

    Returns
    -------
    Any
        Model object.

    Raises
    ------
    FileNotFoundError
        If model files are not found in the directory.
    InvalidSignature
        If verification fails.
    """
    _validate_model_directory(source, check_for_public_key=public_key is None)
    directory = Path(source)

    if public_key is None:
        # Load verification key from model directory
        public_key = _load_public_key(directory / AssetFileNames.PUBLIC_KEY)
    elif isinstance(public_key, Path | str):
        # Load public key from disk
        public_key = _load_public_key(public_key)
    elif not isinstance(public_key, ed25519.Ed25519PublicKey):
        raise TypeError(f"Invalid public key format; got {type(public_key)}")

    # Load signature
    with open(directory / AssetFileNames.SIGNATURE) as f:
        signature = bytes.fromhex(f.read())

    # Load model and verify
    model = _load_model(
        directory / AssetFileNames.MODEL, Verification(public_key, signature)
    )

    return model


def load_keypair(source: Path | str, password: str | bytes | None) -> KeyPair:
    """Load Ed25519 keypair from disk.

    This function:
    1. reads the private key from disk using PEM encapsulation format
    2. decrypts the key using the provided password
    3. derives the public key from the private key

    Parameters
    ----------
    source: Path | str
        Private key filepath.
    password: str | bytes | None
        Password for decrypting private key.
        NOTE: If password is `str`, it will be encoded to `bytes` using UTF-8.

    Returns
    -------
    KeyPair
        Private and public keypair.
    """
    with open(source, "rb") as f:
        # Load private key from disk
        private_key = serialization.load_pem_private_key(
            data=f.read(), password=_encode_password(password)
        )
        if not isinstance(private_key, ed25519.Ed25519PrivateKey):
            raise ValueError(f"Invalid public key format; got {type(private_key)}")

        # Derive the public key from the private key
        public_key = private_key.public_key()

        return KeyPair(private_key, public_key)


def save_private_key(
    private_key: ed25519.Ed25519PrivateKey,
    destination: Path | str,
    password: str | bytes | None,
) -> None:
    """Save private key to disk using PEM encapsulation format.

    Parameters
    ----------
    private_key: Ed25519PrivateKey
        Private key.
    destination: Path | str
        Destination filepath.
    password: str | bytes | None
        Password for encrypting private key.
        NOTE: If password is `str`, it will be encoded to `bytes` using UTF-8.

    Example
    -------
    >>> from tempfile import TemporaryDirectory
    >>> from r2seedo.io import generate_keypair, save_private_key
    >>> keypair = generate_keypair()
    >>> with TemporaryDirectory() as temp_dir:
    ...     key_path = f"{temp_dir}/private.pem"
    ...     save_private_key(keypair.private, key_path, password="password")
    """
    # Define encryption algorithm based on password
    encryption_algorithm: (
        serialization.BestAvailableEncryption | serialization.NoEncryption
    ) = serialization.NoEncryption()
    if (password_bytes := _encode_password(password)) is not None:
        encryption_algorithm = serialization.BestAvailableEncryption(password_bytes)

    with open(destination, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=encryption_algorithm,
            )
        )


def sign_n_save_model(
    model: Any,
    destination: Path | str,
    keypair: KeyPair,
    *,
    save_public_key: bool = True,
    overwrite: bool = False,
) -> None:
    """Sign model bytes and save {model, signature, public key} to directory.

    This function:
    - Serializes and saves a model to disk using `joblib.dump`.
    - Prior to dumping the model bytes, it signs them w/ the private key
    from the provided `keypair`.
    - It saves the signature and (optionally) public key to the same directory
    for verifying the authenticity of the model at load time.

    Parameters
    ----------
    model: Any
        Model object.
    directory: Path | str
        Directory to save model files.
    keypair: KeyPair
        Private and public keypair for signing model bytes.
    save_public_key: bool, optional
        Whether to save the public key to the directory.
        (default = True)
    overwrite: bool, optional
        Whether to overwrite existing files in the directory.
        (default = False)

    Raises
    ------
    FileExistsError
        If directory is not empty and `overwrite` is `False`.

    Notes
    -----
    - Model is saved as `model.joblib`.
    - Signature is saved as `signature.txt`
    - Public key is saved as `public.pem`.
    - Signature is written as hex string.
    - Public key is written in PEM encapsulation format.

    """
    # Validate destination directory
    directory = Path(destination)
    if directory.exists():
        if not overwrite and any(
            (directory / file_name).exists() for file_name in AssetFileNames
        ):
            raise FileExistsError(f"Directory not empty: {directory}")
    else:
        directory.mkdir(parents=False, exist_ok=False)

    # Save model
    receipt = _save_model(model, directory / AssetFileNames.MODEL, keypair.private)

    # Save signature
    if receipt.signature is None:
        raise ValueError("Model not signed.")
    with open(directory / AssetFileNames.SIGNATURE, "w") as f:
        # Write signature as hex string
        f.write(receipt.signature.hex())

    # Save public key
    if save_public_key:
        _save_public_key(keypair.public, directory / AssetFileNames.PUBLIC_KEY)


# Helpers
def _encode_password(password: str | bytes | None) -> bytes | None:
    """Encode password to bytes using UTF-8.

    Parameters
    ----------
    password: str | bytes | None
        Password for decrypting private key.

    Returns
    -------
    bytes | None
        Encoded password.
    """
    if password is None:
        return None
    if isinstance(password, str):
        return password.encode("utf-8")
    return password


def _load_model(source: Path | str, verification: Verification | None) -> Any:
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

    Raises
    ------
    InvalidSignature
        If verification fails.
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


def _load_public_key(source: Path | str) -> ed25519.Ed25519PublicKey:
    """Load Ed25519 public key from disk using PEM encapsulation format.

    Parameters
    ----------
    source: Path | str
        Public key filepath.

    Returns
    -------
    Ed25519PublicKey
        Public key.
    """
    with open(source, "rb") as f:
        key = serialization.load_pem_public_key(f.read())
        if not isinstance(key, ed25519.Ed25519PublicKey):
            raise ValueError("Invalid public key format.")
        return key


def _save_model(
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


def _save_public_key(
    public_key: ed25519.Ed25519PublicKey, destination: Path | str
) -> None:
    """Save public key to disk using PEM encapsulation format.

    Parameters
    ----------
    public_key: Ed25519PublicKey
        Public key.
    destination: Path | str
        Destination filepath.
    """
    with open(destination, "wb") as f:
        f.write(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )


def _validate_model_directory(
    directory: Path | str, *, check_for_public_key: bool = True
) -> None:
    """Validate model directory exists and has necessary files.

    Parameters
    ----------
    directory: Path
        Model directory.
    check_for_public_key: bool, optional
        Whether to check for the presence of a public key in directory.
        (default = True)

    Raises
    ------
    FileNotFoundError
        If model files are not found in the directory.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    model_path = directory / AssetFileNames.MODEL
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    signature_path = directory / AssetFileNames.SIGNATURE
    if not signature_path.is_file():
        raise FileNotFoundError(f"Signature not found: {signature_path}")

    public_key_path = directory / AssetFileNames.PUBLIC_KEY
    if check_for_public_key and not public_key_path.is_file():
        raise FileNotFoundError(f"Public key not found: {public_key_path}")
