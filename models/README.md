# Models

## Trained Models

### Q-Learning

- [Frozen Lake](q-table-frozen-lake/README.md)

### Deep Q-Learning

- [Space Invaders](deep-q-space-invaders/README.md)

## Policy Optimization

- [Lunar Lander](lunar-lander/README.md)

## Saving & Loading Models

### 1. Generate a keypair for signing models

If you want to sign a model with a private key and then include a public
key for verification of your signature you need first generate a key pair

```python
from dotenv import dotenv_values
from r2seedo.io import generate_keypair, save_private_key

# Generate a Ed25519 private key and its public derivative
key_pair = generate_keypair()

# Encrypt and save your private key
env_config = dotenv_values()

save_private_key(
    key_pair.private,
    destination="path/to/private.pem",  # path to save secrete key
    password=env_config["MODEL_KEY_PASSWORD"] # password
)
```

### 2. Sign and save a model

Sign model bytes and save {model, signature, public key} to directory.

- Serializes and saves a model to disk using `joblib.dump`.
- Prior to dumping the model bytes, it signs them w/ the private key
from the provided `keypair`.
- It saves the signature and (optionally) public key to the same directory
for verifying the authenticity of the model at load time.

```python
from dotenv import dotenv_values
from r2seedo.io import generate_keypair, save_private_key

env_config = dotenv_values()

sign_n_save_model(
    model=...,  # Python model object
    destination="path/to/my_model",  # Folder name for model
    keypair=load_keypair(
        "path/to/private.pem",
        password=env_config["MODEL_KEY_PASSWORD"]
    ),
)
```

### 3. Load and verify model

Load model from directory and verify authenticity.

- Reads the model bytes, signature, and public key from the directory.
- Verifies the authenticity of the model bytes.
- Deserializes the model bytes using `joblib.load`.

```python
from dotenv import dotenv_values
from r2seedo.io import load_n_verify_model

model = load_n_verify_model("path/to/my_model")
```
