# r2seedo

R2 See, Do - a journey in deep reinforcement learning.

## Table of contents

- [Notebooks](notebooks/README.md)
- [Project Setup](#project-setup)
- [RL Environments](#create-separate-virtual-environments-for-specific-rl-environments)
- [Running Tests](#running-tests)

## Project Setup

### 0. Install `hatch` & `hatch-conda`

`r2seedo` uses [`hatch`](https://hatch.pypa.io/latest/) for project management. You'll
need it installed (ideally in an isolated environment) before setting up `r2seedo`.

#### (optional) Install `hatch-conda`

`r2seedo` depends on [`gymnasium`](https://gymnasium.farama.org/), which
is best installed with conda / mamba, thus [`hatch-conda`](https://github.com/OldGrumpyViking/hatch-conda)
is also recommended.

```sh
# Update mamba
mamba update -n base mamba

# Update base environment packages 
mamba update -n base --all

# Install hatch & hatchling
mamba install -n base -c conda-forge hatch hatchling

# Install hatch-conda
mamba activate base && python -m pip install hatch-conda
```

### 1. Clone the `r2seedo` repository

```sh
git clone git@github.com:libertininick/r2seedo.git
```

### 2. Create the default (virtual) environment

`hatch` will install `r2seedo` in development mode along with its development dependencies
inside of a virtual environment managed by `hatch`.

```sh
# Navigate to root project directory
cd r2seedo

# (optional) if using conda / mamba envs activate the base environment
mamba activate base

# Create default environment
hatch env create
```

[Table of Contents](#table-of-contents)

## Create separate virtual environments for specific RL environments

- RL environment packages are best installed in independent virtual environment
- Each RL environment used in `r2seedo` has its own virtual environment configuration
defined in [pyproject.toml](pyproject.toml)
- An environment can be created using `hatch env create <env name>` :

NOTE: Make sure the base environment is activated before creating a new environment: `mamba activate base`

| Environment | Name | Create |
|-------------|------|--------|
| [Atari](https://gymnasium.farama.org/environments/atari/) | gym-atari | `hatch env create gym-atari` |
| [Box2D](https://gymnasium.farama.org/environments/box2d/) | gym-box2d | `hatch env create gym-box2d` |
| [Toy Text](https://gymnasium.farama.org/environments/toy_text/) | gym-toy_text | `hatch env create gym-toy_text` |

[Table of Contents](#table-of-contents)

## Running Tests

Run tests and coverage report using `hatch`

```sh
hatch run default:test-cov
```

[Table of Contents](#table-of-contents)
