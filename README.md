# r2seedo

R2 See, Do - a journey in deep reinforcement learning.

## Table of contents

- [Notebooks](notebooks/README.md)
- [Project Setup](#project-setup)
- [Update Environment](#update-environment)
- [Running Tests](#running-tests)

## Project Setup

### 0. Install `hatch` & `hatch-conda`

`r2seedo` uses [`hatch`](https://hatch.pypa.io/latest/) for project management. You'll
need it installed (ideally in an isolated environment) before setting up `r2seedo`.

(optional) This `r2seedo` depends on [`gymnasium`](https://gymnasium.farama.org/), which is best install with conda,
thus [`hatch-conda`](https://github.com/OldGrumpyViking/hatch-conda) is also recommended.

### 1. Clone the `r2seedo` repository

```sh
git clone git@github.com:libertininick/r2seedo.git
```

### 2. Create the default (virtual) environment

`hatch` will install `r2seedo` in development mode along with its development dependencies
inside of a virtual environment managed by `hatch`.

```sh
cd r2seedo
hatch env create
```

[Table of Contents](#table-of-contents)

## Update Environment

If you make changes to [environment.yml](environment.yml) and want to update the default
environment:

```sh
hatch run default:update
```

[Table of Contents](#table-of-contents)

## Running Tests

Run tests and coverage report using `hatch`

```sh
hatch run default:test-cov
```

[Table of Contents](#table-of-contents)
