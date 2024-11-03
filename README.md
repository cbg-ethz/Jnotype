[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![build](https://github.com/cbg-ethz/Jnotype/actions/workflows/test.yml/badge.svg)](https://github.com/cbg-ethz/Jnotype/actions/workflows/test.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI Latest Release](https://img.shields.io/pypi/v/jnotype.svg)](https://pypi.org/project/jnotype/)

# Jnotype

[JAX](https://github.com/google/jax)-powered Python package for exploratory analysis of binary data.
This includes genotype data, binary images, and data sets used in ecology.

**Note:** this package is in early development stage.

  - **Source code:** [https://github.com/cbg-ethz/Jnotype](https://github.com/cbg-ethz/Jnotype)
  - **Bug reports:** [https://github.com/cbg-ethz/Jnotype/issues](https://github.com/cbg-ethz/Jnotype/issues)
  - **PyPI package**: [https://pypi.org/project/jnotype/](https://pypi.org/project/jnotype/)

## Installation

**Note:** The package has not reached a stable API yet. Frequent changes may appear.

We recommend setting up a new [virtual environment](https://docs.python.org/3/library/venv.html).
You can install the released version of the package from PyPI:

```bash
$ python -m pip install jnotype
```

or install the development version from GitHub:

```bash
$ python -m pip install 'jnotype @ git+https://github.com/cbg-ethz/jnotype'
```

### Using GPU

Instructions above install the CPU version of JAX.
To use GPU, you may need to follow the [official JAX installation tutorial](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier).

## Getting started

Directory `examples/` contains [Quarto](https://quarto.org/) notebooks, which demonstrate basic functionalities of the package.

