[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

# Jnotype

[JAX](https://github.com/google/jax)-powered Python package for exploratory analysis of binary data.
This includes genotype data, binary images, and data sets used in ecology.

**Note:** this package is in early development stage.


## Installation

**Note:** The package has not reached a stable API yet. Frequent changes may appear 

We recommend setting up a new [virtual environment](https://docs.python.org/3/library/venv.html). You can install the package using PIP:

```bash
$ python -m pip install 'jnotype @ git+https://github.com/cbg-ethz/jnotype'
```

However, this will install the CPU version of JAX.
To use GPU, you may need to follow the [official JAX installation tutorial](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier).

## Getting started

Directory `examples/` contains [Quarto](https://quarto.org/) notebooks, which demonstrate basic functionalities of the package.

