## Usage

1. The workflow can be invoked by simply running `snakemake` under this directory.

2. The `explore.py` provides boilerplate code for check the sampled posterior values of elements.

Simply running `snakemake` will start the sampling workflow

## TODOs

1. The relaive imports in python files are temporarily managed by `pip install -e .`, since other `*.smk` files under workflow uses the format`jnotype.*`. Verify if this is enough.

2. Update the Snakefile to be `.smk` files. Make the workflow runnable on Euler.

3. Incorporate correct `beta` calculation.
