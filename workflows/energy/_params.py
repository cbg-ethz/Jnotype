# -------------------------
# PARAMETERS
# -------------------------
N_VALUES = [1000]  # Different sample sizes
G_LIST = [3]
RANDOM_SEED = 42
SPIKE_WEIGHT = 0.1  # probability that off-diagonal entries are non-zero
SPIKE_SCALE = 10.0  # scale for the off-diagonal spike normal distribution
DIAG_SCALE = 5.0  # scale for the diagonal normal distribution
SLAB_PRIOR = 0.01  # scale for the slab prior in Bayes
NUM_WARMUP = 200  # number of warmup iterations for MCMC
NUM_SAMPLES = 1000  # number of samples for MCMC
NUM_CHAINS = 4
