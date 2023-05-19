"""Generic Gibbs sampler."""
import abc
import logging
import time
from typing import Optional, Sequence

import tqdm

from jnotype.sampling._chunker import DatasetInterface

_LOGGER = logging.getLogger(__name__)


class AbstractGibbsSampler(abc.ABC):
    """Abstract Gibbs sampler.

    All children classes should implement:
      dimensions: describes the sample and the shapes
      new_sample: Markov chain transition to a new point

    """

    def __init__(
        self,
        datasets: Sequence[DatasetInterface],
        *,
        warmup: int = 2_000,
        steps: int = 3_000,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            datasets: data sets storing the samples
        """
        self.datasets = list(datasets)
        self.warmup = warmup
        self.steps = steps
        self.verbose = verbose

    @abc.abstractclassmethod
    def dimensions(cls) -> dict:
        """Returns dictionary describing
        the dimensions, e.g.,:
        {
            "coeffs": ["n_outputs", "n_inputs"],
            "intercepts": ["n_outputs"],
        }
        """
        raise NotImplementedError

    @abc.abstractmethod
    def new_sample(self, sample: dict) -> dict:
        """Transition to a new sample."""
        raise NotImplementedError

    @abc.abstractmethod
    def initialise(self) -> dict:
        """Initializes the sample."""
        raise NotImplementedError

    def _append(self, sample: dict) -> None:
        """Appends the sample to all data sets."""
        for dataset in self.datasets:
            dataset.append_sample(sample)

    def _end_run(self) -> None:
        """Signalizes to all data sets that sampling
        has finished."""
        for dataset in self.datasets:
            dataset.end()

    def _init_sample(self, start: Optional[dict]) -> dict:
        """Initialise first sample using `initialise` method
        and overwriting the fields with `start`.

        Raises:
            KeyError, if there are fields in `start` which are not
              in the sample from `initialise`
        """
        # Initialise first sample
        sample = self.initialise()
        start = start or {}
        if not set(start.keys()).issubset(sample.keys()):
            raise KeyError(
                f"Init keys: {sample.keys()}. Tried to update with: {start.keys()}."
            )
        sample.update(start)

        return sample

    def run(self, start: Optional[dict] = None) -> None:
        """Full Gibbs sampling run."""
        # Initialise the sample
        _LOGGER.info("Initialising the first sample...")
        sample = self._init_sample(start)

        # Warmup period
        _LOGGER.info(f"Starting warmup period with {self.warmup} steps...")
        t0 = time.time()

        for _ in tqdm.tqdm(
            range(self.warmup), total=self.warmup, disable=not self.verbose
        ):
            sample = self.new_sample(sample)

        dt = time.time() - t0
        _LOGGER.info(
            f"Warmup finished in {dt:.1f} seconds "
            f"({(self.warmup/dt):.1f} steps/s). "
            f"Starting proper sampling..."
        )

        t1 = time.time()
        for _ in tqdm.tqdm(
            range(1, self.steps + 1), total=self.steps, disable=not self.verbose
        ):
            sample = self.new_sample(sample)
            self._append(sample)

        t2 = time.time()
        dt = t2 - t1

        _LOGGER.info(
            f"Finished sampling in {dt:.1f} seconds." f"({self.steps/dt:.1f}) steps/s"
        )

        self._end_run()
        _LOGGER.info(f"Run finished in {t2-t0:.1} seconds.")
