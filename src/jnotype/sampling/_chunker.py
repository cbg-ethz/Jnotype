"""Utilities for saving samples in chunks, to limit RAM usage."""
import abc

from datetime import datetime
from typing import Optional, Protocol, Union
from pathlib import Path

import numpy as np
import xarray as xr


class DatasetInterface(Protocol):
    """Interface for saving samples.

    One needs to implement:

    append_sample: appends a new sample
    end: executed at the end of sampling
    """

    def append_sample(self, sample: dict) -> None:
        """Adds a new sample to the data set.

        Note:
            This function *cannot* modify the `sample`.
        """
        pass

    def end(self) -> None:
        """Executed at the end of the sampling."""
        pass


class ListDataset(DatasetInterface):
    """Appends samples to a list."""

    def __init__(self, thinning: Optional[int] = None) -> None:
        """
        Args:
            thinning: thinning to be applied
        """
        self.thinning = thinning or 1
        self.samples = []
        self.iteration: int = 0

    def append_sample(self, sample: dict) -> None:
        """Appends a new sample to the list."""
        self.iteration += 1

        if self.iteration % self.thinning == 0:
            self.samples.append(sample)

    def end(self) -> None:
        """This function does nothing.
        It is just to make sure the interface
        is implemented."""
        return


class AbstractChunkedDataset(abc.ABC):
    """Abstract base class for chunked data sets.

    Usage:
        `append_sample` adds a new sample to the data set

    Note:
        The data set is automatically saved to the disk every `buffer_size` samples.
        We recommend running `save` method manually at the end of the sampling, so that
        an incomplete buffer is also saved.
    """

    def __init__(
        self,
        directory: Union[str, Path],
        buffer_size: int = 1_000,
        thinning: Optional[int] = None,
    ) -> None:
        """
        Args:
            directory: where the chunks will be saved
            buffer_size: controls how oftern the buffer will be saved to the disk
            thinning: whether thinning should be used.
                Set to `None` or 1 for no thinning (recommended)
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.thinning: int = thinning or 1
        self.buffer_size: int = buffer_size

        self.iteration: int = 0
        self.file_index: int = 1

        self._buffer: list[dict] = []
        # Sample index, shared across files, so that it's easier to concatenate them.
        self._sample_index: list[int] = []
        self._global_index: int = 1

    @abc.abstractmethod
    def _filename(self, file_index: int) -> str:
        """A pure function generating file name from `file_index` provided."""
        raise NotImplementedError

    @abc.abstractmethod
    def _save_buffer(
        self, buffer: list[dict], sample_index: list[int], filepath: Path
    ) -> None:
        """A pure function saving `buffer` and `sample_index` to `filepath`."""
        raise NotImplementedError

    def save(self) -> None:
        """Can be used to manually save the buffer."""
        if not len(self._buffer):
            return

        filepath = self.directory / self._filename(self.file_index)
        self.file_index += 1

        self._save_buffer(self._buffer, self._sample_index, filepath)
        self._reset_buffer()

    def _reset_buffer(self) -> None:
        """Empties the buffer. This function is not pure."""
        self._buffer = []
        self._sample_index = []

    def append_sample(self, sample: dict) -> None:
        """Adds a new sample to the buffer."""
        self.iteration += 1

        if self.iteration % self.thinning == 0:
            self._buffer.append(sample)
            self._sample_index.append(self._global_index)
            self._global_index += 1

        if len(self._buffer) >= self.buffer_size:
            self.save()

    def end(self) -> None:
        """Executed at the end of the training."""
        self.save()


class XArrayChunkedDataset(AbstractChunkedDataset):
    """A chunked data set based on Xarray's Dataset utilities
    and saving the samples to NetCDF format."""

    def __init__(
        self,
        directory: Union[str, Path],
        basic_dimensions: dict[str, Union[tuple[str, ...], list[str]]],
        *,
        buffer_size: int = 1000,
        thinning: Optional[int] = None,
        attrs: Optional[dict] = None,
        coords: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Args:
            basic_dimensions: specify the dimensions of each item in the sample

        Example:
            If in one sample you have array "coordinates"
              it can have shape ("dimension",).
              Then you should specify {"coordinates": ["dimension"]}
        """
        # TODO(Pawel): Check for unspecified behaviour
        #   when a sample is array(3.0), i.e., of shape (,)
        super().__init__(
            directory=directory, buffer_size=buffer_size, thinning=thinning
        )

        self._attrs = attrs or {}
        self._coords = coords or {}
        self._basic_dimensions = basic_dimensions

    def _filename(self, file_index: int) -> str:
        return f"{file_index:05}.nc"

    def append_sample(self, sample: dict) -> None:
        """Appends a new sample.

        Raises:
            KeyError, if the sample has different keys than declared
        """
        # Check if the keys are right
        if self.iteration < 3 or self.iteration % 100 == 0:
            if set(self._basic_dimensions.keys()) != set(sample.keys()):
                msg = (
                    f"Keys mismatch: {self._basic_dimensions.keys()} "
                    f"!= {sample.keys()}."
                )
                raise KeyError(msg)

        super().append_sample(sample)

    @staticmethod
    def _extract_from_buffer(buffer: list[dict], label: str) -> np.ndarray:
        return np.asarray([item[label] for item in buffer])

    def _coords_for_label(self, label: str) -> list[str]:
        if label not in self._basic_dimensions:
            raise ValueError(f"Label {label} does not have dimensions assigned.")
        return ["sample"] + list(self._basic_dimensions[label])

    def _buffer_to_dataset(
        self, buffer: list[dict], sample_index: list[int]
    ) -> xr.Dataset:
        attrs = {
            "thinning": self.thinning,
            "n_samples_in_batch": len(buffer),
            "timestamp_save": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        } | self._attrs

        coords = {
            "sample": np.asarray(sample_index),
        } | self._coords

        variables = {
            label: (
                self._coords_for_label(label),
                self._extract_from_buffer(buffer=buffer, label=label),
            )
            for label in self._basic_dimensions.keys()
        }

        return xr.Dataset(
            data_vars=variables,
            coords=coords,
            attrs=attrs,
        )

    def _save_buffer(
        self, buffer: list[dict], sample_index: list[int], filepath: Path
    ) -> None:
        dataset = self._buffer_to_dataset(buffer, sample_index)
        dataset.to_netcdf(filepath)
