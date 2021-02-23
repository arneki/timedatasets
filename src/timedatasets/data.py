"""
Classes and functions to deal with data in classification experiments.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence, Iterable as ABCIterable
import dataclasses
from enum import Enum, IntEnum
import inspect
from itertools import chain
import logging
from pathlib import Path
import pickle
from typing import Any, List, Optional, Union, Tuple, Type, TypeVar, Iterable,\
    Dict, Callable
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal

from timedatasets import common, helpers


class Label(IntEnum):
    pass


class Channel(Enum):
    """
    A data channel.
    """
    @staticmethod
    def _generate_next_value_(name, *_):
        return name


TimeSampleType = TypeVar('TimeSampleType', bound='TimeSample', covariant=True)


@dataclasses.dataclass(eq=False, repr=False, frozen=True)
class TimeSample(ABC):
    """
    A generic sample used in classification of time-based data.

    :param data:        The actual data to classify.
    :param label:       The corresponding label.
    :param sample_rate: Sample rate of the data.
    :param filename:    Name of the file the data was originally read from.
    """

    data: Dict[Channel, np.ndarray]
    label: Label
    sample_rate: float
    filename: str

    def __post_init__(self):
        data_iter = iter(self.data.values())
        first_dtype = next(data_iter).dtype
        if any(first_dtype != d.dtype for d in data_iter):
            raise TypeError("channel data has to be of the same dtype")

    @abstractmethod
    def __len__(self) -> int:
        """
        The total length of the sample.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, subscript: slice) -> TimeSample:
        """
        Sliced access to the data.
        """
        raise NotImplementedError

    @abstractmethod
    def __add__(self: TimeSampleType, other: TimeSample) -> TimeSampleType:
        raise NotImplementedError

    @abstractmethod
    def resampled(self: TimeSampleType, sample_rate: float) -> TimeSampleType:
        """
        Returns a resampled version of the sample.
        """
        raise NotImplementedError

    def equal_without_data(self, other: TimeSample) -> bool:
        """
        Compares everything except of the raw data and length.
        """
        if self is other:
            return True
        if self.__class__ is not other.__class__:
            return False
        # compare channels
        if tuple(self.data.keys()) != tuple(other.data.keys()):
            return False
        # compare fields (except length and data)
        dict1 = dataclasses.asdict(self)
        dict2 = dataclasses.asdict(other)
        keys_to_compare = filter(
            lambda key: key not in ("length", "data"), dict1.keys())
        for key in keys_to_compare:
            if not dict1[key] == dict2[key]:
                return False
        return True

    def __eq__(self, other) -> bool:
        if not self.equal_without_data(other):
            return False
        return all(helpers.array_safe_equal(d1, d2)
                   for d1, d2 in zip(self.data.values(), other.data.values()))

    @staticmethod
    def _data_repr(channel_data: np.ndarray) -> str:
        """
        Allows modification of the data representation string.
        """
        with np.printoptions(threshold=5, precision=3):
            return repr(channel_data)

    def __repr__(self) -> str:
        repr_rows = [f"length={len(self)}"]
        for field in dataclasses.fields(self):
            if field.repr:
                value = getattr(self, field.name)
                if field.name == "data":
                    value = ",\n        ".join(
                        f"{str(ch)}: {self._data_repr(d)}"
                        for ch, d in value.items())
                    repr_rows.append(f"data={{\n        {value}\n    }}")
                else:
                    repr_rows.append(f"{field.name}={value!r}")
        repr_str = ",\n    ".join(repr_rows)
        return f"{self.__class__.__name__}(\n    {repr_str}\n)"

    def replace(self: TimeSampleType, **kwargs) -> TimeSampleType:
        """
        Returns a new object replacing the specified fields with new values.
        """
        return dataclasses.replace(self, **kwargs)

    @property
    def channels(self) -> List[Channel]:
        """
        All contained channels.
        """
        return list(self.data.keys())

    @property
    def dtype(self) -> np.dtype:
        """
        The dtype of the data.
        """
        return next(iter(self.data.values())).dtype

    @abstractmethod
    def plot(self, axes: Optional[mpl.axes.Axes] = None, **kwargs):
        """
        Plot the sample.

        :param axes: The matplotlib axes, defaults to current one.
        :param kwargs: Parameters for pyplot's plot function.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def empty_like(cls: Type[TimeSampleType], sample: TimeSample,
                   length: Optional[int] = None) -> TimeSampleType:
        """
        Create a new sample with the same properties as the given sample and
        empty data of given length.

        :param sample: Sample to replicate properties from
        :param length: Length of the returned sample (default: ``len(sample)``)
        """
        raise NotImplementedError

    def as_empty(self: TimeSampleType, length: Optional[int] = None
                 ) -> TimeSampleType:
        """
        Returns an empty version of the sample.

        :param length: Length of the returned sample (default: ``len(self)``)
        """
        return self.__class__.empty_like(self, length)

    @abstractmethod
    def to_tuple(self) -> Tuple[Any, Label]:
        """
        Returns a tuple containing data and label to be used in ML frameworks.
        """
        raise NotImplementedError


@dataclasses.dataclass(eq=False, repr=False, frozen=True)
class TimeseriesSample(TimeSample):
    """
    A generic sample used in classification of timeseries data.
    The data is a continuous sequence that contains evenly distributed values.
    """

    def __post_init__(self):
        super().__post_init__()
        data_iter = iter(self.data.values())
        first_length = len(next(data_iter))
        if any(first_length != len(d) for d in data_iter):
            raise ValueError("channel data has to be of the same length")

    def __len__(self) -> int:
        return len(next(iter(self.data.values())))

    def __getitem__(self, subscript: slice) -> TimeseriesSample:
        if not isinstance(subscript, slice):
            raise TypeError("__getitem__ provides sliced access only")
        return self.replace(
            data={ch: d[subscript] for ch, d in self.data.items()},
            sample_rate=self.sample_rate / (subscript.step or 1))

    def __add__(self, other: TimeSample) -> TimeseriesSample:
        if not self.equal_without_data(other):
            raise ValueError("everything except of the data has to match")
        return self.replace(data={ch: np.concatenate((d, other.data[ch]))
                                  for ch, d in self.data.items()})

    @property
    def time(self) -> np.ndarray:
        """
        Returns the time of all datapoints according to the sample rate.
        """
        return np.arange(len(self)) / self.sample_rate

    def plot(self, axes: Optional[mpl.axes.Axes] = None, **kwargs):
        ax1 = axes or plt.gca()
        for channel, trace in self.data.items():
            ax1.plot(self.time, trace, label=channel.name,
                     **kwargs)
        ax1.set_xlabel('time [s]')
        ax1.label_outer()

    @classmethod
    def empty_like(cls, sample: TimeSample, length: Optional[int] = None
                   ) -> TimeseriesSample:
        length = len(sample) if length is None else length
        return cls(
            data={ch: np.zeros(length, sample.dtype) for ch in sample.data},
            label=sample.label,
            filename=sample.filename,
            sample_rate=sample.sample_rate)

    def to_tuple(self) -> Tuple[np.ndarray, Label]:
        return (np.stack(tuple(self.data.values())), self.label)

    def resampled(self: TimeSampleType, sample_rate: float) -> TimeSampleType:
        target_length = round(len(self) / self.sample_rate * sample_rate)
        return self.replace(
            data={channel: scipy.signal.resample(trace, target_length)
                  for channel, trace in self.data.items()},
            sample_rate=sample_rate)


@dataclasses.dataclass(eq=False, repr=False, frozen=True)
class SpikingSample(TimeSample):
    """
    A generic sample used in classification of spike data.
    The data is a sequence that holds spike positions as integers.

    :param length: Total number of covered time steps.
    """

    length: int = dataclasses.field(repr=False)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, subscript: slice) -> SpikingSample:
        if not isinstance(subscript, slice):
            raise TypeError("__getitem__ provides sliced access only")

        start, stop, step = subscript.indices(len(self))
        new_data = {}
        for channel, spikes in self.data.items():
            new_data[channel] = \
                (spikes[(spikes >= start)
                        & (spikes < stop)
                        & (spikes % step == 0)] - start) // step
        return self.replace(data=new_data,
                            length=(stop - start) // step,
                            sample_rate=self.sample_rate / step)

    def __add__(self, other: TimeSample) -> SpikingSample:
        if not self.equal_without_data(other):
            raise ValueError("Everything except of the data has to match")

        new_data = {ch: np.concatenate((ds, do + len(self))) for ch, ds, do
                    in zip(self.data, self.data.values(), other.data.values())}
        return self.replace(data=new_data, length=len(self) + len(other))

    def plot(self, axes: Optional[mpl.axes.Axes] = None, **kwargs):
        """
        Plot the spike positions as vertical lines.
        """
        ax1 = axes or plt.gca()
        for channel, spikes in self.data.items():
            for spike_pos in spikes[spikes >= 0]:
                ax1.axvline(spike_pos / self.sample_rate, label=channel.name,
                            **kwargs)
        ax1.set_xlabel('time [s]')
        ax1.label_outer()

    @classmethod
    def empty_like(cls, sample: TimeSample, length: Optional[int] = None
                   ) -> SpikingSample:
        length = len(sample) if length is None else length
        return cls(
            data={ch: np.empty(0, sample.dtype) for ch in sample.data},
            label=sample.label,
            filename=sample.filename,
            sample_rate=sample.sample_rate,
            length=length)

    def to_tuple(self) -> Tuple[Tuple[np.ndarray], Label]:
        return (tuple(self.data.values()), self.label)

    def resampled(self: TimeSampleType, sample_rate: float) -> TimeSampleType:
        if sample_rate <= 0:
            raise ValueError("Sample rate has to be positive.")
        factor = sample_rate / self.sample_rate
        return self.replace(
            data={channel: (positions * factor).astype(positions.dtype)
                  for channel, positions in self.data.items()},
            sample_rate=sample_rate)


class Dataset(Sequence):
    """
    Class to hold samples that may be altered by a transformation.
    """

    def __init__(self, samples: Iterable[TimeSample]):
        """
        :param samples: Iterable containing all samples
        """
        self.samples = list(samples)

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if self.__class__ is not other.__class__:
            return NotImplemented
        return all(sample1 == sample2 for sample1, sample2 in zip(self, other))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: Union[int, slice, Iterable[int]]
                    ) -> Union[TimeSample, Dataset]:
        if isinstance(idx, int):
            return self.samples[idx]
        # allow subsets from iterable of indices
        if isinstance(idx, ABCIterable):
            return type(self)(self.samples[i] for i in idx)
        # allow slicing
        return type(self)(self.samples[idx])

    def __add__(self, other: Dataset) -> Dataset:
        return type(self)(chain.from_iterable([self, other]))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(samples.length={len(self)})"

    def to_tuple(self) -> Tuple[Tuple[Any, Label]]:
        """
        Returns the dataset as a tuple containing the samples as tuples.
        """
        return tuple(map(lambda sample: sample.to_tuple(), self))


class Transform(ABC, common.UniquelyIdentifiable):
    """
    Abstract base class for arbitrary transformations on datasets.
    """

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Dataset:
        """
        :param dataset: The dataset to transform.

        :return: Transformed dataset
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        keys = list(inspect.signature(self.__class__.__init__).parameters)[1:]
        key_strings = [f"{key}={self.__dict__.get(key)!r}" for key in
                       filter(lambda key: key in self.__dict__, keys)]
        repr_str = f"{self.__class__.__name__}"
        if key_strings:
            repr_str += f"({', '.join(key_strings)})"

        return repr_str

    def __eq__(self, other):
        if self is other:
            return True
        if self.__class__ is not other.__class__:
            return NotImplemented
        # numpy arrays need some extra care
        shared = tuple(k for k in self.__dict__ if k in other.__dict__
                       and helpers.array_safe_equal(self.__dict__[k],
                                                    other.__dict__[k]))
        return len(shared) == len(self.__dict__)


class DatasetLoader(ABC, common.UniquelyIdentifiable):
    """
    A loader that loads data from a given path and returns a dataset.
    """
    @property
    @abstractmethod
    def _available_labels(self) -> List[Label]:
        """
        All available labels.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _available_channels(self) -> List[Channel]:
        """
        All available channels.
        """
        raise NotImplementedError

    def __init__(self, path: Path,
                 channels: Optional[List[Channel]] = None,
                 labels: Optional[List[Label]] = None):
        """
        :param path: Path of the dataset.
        :param channels: The channels to include (defaults to all).
        :param labels: The labels to include (defaults to all).
        """
        labels = labels or self._available_labels
        if not all(label in self._available_labels for label in labels):
            raise ValueError("Given labels have to be a subset of "
                             f"{[str(lb) for lb in self._available_labels]}.")
        channels = channels or self._available_channels
        if not all(ch in self._available_channels for ch in channels):
            raise ValueError("Given channels have to be a subset of "
                             f"{[str(c) for c in self._available_channels]}.")
        self.path = path
        self.channels = channels
        self.labels = labels

    @abstractmethod
    def load_train(self) -> Dataset:
        """
        Load the training dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def load_test(self) -> Dataset:
        """
        Load the test dataset.
        """
        raise NotImplementedError


class SplitDatasetLoader(DatasetLoader):
    """
    Wrapper that splits a fraction from the training set and returns it as
    test data for validation.
    """

    def __init__(self, data_loader: DatasetLoader,
                 validation_split: float = 0.2, seed: int = 0x5EEEED):
        """
        :param data_loader: The dataloader to wrap. The test data will be
            discarded and replaced by a fraction of the train data.
        :param validation_split: The fraction to use as validation data.
        :param seed: Seed to initialize the random number generator.
        """
        self.validation_split = validation_split
        self.seed = seed
        self._data_loader = data_loader
        self._dataset: Optional[Dataset] = None
        super().__init__(data_loader.path)

    @property
    def _available_labels(self) -> List[Label]:
        return self._data_loader.labels

    @property
    def _available_channels(self) -> List[Channel]:
        return self._data_loader.channels

    def load_train(self) -> Dataset:
        if self._dataset is None:
            self._dataset = self._data_loader.load_train()
        return split_dataset(
            self._dataset, self.validation_split, self.seed)[0]

    def load_test(self) -> Dataset:
        if self._dataset is None:
            self._dataset = self._data_loader.load_train()
        return split_dataset(
            self._dataset, self.validation_split, self.seed)[1]


class TransformDatasetLoader(DatasetLoader):
    """
    Wrapper that returns the transformed data of the provided data loader.
    If a cache directory is provided, it tries to use a cached version and will
    build the cache if it does not exist.
    """

    def __init__(self, data_loader: DatasetLoader,
                 transform: Optional[Transform] = None,
                 cache_dir: Optional[Path] = None, *,
                 test_transform: Optional[Transform] = None,
                 cache_interim_results: bool = False):
        """
        :param data_loader: The dataloader to wrap.
        :param transform: Transformation to be applied to the train data.
        :param cache_dir: Directory that will be used for caching. Setting to
            ``None`` will disable caching.
        :param test_transform: Transformation to be applied to the test data,
            defaults to ``transform``.
        :param cache_interim_results: If set to ``True``, also interim results
            from a composed transformation will be cached. This allows
            changes to the preprocessing chain without processing everything
            from the beginning, but increases the cache size.
        """
        self._data_loader = data_loader
        self.cache_dir = cache_dir
        self.train_transform = transform
        self.cache_interim_results = cache_interim_results
        self.test_transform = test_transform or transform
        super().__init__(data_loader.path)

    @property
    def _available_labels(self) -> List[Label]:
        return self._data_loader.labels

    @property
    def _available_channels(self) -> List[Channel]:
        return self._data_loader.channels

    def _load_cached(self, load_fn: Callable[[], Dataset],
                     transform: Optional[Transform],
                     cache_suffix: str,
                     cache_result: bool = True) -> Dataset:
        """
        :param load_fn: Function to load the raw dataset
        :param transform: The transformation to be applied to the dataset
        :param cache_suffix: Suffix that will be appended to the cache filename
        :param cache_result: Whether the result should be cached
        """
        # get path of the cache file from uuid of loader and transformation:
        cache_key = self._data_loader.uuid
        if transform is not None:
            cache_key = Transform.concat_uuids([cache_key, transform.uuid])
        cache_file = self.cache_dir.joinpath(
            f"{cache_key}{cache_suffix}.dataset")

        if cache_file.exists():
            logging.info(f'Load cached data from "{cache_file.name}"')
            with cache_file.open(mode="rb") as dataset_file:
                return pickle.load(dataset_file)

        if cache_key == self._data_loader.uuid:
            # transform is the identity
            logging.info(
                f'Load raw data from "{self._data_loader.path}"')
            dataset = load_fn()
        elif isinstance(transform, Sequence):
            # sequential transformations will be recursively cached
            dataset = self._load_cached(
                load_fn, transform[:-1],
                cache_suffix=cache_suffix,
                cache_result=self.cache_interim_results,
            )
            dataset = transform[-1](dataset)
        else:
            # single transformation that is not already cached
            dataset = self._load_cached(
                load_fn,
                transform=None,
                cache_suffix=cache_suffix,
                cache_result=self.cache_interim_results,
            )
            if transform is not None:
                dataset = transform(dataset)
        if cache_result:
            logging.info(f'Cache dataset to "{cache_file}"')
            with cache_file.open(mode="wb") as dataset_file:
                pickle.dump(dataset, dataset_file)
        return dataset

    def _load(self, load_fn: Callable[[], Dataset],
              transform: Optional[Transform], cache_suffix: str) -> Dataset:
        """
        :param load_fn: Function to load the raw dataset
        :param transform: The transformation to be applied to the dataset
        :param cache_suffix: Suffix that will be appended to the cache filename
        """
        if self.cache_dir is None:
            return transform(load_fn()) if transform is not None else load_fn()
        return self._load_cached(load_fn, transform, cache_suffix=cache_suffix)

    def load_train(self) -> Dataset:
        return self._load(
            self._data_loader.load_train, self.train_transform,
            cache_suffix="_train",
        )

    def load_test(self) -> Dataset:
        return self._load(
            self._data_loader.load_test, self.test_transform,
            cache_suffix="_test",
        )


def split_dataset(dataset: Dataset, validation_split: float = 0.2,
                  seed: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Split a dataset randomly in training and validation data.
    Keeps the relative distribution of classes.

    :param dataset: Dataset to split.
    :param validation_split: The fraction to use as validation data.
    :param seed: Seed to initialize the random number generator.

    :returns: train data, validation data
    """

    if seed is not None:
        np.random.seed(seed)

    class_indices = defaultdict(list)
    sample: TimeSample
    for idx, sample in enumerate(dataset):
        label = sample.label if hasattr(sample, "label") else sample[1]
        class_indices[label].append(idx)

    training_indices = list()
    validation_indices = list()
    for indices in class_indices.values():
        indices = np.random.permutation(indices).tolist()
        split_idx = round(len(indices) * validation_split)
        training_indices += indices[split_idx:]
        validation_indices += indices[:split_idx]

    return dataset[training_indices], dataset[validation_indices]
