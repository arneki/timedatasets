"""
Callable classes to transform EcgData.
"""
from abc import abstractmethod
import collections.abc
from collections import defaultdict
import numbers
from operator import itemgetter
from typing import Union, Callable, Dict, List, Type, Optional, Tuple, final
import uuid
import numpy as np
from scipy.ndimage.morphology import grey_opening, grey_closing
from scipy.signal import find_peaks

from timedatasets import data
from timedatasets.data import Transform


@final
class Identity(Transform):
    """
    The identity transformation.

    This transformation does not modify the dataset and its uuid is the eins of
    the `concat_uuid` operation.
    """

    @property
    def uuid(self):
        return self.neutral_uuid

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        return dataset


@final
class Compose(Transform, collections.abc.Sequence):
    """
    Composes several transforms together.
    """

    def __init__(self, *transforms: Transform):
        """
        :param transforms: List of transforms.
        """
        self.transforms = transforms

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        for transform in self.transforms:
            dataset = transform(dataset)
        return dataset

    def __len__(self) -> int:
        return len(self.transforms)

    def __getitem__(self, idx: Union[int, slice]) -> Transform:
        # allow slicing
        if isinstance(idx, slice):
            return self.__class__(*self.transforms[idx])
        return self.transforms[idx]

    @property
    def uuid(self) -> uuid.UUID:
        return self.concat_uuids(t.uuid for t in self.transforms)


class SpikesToTimeseries(Transform):
    """
    Transforms the given spike train to a continuous boolean sample.
    """
    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        samples = []
        for sample in dataset:
            if not isinstance(sample, data.SpikingSample):
                raise TypeError(f"{sample} is not a SpikingSample")
            new_data = {}
            for channel, spikes in sample.data.items():
                new_data[channel] = np.zeros(len(sample), bool)
                new_data[channel][spikes] = True
            samples.append(data.TimeseriesSample.empty_like(sample).replace(
                data=new_data))
        return data.Dataset(samples)


class BalanceClasses(Transform):
    """
    Balances the classes in the dataset by randomly downsample / upsample the
    number of samples per class.
    """

    def __init__(self, seed: int, resample_factor: float = 0.):
        """
        :param seed: Seed for the random number generator
        :param resample_factor: Controls the resample behavior in the range
            from ``-1`` to ``1``. Its value corresponds to:
            ``-1.``: Downsample every class to the number in the rarest
            ``1.``: Upsample every class to the number in the most abundant
        """
        self.seed = seed
        if abs(resample_factor) > 1:
            raise ValueError(
                "Resample factor has to be in the interval [-1, 1]")
        self.resample_factor = resample_factor

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        np.random.seed(self.seed)
        class_n: Dict[data.Label, int] = defaultdict(lambda: 0)
        class_indices: Dict[data.Label, List[int]] = defaultdict(list)
        for idx, sample in enumerate(dataset):
            class_n[sample.label] += 1
            class_indices[sample.label].append(idx)
        samples_per_class = round(
            (max(class_n.values()) * (self.resample_factor + 1)
             - min(class_n.values()) * (self.resample_factor - 1)) / 2)

        new_indices = []
        for indices in class_indices.values():
            samples_to_add = samples_per_class
            indices = np.random.permutation(indices).tolist()
            while samples_to_add > 0:
                new_indices += indices[:samples_to_add]
                samples_to_add -= len(indices)

        return data.Dataset(itemgetter(*new_indices)(dataset))


class FixedLength(Transform):
    """
    Transforms the data to a fixed length.
    The traces will be cut at the end, shorter traces will either be filled
    with zeros or repeated from the beginning.
    """

    def __init__(self, length: int, start: int = 0, repeat: bool = False):
        """
        :param length: The target trace length.
        :param start: Start index. Everything up to this index will be skipped.
        :param repeat: Whether the trace should start from the beginning for
            shorter traces. Will be filled with zeros / no spikes otherwise.
        """
        self.length = length
        self.start = start
        self.repeat = repeat

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        samples = []
        for sample in dataset:
            new_sample = sample[self.start:self.start + self.length]
            len_diff = self.length - len(new_sample)
            if not self.repeat and len_diff > 0:
                samples.append(new_sample + sample.as_empty(len_diff))
                continue
            while len_diff > 0:
                new_sample += new_sample[:len_diff]
                len_diff = self.length - len(new_sample)
            samples.append(new_sample)

        return data.Dataset(samples)


class Standardize(Transform):
    """
    Standardize all channels of timeseries samples by subtracting the mean and
    dividing by the standard deviation of the respective channel in the whole
    dataset.
    """

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        avgs = defaultdict(list)
        lengths = defaultdict(list)
        variances = defaultdict(list)
        for sample in dataset:
            if not isinstance(sample, data.TimeseriesSample):
                raise TypeError(f"{sample} is not a TimeseriesSample")
            for channel, trace in sample.data.items():
                avgs[channel].append(np.mean(trace))
                lengths[channel].append(len(trace))
                variances[channel].append(np.var(trace))
        avg = {}
        std_dev = {}
        for channel in avgs:
            weights = np.divide(lengths[channel], np.sum(lengths[channel]))
            avg[channel] = np.multiply(avgs[channel], weights).sum()
            # Steiner's theorem:
            std_dev[channel] = np.sqrt(
                ((variances[channel] + (avgs[channel] - avg[channel])**2)
                 * weights).sum())

        samples = []
        for sample in dataset:
            samples.append(sample.replace(
                data={ch: (d - avg[ch]) / std_dev[ch]
                      for ch, d in sample.data.items()}))

        return data.Dataset(samples)


class Offset(Transform):
    """
    Adds an offset to every channel of timeseries data.
    """

    def __init__(self, offset: np.number):
        """
        :param offset: The offset to add to the data.
        """
        self.offset = offset

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        samples = []
        for sample in dataset:
            if not isinstance(sample, data.TimeseriesSample):
                raise TypeError(f"{sample} is not a TimeseriesSample")
            samples.append(sample.replace(
                data={ch: d + self.offset for ch, d in sample.data.items()}))

        return data.Dataset(samples)


class RolledVariants(Transform):
    """
    Rolls samples along the time axis for each multiple of the step size,
    elements that are shifted beyond the last position are re-introduced at the
    first.

    The returned dataset contains multiple variants of each sample, its size
    will grow by a factor of ``num_variants``.
    """

    def __init__(self, step_size: int, num_variants: Optional[int] = None):
        """
        :param step_size: The step size of the operation
        :param num_variants: The number of variants of each sample
            (defaults to ``len(sample) // step_size``)
        """
        if not step_size > 0:
            raise ValueError("step_size has to be positive")
        if num_variants is not None and not num_variants > 0:
            raise ValueError("num_variants has to be positive")
        self.step_size = step_size
        self.num_variants = num_variants

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        samples = []
        for sample in dataset:
            num_variants = self.num_variants or len(sample) // self.step_size
            samples.append(sample)
            for variant_id in range(1, num_variants):
                shift = variant_id * self.step_size
                samples.append(sample[-shift:] + sample[:-shift])
        return data.Dataset(samples)


class Split(Transform):
    """
    Splits each sample in multiple smaller ones with given length, discards a
    rest at the end.
    """

    def __init__(self, length, overlap: int = 0):
        """
        :param length: The target trace length.
        :param overlap: Overlap of two successive samples.
        """
        self.length = length
        self.overlap = overlap

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        samples = []
        for sample in dataset:
            for start in range(0, len(sample) - self.length + 1,
                               self.length - self.overlap):
                samples.append(sample[start:start + self.length])
        return data.Dataset(samples)


class ChangeDataType(Transform):
    """
    Cast the data to the given dtype.
    """

    def __init__(self, dtype: Type[np.number]):
        """
        :param dtype: The dtype to cast the data to.
        """
        self.dtype = dtype

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        return data.Dataset([
            s.replace(
                data={ch: d.astype(self.dtype) for ch, d in s.data.items()})
            for s in dataset])


class ChangeLabelType(Transform):
    """
    Cast the label to the given dtype.
    """

    def __init__(self, dtype: Type):
        """
        :param dtype: The dtype to cast the label to.
        """
        self.dtype = dtype

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        return data.Dataset(
            [s.replace(label=self.dtype(s.label)) for s in dataset])


class Pool(Transform):
    """
    Abstract pooling class to reduce the sample rate.
    """

    @abstractmethod
    def _pool(self, reshaped_data: np.ndarray):
        """
        The actual pooling.

        :param reshaped_data: The reshaped data of the sample.
            The length of the last axis has to be the same as ``self.width``.
        """
        raise NotImplementedError

    def __init__(self, width: int, *, stride: Optional[int] = None):
        """
        :param width: The width of the pooling operation.
        :param stride: The stride of the operation, defaults to ``width``.
        """
        if not width > 0:
            raise ValueError("Width has to be positive")
        if stride is None:
            stride = width
        if not stride > 0:
            raise ValueError("Stride has to be positive")
        self.width = width
        self.stride = stride

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        samples = []
        for sample in dataset:
            if not isinstance(sample, data.TimeseriesSample):
                raise TypeError(f"{sample} is not a TimeseriesSample")
            shape_view = ((len(sample) - self.width) // self.stride + 1,
                          self.width)
            new_data = {}
            for channel, trace in sample.data.items():
                stride_in = trace.strides[0]
                strides_view = (stride_in * self.stride, stride_in)
                trace_view = np.lib.stride_tricks.as_strided(
                    trace, shape=shape_view, strides=strides_view,
                    writeable=False)
                new_data[channel] = self._pool(trace_view)
            samples.append(sample.replace(
                data=new_data,
                sample_rate=sample.sample_rate / self.stride))
        return data.Dataset(samples)


class Convolve(Pool):
    """
    Convolve the last axis of the data with given coefficient array and stride.
    This corresponds to a FIR-filter.
    """
    def __init__(self, kernel: np.ndarray, stride: int = 1):
        """
        :param kernel: The coefficients of the filter.
        :param stride: The stride of the output.
        """
        super().__init__(len(kernel), stride=stride)
        self.kernel = kernel

    def _pool(self, reshaped_data: np.array):
        return np.dot(reshaped_data, self.kernel)


class MeanPool(Pool):
    """
    Take the mean of data points of a trace to reduce the sample rate and
    denoise the signal.
    """

    def _pool(self, reshaped_data: np.ndarray):
        return np.mean(reshaped_data, axis=-1)


class MaxPool(Pool):
    """
    Reduce the sample rate by max pooling.
    """

    def _pool(self, reshaped_data: np.ndarray):
        return np.amax(reshaped_data, axis=-1)


class MinPool(Pool):
    """
    Reduce the sample rate by min pooling.
    """

    def _pool(self, reshaped_data: np.ndarray):
        return np.amin(reshaped_data, axis=-1)


class MaxMinDiffPool(Pool):
    """
    Returns the difference of max and min pooling.
    """

    def _pool(self, reshaped_data: np.ndarray):
        d_max = np.amax(reshaped_data, -1)
        d_min = np.amin(reshaped_data, -1)
        return np.subtract(d_max, d_min, out=d_max)


class LambdaPool(Pool):
    """
    Apply a custom pool function to the data.
    """

    def _pool(self, reshaped_data: np.ndarray):
        return np.apply_along_axis(self.poolfunction, -1, reshaped_data)

    def __init__(self, width: int,
                 poolfunction: Callable[[np.ndarray], numbers.Integral], *,
                 stride: Optional[int] = None):
        """
        :param poolfunction: The function to pool the data.
        """
        super().__init__(width, stride=stride)
        self.poolfunction = poolfunction


class Difference(Transform):
    """
    Simple differentiation by subtracting previous datapoints.
    """

    def __init__(self, stride: int = 1):
        """
        :param stride: The stride of the diff operation.

        The resulting data will be equal to `data[stride:] - data[:-stride]`
        """
        self.stride = stride

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        samples = []
        for sample in dataset:
            if not isinstance(sample, data.TimeseriesSample):
                raise TypeError(f"{sample} is not a TimeseriesSample")
            samples.append(
                sample.replace(data={ch: d[self.stride:] - d[:-self.stride]
                                     for ch, d in sample.data.items()}))
        return data.Dataset(samples)


class Clip(Transform):
    """
    Clips the data to the given dynamic range. This allows to ignore outliers.
    Values outside the given interval are clipped to the interval edges.
    """
    def __init__(self, vmin: Optional[float] = None,
                 vmax: Optional[float] = None):
        """
        :param vmin: Minimum value of the clipping interval.
        :param vmax: Maximum value of the clipping interval.
        """
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        samples = []
        for sample in dataset:
            if not isinstance(sample, data.TimeseriesSample):
                raise TypeError(f"{sample} is not a TimeseriesSample")
            samples.append(
                sample.replace(data={ch: np.clip(d, self.vmin, self.vmax)
                                     for ch, d in sample.data.items()}))
        return data.Dataset(samples)


class ReLU(Clip):
    """
    Applies the rectified linear unit function element-wise.
    """
    def __init__(self):
        super().__init__(vmin=0)


class Pad(Transform):
    """
    Pad the data with zeros on both sides.
    """

    def __init__(self, pad_width: Union[int, Tuple[int, int]]):
        """
        :param pad_width: Number of values padded to the edges of the trace.
            An integer will pad symmetrically to both edges, a tuple allows to
            set the before and after pads individually.
        """
        self.pad_width = pad_width if isinstance(pad_width, tuple) \
            else (pad_width,) * 2

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        samples = []
        for sample in dataset:
            pad_begin = pad_end = sample.as_empty(self.pad_width[0])
            if self.pad_width[0] != self.pad_width[1]:
                pad_end = sample.as_empty(self.pad_width[1])
            samples.append(pad_begin + sample + pad_end)
        return data.Dataset(samples)


class CombineTimeseriesChannels(Transform):
    """
    Combines the data of all channels of a timeseries to a common one.
    """
    def __init__(self, combine_fn: Callable[[np.ndarray], np.number],
                 new_channel: data.Channel):
        """
        :param combine_fn: Function that takes an array of all channel data at
            one position in time and returns a combined value.
        :param new_channel: The channel of the combined data.
        """
        self.combine_fn = combine_fn
        self.new_channel = new_channel

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        samples = []
        for sample in dataset:
            if not isinstance(sample, data.TimeseriesSample):
                raise TypeError(f"{sample} is not a TimeseriesSample")
            new_data = np.stack(list(sample.data.values()), axis=0)
            new_data = np.apply_along_axis(self.combine_fn, 0, new_data)
            samples.append(sample.replace(data={self.new_channel: new_data}))
        return data.Dataset(samples)


class Resample(Transform):
    """
    Resample every sample with :py:func:`scipy.signal.resample`.
    Because a Fourier method is used, the trace is assumed to be periodic.
    """

    def __init__(self, sample_rate: float):
        """
        :param sample_rate: The target sample rate.
        """
        self.sample_rate = sample_rate

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        samples = []
        for sample in dataset:
            samples.append(sample.resampled(self.sample_rate))
        return data.Dataset(samples)


class BaselineCorrection(Transform):
    """
    Correct the given ecg data by subtracting the baseline as proposed by
    Liu, Wang, Liu, "ECG Signal Denoising Based on Morphological Filtering"
    https://ieeexplore.ieee.org/document/5780239
    """

    def __call__(self, dataset: data.Dataset) -> data.Dataset:

        def _oc_co(trace: np.ndarray, size: int, tmparray: np.ndarray):
            """
            Filter the signal by applying an opening/closing-closing/opening
            filter with given width.
            """
            grey_closing(trace, size=size, output=tmparray)
            grey_opening(tmparray, size=size, output=tmparray)
            grey_opening(trace, size=size, output=trace)
            grey_closing(trace, size=size, output=trace)
            trace += tmparray
            trace /= 2

        def _correct_trace(trace: np.ndarray, sample_rate):
            """
            Correct the trace by subtracting the baseline.
            """
            baseline = np.copy(trace)
            tmparray = np.empty_like(trace)
            qrs_width = 0.11 * sample_rate
            p_t_width = 0.27 * sample_rate
            for size in [qrs_width, p_t_width]:
                _oc_co(baseline, int(size), tmparray)

            return trace - baseline

        samples = []
        for sample in dataset:
            if not isinstance(sample, data.TimeseriesSample):
                raise TypeError(f"{sample} is not a TimeseriesSample")
            samples.append(sample.replace(
                data={ch: _correct_trace(d, sample.sample_rate)
                      for ch, d in sample.data.items()}))
        return data.Dataset(samples)


class DetectPeaks(Transform):
    """
    Detect peaks of the data with `find_peaks` from scipy.
    """

    def __init__(self, prominence: float = 0.180,
                 distance: Optional[float] = 0.4):
        """
        :param prominence: The minimum peak prominence.
        :param distance: The minimum distance between two peaks in seconds.
        """
        self.prominence = prominence
        self.distance = distance

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        samples = []
        for sample in dataset:
            if not isinstance(sample, data.TimeseriesSample):
                raise TypeError(f"{sample} is not a TimeseriesSample")

            new_data = {}
            distance_abs = None if self.distance is None \
                else self.distance * sample.sample_rate
            for channel in sample.channels:
                new_data[channel] = np.array(find_peaks(
                    sample.data[channel],
                    prominence=self.prominence,
                    distance=distance_abs,
                )[0])

            samples.append(data.SpikingSample.empty_like(sample).replace(
                data=new_data))

        return data.Dataset(samples)


class SimpleThresholdPeaks(Transform):
    """
    Very simple transform that uses a fixed threshold to detect peaks.
    """

    def __init__(self, threshold: float, distance: float = 0.4):
        """
        :param threshold: Minimum value to be considered as a spike.
        :param distance: Minimum distance between two peaks in seconds.
        """
        self.threshold = threshold
        self.distance = distance

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        samples = []
        for sample in dataset:
            if not isinstance(sample, data.TimeseriesSample):
                raise TypeError(f"{sample} is not a TimeseriesSample")

            new_data = {}
            for channel in sample.channels:
                peak_mask = sample.data[channel] >= self.threshold
                # detect only rising edges
                peaks = np.nonzero(np.diff(peak_mask.astype(int)) == 1)[0] + 1
                if self.distance > 0:
                    deadtime = int(self.distance * sample.sample_rate)
                    count = 1
                    for time in peaks[1:]:
                        if (time - peaks[count - 1]) >= deadtime:
                            peaks[count] = time
                            count += 1
                    peaks = peaks[:count]
                new_data[channel] = peaks

            samples.append(data.SpikingSample.empty_like(sample).replace(
                data=new_data))

        return data.Dataset(samples)
