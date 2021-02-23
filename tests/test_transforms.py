"""
Tests for the transforms module.
"""
from abc import ABCMeta, abstractmethod
import copy
import dataclasses
import functools
from typing import ClassVar, List, Optional, Tuple
import unittest
import numpy as np
from timedatasets import data, transforms, helpers


TransformClassType = type(transforms.Transform)


class Arguments:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        str_args = [str(a) for a in self.args]
        str_args += [f"{key}={value}" for key, value in self.kwargs.items()]
        return f"{self.__class__.__name__}({', '.join(str_args)})"


class TestLabel(data.Label):
    One = 1
    Two = 2
    Three = 3


class TestChannel(data.Channel):
    One = 1
    Two = 2


class NoArgumentsMixin:
    """
    Sets reasonable defaults in case the transformation has no arguments.
    """
    valid_arguments: ClassVar[List[Arguments]] = [Arguments()]
    invalid_arguments: ClassVar[List[Tuple[Arguments, Exception]]] = []


class TransformTestBase(unittest.TestCase, metaclass=ABCMeta):
    """
    Test case that provides a base for tests of transforms on datasets.

    :cvar dataset: Dataset to test the transformation
    :cvar timeseries_dataset: Dataset that contains timeseries samples
    :cvar spiking_dataset: Dataset that contains spiking samples
    """
    timeseries_dataset: ClassVar[data.Dataset] = data.Dataset([
        data.TimeseriesSample(
            {TestChannel.One:
             np.array([-30, -46, -38, -25, -11, 22, 2, 18, 19, 52,
                       100, 95, 62, 117, 815, 3, -33, 3, 41, 108,
                       164, 193, 136, 91, 98, 102, 71, 116, 129, 128,
                       117, 98, 98, 158, 107, 59, 117, 845, -10, -68,
                       -22, -14, 25, 103, 107, 85, -20, -36, -10, -90,
                       -47, -17, -54, -33, -29, -33, -2, -32, -81, -3,
                       714, -140, -213, -188, -159, -101, -22, -4, -66, -130,
                       -123, -121, -136, -103, -87, -78, -104, -88, -77, 10,
                       -85, -122, 101, 481, -171, -180, -143, -120, -36, 33,
                       37, -41, -73, -63, -55, -45, -14, 1, 4, 16]) * 0.001},
            TestLabel.One,
            filename="file1", sample_rate=30.,
        ),
        data.TimeseriesSample(
            {TestChannel.One:
             np.array([-85, -122, 101, 481, -171,
                       -180, -143, -120, -36, 33]) * 0.001,
             TestChannel.Two:
             np.array([37, -41, -73, -63, -55, -45, -14, 1, 4, 16]) * 0.001},
            TestLabel.Two,
            filename="file2", sample_rate=30.,
        )
    ])
    spiking_dataset: ClassVar[data.Dataset] = data.Dataset([
        data.SpikingSample(
            {TestChannel.One: np.array([14, 37, 60, 83])},
            TestLabel.One,
            filename="file1", sample_rate=30.,
            length=100),
        data.SpikingSample(
            {TestChannel.One: np.array([3]),
             TestChannel.Two: np.array([], dtype=int)},
            TestLabel.Two,
            filename="file2", sample_rate=30.,
            length=10),
    ])
    dataset: ClassVar[data.Dataset] = timeseries_dataset + spiking_dataset

    @property
    @abstractmethod
    def transform_class(self) -> TransformClassType:
        """ The class of the transformation to test """
        raise NotImplementedError

    @property
    @abstractmethod
    def valid_arguments(self) -> List[Arguments]:
        """ Valid arguments for the transformation """
        raise NotImplementedError

    @property
    @abstractmethod
    def invalid_arguments(self) -> List[Tuple[Arguments, Exception]]:
        """ Invalid arguments and resulting exception """
        raise NotImplementedError

    @functools.cached_property
    def transformed(self) -> List[data.Dataset]:
        """
        List of transformed datasets for every set of valid arguments
        """
        original_dataset = copy.deepcopy(self.dataset)
        transformed = []
        for arguments in self.valid_arguments:
            with self.subTest(arguments):
                transformed.append(
                    self.transform_class(
                        *arguments.args, **arguments.kwargs)(self.dataset))
                self.assertDatasetEqual(
                    self.dataset, original_dataset,
                    "Transform alters the original dataset.")
        return transformed

    def test_valid_arguments(self):
        for arguments in self.valid_arguments:
            with self.subTest(arguments):
                transform = self.transform_class(
                    *arguments.args, **arguments.kwargs)
                self.assertIsInstance(transform, self.transform_class)

    def test_invalid_arguments(self):
        for arguments, exception in self.invalid_arguments:
            with self.subTest(arguments, exception=exception):
                with self.assertRaises(exception):
                    self.transform_class(
                        *arguments.args, **arguments.kwargs)(self.dataset)

    def assertSampleEqual(self, sample1: data.TimeSample,  # pylint: disable=invalid-name
                          sample2: data.TimeSample,
                          msg: Optional[str] = None):
        """
        A sample specific equality assertion.
        """
        def compare_elements(element1, element2, msg, field_name):
            if not helpers.array_safe_equal(element1, element2):
                msg += f"{field_name} differs:\n" \
                    f"{element1!r}\n!=\n{element2!r}"
                if isinstance(element1, np.ndarray) \
                        and isinstance(element2, np.ndarray):
                    try:
                        msg += f"\ndiff:\n" \
                            f"{np.around(element1 - element2, 3)!r}"
                    except ValueError:
                        pass
                self.fail(msg)

        if sample1 == sample2:
            return
        msg = f"{msg}\n" if msg else ""
        if sample1.__class__ is not sample2.__class__:
            self.fail(f"{msg}Class differs: "
                      f"{sample1.__class__} != {sample2.__class__}")
        for field, element1, element2 in zip(dataclasses.fields(sample1),
                                             dataclasses.astuple(sample1),
                                             dataclasses.astuple(sample2)):
            if field.name == "data":
                for channel, data1, data2 in zip(element1,
                                                 element1.values(),
                                                 element2.values()):
                    compare_elements(
                        data1, data2, msg, f"data.{str(channel)}")
            else:
                compare_elements(element1, element2, msg, field.name)

    def assertDatasetEqual(self, dataset1: data.Dataset,  # pylint: disable=invalid-name
                           dataset2: data.Dataset,
                           msg: Optional[str] = None):
        """
        A dataset specific equality assertion.
        """
        if dataset1 == dataset2:
            return
        for sample1, sample2 in zip(dataset1, dataset2):
            self.assertSampleEqual(sample1, sample2, msg=msg)


class TimeseriesTransformTestBase(TransformTestBase, metaclass=ABCMeta):
    """
    Test case that provides a base for tests of transformations on
    timeseries datasets.

    :cvar dataset: Dataset to test the transformation.
    """
    dataset: ClassVar[data.Dataset] = TransformTestBase.timeseries_dataset

    def test_spiking_data(self):
        for arguments in self.valid_arguments:
            with self.subTest(arguments):
                transform = self.transform_class(
                    *arguments.args, **arguments.kwargs)
                with self.assertRaises(TypeError):
                    transform(self.spiking_dataset)


class SpikingTransformTestBase(TransformTestBase, metaclass=ABCMeta):
    """
    Test case that provides a base for tests of transformations on
    spiking datasets.

    :cvar dataset: Dataset to test the transformation.
    """
    dataset: ClassVar[data.Dataset] = TransformTestBase.spiking_dataset

    def test_timeseries_data(self):
        for arguments in self.valid_arguments:
            with self.subTest(arguments):
                transform = self.transform_class(
                    *arguments.args, **arguments.kwargs)
                with self.assertRaises(TypeError):
                    transform(self.timeseries_dataset)


class TestBaseTransform(TransformTestBase):
    """
    Test the abstract base class.
    """

    class Transform(transforms.Transform):
        def __init__(self, label: data.Label):
            self.label = label

        def __call__(self, dataset: data.Dataset) -> data.Dataset:
            return data.Dataset([s.replace(label=self.label) for s in dataset])

    transform_class = Transform
    valid_arguments = [Arguments(label=TestLabel.Three)]
    invalid_arguments = [(Arguments(something=None), TypeError)]

    def test_representation(self):
        transform = self.transform_class(**self.valid_arguments[0].kwargs)
        self.assertEqual(
            repr(transform),
            f"Transform(label={repr(TestLabel.Three)})"
        )

    def test_equality(self):
        class AnotherTransform(self.transform_class):
            pass

        transform = self.transform_class(**self.valid_arguments[0].kwargs)
        self.assertEqual(transform, transform)
        self.assertEqual(transform, self.transform_class(TestLabel.Three))
        self.assertNotEqual(transform, AnotherTransform(TestLabel.Three))
        self.assertNotEqual(transform, self.transform_class(TestLabel.One))


class TestBalanceClasses(TransformTestBase):
    transform_class = transforms.BalanceClasses
    dataset = data.Dataset(list(TimeseriesTransformTestBase.dataset)
                           + [TimeseriesTransformTestBase.dataset[1]] * 2)
    valid_arguments = [
        Arguments(seed=12345, resample_factor=-1),
        Arguments(seed=12345, resample_factor=.35),
        Arguments(seed=12345, resample_factor=1.),
    ]
    invalid_arguments = [
        (Arguments(seed=12345, resample_factor=2), ValueError),
    ]

    def test_transformed(self):
        self.assertEqual(len(self.transformed[0]), 2)
        self.assertEqual(len(self.transformed[1]), 4)
        self.assertEqual(len(self.transformed[2]), 6)
        self.assertEqual(len([s for s in self.transformed[2]
                              if s.label == self.dataset[0].label]), 3)  # pylint: disable=no-member


class TestFixedLength(TransformTestBase):
    transform_class = transforms.FixedLength
    valid_arguments = [
        Arguments(length=13, start=3, repeat=True),
        Arguments(length=13, start=3, repeat=False),
    ]
    invalid_arguments = [
        (Arguments(length=13, start=-3, repeat=False), ValueError),
        (Arguments(length=-13, start=3, repeat=False), ValueError),
    ]

    def test_transformed(self):
        self.assertSampleEqual(
            self.transformed[0][0],
            self.dataset[0][3:16])  # pylint: disable=unsubscriptable-object
        self.assertListEqual(
            self.transformed[0][1].data[TestChannel.Two].tolist(),
            [-.063, -.055, -.045, -.014, .001, .004, .016, -.063, -.055, -.045,
             -.014, .001, .004])

        self.assertSampleEqual(self.transformed[0][0], self.transformed[1][0])
        self.assertListEqual(
            self.transformed[1][1].data[TestChannel.Two].tolist(),
            [-.063, -.055, -.045, -.014, .001, .004, .016, 0, 0, 0, 0, 0, 0])
        self.assertEqual(len(self.transformed[1][1].channels), 2)
        self.assertListEqual(
            self.transformed[1][2].data[TestChannel.One].tolist(), [11])
        self.assertListEqual(
            self.transformed[1][3].data[TestChannel.One].tolist(), [0])


class TestRolledVariants(TransformTestBase):
    transform_class = transforms.RolledVariants
    valid_arguments = [
        Arguments(step_size=20, num_variants=10),
    ]
    invalid_arguments = [
        (Arguments(step_size=0), ValueError),
        (Arguments(step_size=10, num_variants=-8), ValueError),
    ]

    def test_transformed(self):
        transformed = self.transformed[0]
        self.assertEqual(len(transformed), 40)
        self.assertListEqual(
            transformed[1].data[TestChannel.One].tolist(),
            transformed[0].data[TestChannel.One][-20:].tolist()
            + transformed[0].data[TestChannel.One][:-20].tolist())


class TestSplit(TransformTestBase):
    transform_class = transforms.Split
    valid_arguments = [
        Arguments(length=20, overlap=4),
    ]
    invalid_arguments = []

    def test_transformed(self):
        transformed = self.transformed[0]
        self.assertEqual(len(transformed), 12)
        self.assertDatasetEqual(transformed[0],
                                transforms.FixedLength(20)(self.dataset)[0])
        self.assertSampleEqual(transformed[0][-4:], transformed[1][:4])
        self.assertDatasetEqual(transformed[6],
                                transforms.FixedLength(20)(self.dataset)[2])


class TestStandardize(NoArgumentsMixin, TimeseriesTransformTestBase):
    transform_class = transforms.Standardize

    def test_transformed(self):
        concatenated_channel_data = np.concatenate(
            tuple(s.data[TestChannel.One] for s in self.dataset))
        self.assertEqual(concatenated_channel_data.shape, (110,))
        self.assertSampleEqual(
            self.transformed[0][0],
            self.dataset[0].replace(
                data={TestChannel.One:
                      (self.dataset[0].data[TestChannel.One]
                       - concatenated_channel_data.mean())
                      / concatenated_channel_data.std()}))


class TestOffset(TimeseriesTransformTestBase):
    transform_class = transforms.Offset
    valid_arguments = [
        Arguments(offset=12.),
        Arguments(offset=-5),
    ]
    invalid_arguments = []

    def test_transformed(self):
        for channel in self.dataset[1].channels:
            self.assertTrue(helpers.array_safe_equal(
                self.transformed[0][1].data[channel],
                self.dataset[1].data[channel] + 12.))
            self.assertTrue(helpers.array_safe_equal(
                self.transformed[1][1].data[channel],
                self.dataset[1].data[channel] - 5.))


class TestIdentity(NoArgumentsMixin, TransformTestBase):
    transform_class = transforms.Identity

    def test_transformed(self):
        self.assertIs(self.transformed[0], self.dataset)
        id_uuid = self.transform_class().uuid
        self.assertEqual(id_uuid,
                         transforms.Transform.concat_uuids([id_uuid, id_uuid]))


class TestChangeDataType(TransformTestBase):
    transform_class = transforms.ChangeDataType
    valid_arguments = [
        Arguments(dtype=np.float64),
        Arguments(dtype=float),
        Arguments(dtype=bool),
    ]
    invalid_arguments = []

    def test_transformed(self):
        transformed = self.transformed[0]
        orig_dtype = self.dataset[0].data[TestChannel.One].dtype  # pylint: disable=no-member
        self.assertEqual(
            transformed[0].data[TestChannel.One].dtype, np.float64)
        self.assertDatasetEqual(
            transforms.ChangeDataType(dtype=orig_dtype)(transformed)[:2],
            self.dataset[:2])


class TestChangeLabelType(TransformTestBase):
    transform_class = transforms.ChangeLabelType
    valid_arguments = [
        Arguments(dtype=int),
    ]
    invalid_arguments = []

    def test_transformed(self):
        self.assertIs(type(self.transformed[0][0].label), int)
        orig_dtype = type(self.dataset[0].label)  # pylint: disable=no-member
        self.assertDatasetEqual(
            transforms.ChangeLabelType(dtype=orig_dtype)(self.transformed[0]),
            self.dataset)


class TestLambdaPool(TimeseriesTransformTestBase):
    transform_class = transforms.LambdaPool
    valid_arguments = [
        Arguments(width=5, poolfunction=lambda trace: trace[0]),
        Arguments(width=8, poolfunction=lambda trace: trace[0], stride=5),
    ]
    invalid_arguments = [
        (Arguments(0, poolfunction=lambda t: t[0]), ValueError),
    ]

    def test_transformed(self):
        self.assertListEqual(
            self.transformed[0][0].data[TestChannel.One].tolist(),
            self.dataset[0].data[TestChannel.One][..., ::5].tolist())
        self.assertEqual(self.transformed[0][0].sample_rate, 6.)
        self.assertSampleEqual(self.transformed[0][0][:-1],
                               self.transformed[1][0])


class TestMeanPool(TimeseriesTransformTestBase):
    transform_class = transforms.MeanPool
    valid_arguments = [
        Arguments(width=2),
    ]
    invalid_arguments = [
        (Arguments(width=0), ValueError),
    ]

    def test_transformed(self):
        transform2 = transforms.LambdaPool(width=2, poolfunction=np.mean)
        self.assertEqual(self.transformed[0], transform2(self.dataset),
                         "Transformed dataset does not look as expected.")


class TestMaxPool(TimeseriesTransformTestBase):
    transform_class = transforms.MaxPool
    valid_arguments = [
        Arguments(width=2),
    ]
    invalid_arguments = [
        (Arguments(width=-1), ValueError),
    ]

    def test_transformed(self):
        transform2 = transforms.LambdaPool(width=2, poolfunction=max)
        self.assertEqual(self.transformed[0], transform2(self.dataset),
                         "Transformed dataset does not look as expected.")


class TestMinPool(TimeseriesTransformTestBase):
    transform_class = transforms.MinPool
    valid_arguments = [
        Arguments(width=2),
    ]
    invalid_arguments = [
        (Arguments(width=-1), ValueError),
    ]

    def test_transformed(self):
        transform2 = transforms.LambdaPool(width=2, poolfunction=min)
        self.assertDatasetEqual(self.transformed[0], transform2(self.dataset))


class TestMaxMinDiffPool(TimeseriesTransformTestBase):
    transform_class = transforms.MaxMinDiffPool
    valid_arguments = [
        Arguments(width=5),
    ]
    invalid_arguments = [
        (Arguments(width=-1), ValueError),
    ]

    def test_transformed(self):
        transform2 = transforms.LambdaPool(
            width=5, poolfunction=lambda x: max(x) - min(x))
        self.assertDatasetEqual(self.transformed[0], transform2(self.dataset))


class TestDifference(TimeseriesTransformTestBase):
    transform_class = transforms.Difference
    dataset_diff = data.Dataset([
        data.TimeseriesSample(
            {TestChannel.One:
             np.array([52, 48, 56, 44, 63, 78, 93, 44, 98, 763, -97, -128,
                       -59, -76, -707, 161, 226, 133, 50, -10, -62, -122,
                       -20, 38, 30, 15, 27, -18, 29, -21, -58, 19, 747,
                       -168, -175, -81, -131, -820, 113, 175, 107, -6,
                       -61, -113, -197, -132, 3, -18, -23, 61, 14, 15, 22,
                       -48, 26, 747, -138, -181, -107, -156, -815, 118,
                       209, 122, 29, -22, -99, -132, -37, 43, 45, 17, 48,
                       26, 97, -7, -18, 189, 558, -181, -95, -21, -221,
                       -517, 204, 217, 102, 47, -27, -88, -82, 27, 74, 67,
                       71]) * 0.001},
            TestLabel.One,
            filename="file1", sample_rate=30.,
        ),
        data.TimeseriesSample(
            {TestChannel.One:
             np.array([-95, -21, -221, -517, 204]) * 0.001,
             TestChannel.Two:
             np.array([-82, 27, 74, 67, 71]) * 0.001},
            TestLabel.Two,
            filename="file2", sample_rate=30.,
        ),
    ])
    valid_arguments = [
        Arguments(stride=5),
    ]
    invalid_arguments = []

    def test_transformed(self):
        self.assertDatasetEqual(
            self.transformed[0], self.dataset_diff,
            "Transformed dataset does not look as expected.")


class TestConvolve(TimeseriesTransformTestBase):
    transform_class = transforms.Convolve
    valid_arguments = [
        Arguments(kernel=np.array([-1] * 2 + [0] * 4 + [1] * 2), stride=2),
    ]
    invalid_arguments = [
        (Arguments(kernel=np.ones(20)), ValueError),
        (Arguments(kernel=np.ones(2), stride=0), ValueError),
    ]

    def test_transformed(self):
        transform2 = transforms.Compose(
            transforms.LambdaPool(width=2, poolfunction=np.sum),
            transforms.Difference(stride=3)
        )
        self.assertDatasetEqual(
            self.transformed[0], transform2(self.dataset),
            "Transformed dataset does not look as expected.")


class TestBaselineCorrection(NoArgumentsMixin, TimeseriesTransformTestBase):
    transform_class = transforms.BaselineCorrection
    dataset_corrected = data.Dataset([
        data.TimeseriesSample(
            {TestChannel.One: np.array([
                -19.00, -35.00, -27.00, -14.00, 0.00, 11.50, -8.50, -0.50,
                0.25, 16.75, 50.25, 45.25, 12.25, 67.25, 765.25, -46.75,
                -82.75, -46.75, -20.00, 5.75, 57.25, 86.25, 29.25, -15.75,
                -8.75, -4.75, -35.75, 3.75, 16.75, 15.75, 4.75, -9.50,
                -9.50, 50.50, -0.50, -29.00, 29.00, 757.00, -47.50,
                -103.50, -57.50, -49.50, -10.50, 67.50, 71.50, 49.50,
                -5.00, -13.00, 13.00, -60.75, -17.75, 12.25, -21.75, -0.75,
                3.25, -0.75, 30.25, 0.25, -39.00, 39.00, 756.00, -44.25,
                -117.25, -92.25, -63.25, -5.25, 73.75, 91.75, 29.75,
                -22.00, -15.00, -13.00, -28.00, -6.25, 5.75, 12.50, -13.50,
                -1.25, 8.75, 74.00, -21.00, -58.00, 165.00, 545.00,
                -107.00, -116.00, -79.00, -56.00, 9.50, 78.50, 82.50, 4.50,
                -27.50, -17.50, -9.50, -4.50, 26.50, 41.50, 44.50, 56.50
            ]) * 0.001},
            TestLabel.One,
            filename="file1", sample_rate=30.,
        ),
        data.TimeseriesSample(
            {TestChannel.One:
             np.array([18.5, -18.5, 204.5, 584.5, -47.75,
                       -56.75, -19.75, 3.25, 87.25, 156.25]) * 0.001,
             TestChannel.Two:
             np.array([85, 7, -25, -15, -7, -2, 29, 44, 47, 59]) * 0.001},
            TestLabel.Two,
            filename="file2", sample_rate=30.,
        )])

    def test_transformed(self):
        self.assertDatasetEqual(
            self.transformed[0], self.dataset_corrected,
            "Transformed dataset does not look as expected.")


class TestCompose(TransformTestBase):
    transform_class = transforms.Compose

    class CutAtStart(transforms.Transform):
        def __init__(self, start: int):
            self.start = start

        def __call__(self, dataset: data.Dataset) -> data.Dataset:
            return dataset[self.start:]

    valid_arguments = [
        Arguments(CutAtStart(1), CutAtStart(2)),
        Arguments(CutAtStart(3)),
    ]
    invalid_arguments = []

    def test_transformed(self):
        self.assertDatasetEqual(
            self.transformed[0], self.transformed[1],
            "Transformed dataset does not look as expected.")

    def test_slicing(self):
        transform = self.transform_class(*self.valid_arguments[0].args)
        self.assertTupleEqual(tuple(transform), transform.transforms)
        self.assertTupleEqual(transform[:-1].transforms, (transform[0],))

    def test_uuid(self):
        transform = self.transform_class(*self.valid_arguments[0].args)
        self.assertEqual(transform.uuid,
                         transforms.Transform.concat_uuids([
                             transform[0].uuid, transform[1].uuid]))
        self.assertEqual(transform[:1].uuid, transform[0].uuid)
        self.assertEqual(transform[:0].uuid, transforms.Identity().uuid)


class TestClip(TimeseriesTransformTestBase):
    transform_class = transforms.Clip
    valid_arguments = [
        Arguments(vmin=-.087, vmax=.103),
    ]
    invalid_arguments = []

    def test_transformed(self):
        transformed = self.transformed[0]
        self.assertEqual(transformed[0].data[TestChannel.One].min(), -0.087)
        self.assertEqual(transformed[0].data[TestChannel.One].max(), 0.103)


class TestReLU(NoArgumentsMixin, TimeseriesTransformTestBase):
    transform_class = transforms.ReLU

    def test_transformed(self):
        self.assertGreaterEqual(
            self.transformed[0][0].data[TestChannel.One].min(), 0)
        self.assertListEqual(
            self.transformed[0][0].data[TestChannel.One][6:16].tolist(),
            self.dataset[0].data[TestChannel.One][6:16].tolist())


class TestSpikesToTimeseries(NoArgumentsMixin, SpikingTransformTestBase):
    transform_class = transforms.SpikesToTimeseries

    def test_transformed(self):
        transformed_sample = self.transformed[0][0]
        self.assertIsInstance(transformed_sample, data.TimeseriesSample)
        self.assertTrue(transformed_sample.data[TestChannel.One][83])
        self.assertFalse(transformed_sample.data[TestChannel.One][84])
        self.assertListEqual(transformed_sample.channels,
                             [TestChannel.One])


class TestPad(TransformTestBase):
    transform_class = transforms.Pad
    valid_arguments = [
        Arguments(pad_width=2),
    ]
    invalid_arguments = [
        (Arguments(pad_width=-3), ValueError),
    ]

    def test_transformed(self):
        transformed_sample = self.transformed[0][0]
        self.assertListEqual(
            transformed_sample.data[TestChannel.One][:3].tolist(),
            [.0, .0, self.dataset[0].data[TestChannel.One][0]])  # pylint: disable=no-member
        self.assertSequenceEqual(
            transformed_sample.data[TestChannel.One].shape, (104,))


class TestCombineTimeseriesChannels(TimeseriesTransformTestBase):
    transform_class = transforms.CombineTimeseriesChannels
    valid_arguments = [
        Arguments(combine_fn=np.max, new_channel=TestChannel.One),
    ]
    invalid_arguments = []

    def test_transformed(self):
        self.assertSampleEqual(self.transformed[0][0],
                               self.timeseries_dataset[0])
        self.assertSampleEqual(
            self.transformed[0][1],
            self.timeseries_dataset[1].replace(  # pylint: disable=no-member
                data={TestChannel.One: np.array(
                    [37, -41, 101, 481, -55, -45, -14, 1, 4, 33]) * 0.001}))


class TestResample(TransformTestBase):
    transform_class = transforms.Resample
    valid_arguments = [
        Arguments(sample_rate=20),
        Arguments(sample_rate=57),
    ]
    invalid_arguments = [
        (Arguments(sample_rate=-50), ValueError),
    ]

    def test_transformed(self):
        self.assertEqual(len(self.transformed[0][0]), 67)
        self.assertEqual(self.transformed[0][0].sample_rate, 20)
        self.assertEqual(self.transformed[0][3].data[TestChannel.One][0], 2)
        self.assertEqual(self.transformed[0][3].sample_rate, 20)
        self.assertEqual(len(self.transformed[1][0]), 190)
        self.assertEqual(self.transformed[1][0].sample_rate, 57)
        self.assertEqual(self.transformed[1][3].data[TestChannel.One][0], 5)
        self.assertEqual(self.transformed[1][3].sample_rate, 57)


class TestDetectPeaks(TimeseriesTransformTestBase):
    transform_class = transforms.DetectPeaks
    valid_arguments = [
        Arguments(prominence=0.180, distance=0.4),
        Arguments(prominence=1., distance=None),
    ]
    invalid_arguments = [
        (Arguments(prominence=1., distance=-0.3), ValueError),
    ]

    def test_transformed(self):
        self.assertDatasetEqual(
            self.transformed[0], self.spiking_dataset,
            "Transformed dataset does not look as expected")


class TestSimpleThresholdPeaks(TimeseriesTransformTestBase):
    transform_class = transforms.SimpleThresholdPeaks
    valid_arguments = [
        Arguments(threshold=0.4),
        Arguments(threshold=0.4, distance=1.),
    ]
    invalid_arguments = []

    def test_transformed(self):
        self.assertDatasetEqual(
            self.transformed[0], self.spiking_dataset,
            "Transformed dataset does not look as expected")
        self.assertListEqual(
            self.transformed[1][0].data[TestChannel.One].tolist(), [14, 60],
            "Transformed dataset does not look as expected.")


del TransformTestBase
del TimeseriesTransformTestBase
del SpikingTransformTestBase


if __name__ == "__main__":
    unittest.main()
