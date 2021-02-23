"""
Tests of the data module.
"""
from enum import auto
from operator import itemgetter
from pathlib import Path
import tempfile
import unittest
import numpy as np
import matplotlib.pyplot as plt
from timedatasets import data, helpers, transforms


class SimpleLabel(data.Label):
    Nay = -1
    Okayish = +1
    Yay = +2


class SimpleChannel(data.Channel):
    Test1 = auto()
    Test2 = auto()


class TestTimeseriesSample(unittest.TestCase):
    """
    Tests the timeseries sample class.
    """
    samples = [
        data.TimeseriesSample(
            data={
                SimpleChannel.Test1: np.arange(3),
                SimpleChannel.Test2: np.arange(3, 6),
            },
            sample_rate=1.,
            label=SimpleLabel.Yay,
            filename="things1.txt"),
        data.TimeseriesSample(
            data={
                SimpleChannel.Test1: np.arange(10),
            },
            sample_rate=1.,
            label=SimpleLabel.Nay,
            filename="things2.txt"),
    ]

    def test_equal(self):
        """ Test the equal & not equal functionality """
        self.assertNotEqual(self.samples[0], self.samples[1])
        self.assertEqual(self.samples[0], self.samples[0])
        self.assertNotEqual(self.samples[0], 25.)

    def test_repr(self):
        """ Test the representation """
        self.assertEqual(
            "TimeseriesSample(\n"
            "    length=3,\n"
            "    data={\n"
            "        SimpleChannel.Test1: array([0, 1, 2]),\n"
            "        SimpleChannel.Test2: array([3, 4, 5])\n"
            "    },\n"
            "    label=<SimpleLabel.Yay: 2>,\n"
            "    sample_rate=1.0,\n"
            "    filename='things1.txt'\n"
            ")",
            repr(self.samples[0])
        )

    def test_channels(self):
        """ Test the channels of the sample """
        self.assertListEqual(self.samples[0].channels,
                             [SimpleChannel.Test1, SimpleChannel.Test2])

    def test_dtype(self):
        """ Test the dtype of the sample """
        self.assertEqual(self.samples[0].dtype, np.dtype(int))

    def test_equal_without_data(self):
        """ Test the equality of samples excluding the actual data """
        self.assertTrue(self.samples[1].equal_without_data(
            self.samples[1].replace(data={SimpleChannel.Test1: np.arange(2)})))
        self.assertFalse(
            self.samples[0].equal_without_data(self.samples[0].replace(
                data={SimpleChannel.Test2: np.array([1])})))

    def test_as_empty(self):
        """ Test the creation of empty samples from existing ones """
        empty_sample = self.samples[0].as_empty()
        self.assertTrue(self.samples[0].equal_without_data(empty_sample))

    def test_replace(self):
        """ Test the replacing of sample properties """
        sample_copy = self.samples[0].replace()
        self.assertEqual(self.samples[0], sample_copy,
                         "Copy ist not equal to original sample.")
        sample_replaced = self.samples[0].replace(label=SimpleLabel.Okayish)
        self.assertNotEqual(self.samples[0], sample_replaced,
                            "Modifying the copy alters the original sample.")
        self.assertEqual(sample_replaced.label, SimpleLabel.Okayish,
                         "Replaced label looks not as expected.")
        # not existing field
        with self.assertRaises(TypeError):
            self.samples[0].replace(foo='bar')
        # different channel dtypes
        with self.assertRaises(TypeError):
            self.samples[0].replace(
                data={SimpleChannel.Test1: np.arange(2),
                      SimpleChannel.Test2: np.linspace(1, 2, 2)})
        # different channel lengths
        with self.assertRaises(ValueError):
            self.samples[0].replace(data={SimpleChannel.Test1: np.arange(2),
                                          SimpleChannel.Test2: np.arange(5)})

    def test_length(self):
        """ Test the length of a sample """
        self.assertEqual(len(self.samples[0]), 3)

    def test_concat(self):
        """ Test the concatenation of two timeseries samples """
        doublesample = self.samples[0] + self.samples[0]
        self.assertListEqual(doublesample.data[SimpleChannel.Test1].tolist(),
                             [0, 1, 2, 0, 1, 2])
        self.assertListEqual(doublesample.data[SimpleChannel.Test2].tolist(),
                             [3, 4, 5, 3, 4, 5])
        # label, filename and channels do not match
        with self.assertRaises(ValueError):
            _ = self.samples[0] + self.samples[1]

    def test_slice(self):
        """ Test the slicing of timeseries samples """
        self.assertListEqual(
            self.samples[0][1:4].data[SimpleChannel.Test1].tolist(), [1, 2])
        self.assertEqual(self.samples[0][::2].sample_rate, 0.5)
        with self.assertRaises(TypeError):
            _ = self.samples[0][2]

    def test_empty_like(self):
        """ Test TimeseriesSample.empty_like """
        self.assertTrue(self.samples[0].equal_without_data(
            data.TimeseriesSample.empty_like(self.samples[0])))
        self.assertTrue(self.samples[1].equal_without_data(
            data.TimeseriesSample.empty_like(TestSpikingSample.samples[1])))
        self.assertListEqual(
            data.TimeseriesSample.empty_like(
                self.samples[1], length=25).data[SimpleChannel.Test1].tolist(),
            [0] * 25)

    def test_time(self):
        """ Test the time property """
        self.assertEqual(np.diff(self.samples[0].time)[0], 1.)

    def test_plot(self):
        """ Test plotting of a timeseries sample """
        plt.figure()
        self.samples[0].plot()
        axes = plt.gcf().axes
        self.assertEqual(len(axes), 1)
        self.assertEqual(axes[0].get_xlabel(), "time [s]")
        self.assertListEqual(
            list(axes[0].lines[0].get_ydata()),
            next(iter(self.samples[0].data.values())).tolist())
        plt.close()

    def test_to_tuple(self):
        """ Test the tuple representation """
        tuple_sample = self.samples[0].to_tuple()
        self.assertIsInstance(tuple_sample, tuple)
        self.assertTrue(helpers.array_safe_equal(tuple_sample[0][0],
                                                 np.arange(3)))
        self.assertEqual(tuple_sample[1], self.samples[0].label)


class TestSpikingSample(unittest.TestCase):
    """
    Tests the spiking sample class.
    """
    samples = [
        data.SpikingSample(
            data={
                SimpleChannel.Test1: np.array([2]),
                SimpleChannel.Test2: np.array([0, 2]),
            },
            sample_rate=1.,
            label=SimpleLabel.Yay,
            filename="things1.txt",
            length=3),
        data.SpikingSample(
            data={
                SimpleChannel.Test1: np.array([2, 5, 7]),
            },
            sample_rate=1.,
            label=SimpleLabel.Nay,
            filename="things2.txt",
            length=10),
    ]

    def test_length(self):
        """ Test the length of a sample """
        self.assertEqual(len(self.samples[0]), 3)

    def test_concat(self):
        """ Test the concatenation of two spiking samples """
        doublesample = self.samples[0] + self.samples[0]
        self.assertListEqual(doublesample.data[SimpleChannel.Test1].tolist(),
                             [2, 5])
        self.assertListEqual(doublesample.data[SimpleChannel.Test2].tolist(),
                             [0, 2, 3, 5])
        self.assertEqual(len(doublesample), 6)
        # label, filename and channels do not match
        with self.assertRaises(ValueError):
            _ = self.samples[0] + self.samples[1]

    def test_slicing(self):
        """ Test the slicing of spiking samples """
        self.assertListEqual(
            self.samples[1][1:4].data[SimpleChannel.Test1].tolist(), [1])
        self.assertEqual(self.samples[1][::2].sample_rate, 0.5)
        with self.assertRaises(TypeError):
            _ = self.samples[1][2]

    def test_empty_like(self):
        """ Test SpikingSample.empty_like """
        self.assertTrue(self.samples[0].equal_without_data(
            data.SpikingSample.empty_like(self.samples[0])))
        self.assertTrue(self.samples[1].equal_without_data(
            data.SpikingSample.empty_like(TestTimeseriesSample.samples[1])))
        empty_sample = data.SpikingSample.empty_like(self.samples[1], length=2)
        self.assertListEqual(empty_sample.channels, [SimpleChannel.Test1])
        self.assertTrue(helpers.array_safe_equal(
            empty_sample.data[SimpleChannel.Test1].data,
            np.zeros(0, dtype=int)))

    def test_plot(self):
        """ Test plotting of a spiking sample """
        plt.figure()
        self.samples[1].plot()
        axes = plt.gcf().axes
        self.assertEqual(len(axes), 1)
        self.assertEqual(axes[0].get_xlabel(), "time [s]")
        plt.close()

    def test_to_tuple(self):
        """ Test the tuple representation """
        tuple_sample = self.samples[0].to_tuple()
        self.assertIsInstance(tuple_sample, tuple)
        self.assertTrue(helpers.array_safe_equal(tuple_sample[0][0],
                                                 np.array([2])))
        self.assertEqual(tuple_sample[1], self.samples[0].label)


class TestDataset(unittest.TestCase):
    """
    Tests the dataset class.
    """
    dataset = data.Dataset(
        TestTimeseriesSample.samples + TestSpikingSample.samples)

    def test_length(self):
        """ Test the length functionality """
        self.assertEqual(len(self.dataset), 4)

    def test_equal(self):
        """ Test the equal & not equal functionality """
        dataset2 = data.Dataset(self.dataset)

        self.assertEqual(self.dataset, dataset2)
        self.assertIsNot(self.dataset, dataset2)
        self.assertEqual(self.dataset, self.dataset)
        self.assertNotEqual(self.dataset, 25.)

    def test_getitem(self):
        """ Test the getitem functionality """
        self.assertIs(self.dataset[0], TestTimeseriesSample.samples[0])
        self.assertIsInstance(self.dataset[:1], data.Dataset)
        self.assertIsInstance(self.dataset[[0, 2]], data.Dataset)
        self.assertEqual(self.dataset[[0, 2]][1], self.dataset.samples[2])

    def test_add(self):
        """ Test the concatenation of datasets """
        concatenated_dataset = self.dataset + self.dataset
        self.assertEqual(len(concatenated_dataset), 8)
        self.assertIs(concatenated_dataset.samples[0],
                      concatenated_dataset.samples[4])

    def test_repr(self):
        """ Test the string representation of Dataset """
        self.assertIn("Dataset(samples.length=4", repr(self.dataset))

    def test_to_tuple(self):
        """ Test the tuple representation """
        tuple_dataset = self.dataset.to_tuple()
        self.assertIsInstance(tuple_dataset, tuple)
        self.assertEqual(len(tuple_dataset), 4)
        for label in map(itemgetter(1), tuple_dataset):
            with self.subTest(label=label):
                self.assertIsInstance(label, data.Label)


class TestChannel(unittest.TestCase):
    """
    Test the channel class.
    """
    def test_value(self):
        """ Test the value """
        self.assertEqual(SimpleChannel.Test1.value, "Test1")


class SimpleDatasetLoader(data.DatasetLoader):
    """
    A simple loader to test the dataset wrappers.
    """
    _available_labels = [SimpleLabel.Yay, SimpleLabel.Nay]
    _available_channels = [SimpleChannel.Test1, SimpleChannel.Test2]

    def load_train(self):
        return data.Dataset([TestTimeseriesSample.samples[0]] * 10
                            + [TestSpikingSample.samples[1]] * 15)

    def load_test(self):
        return data.Dataset([TestTimeseriesSample.samples[1]] * 3
                            + [TestSpikingSample.samples[0]] * 2)


class TestSplitDatasetLoader(unittest.TestCase):
    """
    Tests the split loader.
    """
    base_loader = SimpleDatasetLoader(Path("test/path"))
    wrapped_loader = data.SplitDatasetLoader(base_loader, validation_split=.2)

    def test_load(self):
        """ Test the loading of train and validation data """
        train_data = self.wrapped_loader.load_train()
        test_data = self.wrapped_loader.load_test()
        self.assertEqual(len(train_data), 20)
        self.assertEqual(len(test_data), 5)

    def test_channels(self):
        self.assertListEqual(self.base_loader.channels,
                             [SimpleChannel.Test1, SimpleChannel.Test2])
        self.assertListEqual(self.wrapped_loader.channels,
                             [SimpleChannel.Test1, SimpleChannel.Test2])

    def test_labels(self):
        self.assertListEqual(self.base_loader.labels,
                             [SimpleLabel.Yay, SimpleLabel.Nay])
        self.assertListEqual(self.wrapped_loader.labels,
                             [SimpleLabel.Yay, SimpleLabel.Nay])

    def test_path(self):
        self.assertEqual(self.base_loader.path, Path("test/path"))
        self.assertEqual(self.wrapped_loader.path, Path("test/path"))


class TestTransformDatasetLoader(unittest.TestCase):
    """
    Tests the transform dataset loader.
    """

    @classmethod
    def setUpClass(cls):
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls.base_loader = SimpleDatasetLoader(Path("test/path"))
        cls.cached_loader = data.TransformDatasetLoader(
            cls.base_loader,
            transforms.Compose(transforms.FixedLength(10),
                               transforms.Identity()),
            cache_dir=Path(cls.cache_dir.name),
            test_transform=transforms.FixedLength(12))
        cls.uncached_loader = data.TransformDatasetLoader(
            cls.base_loader, transform=transforms.FixedLength(8))

    @classmethod
    def tearDownClass(cls):
        cls.cache_dir.cleanup()

    def test_load_cached(self):
        """ Test loading of cached train and validation data """
        cache = Path(self.cache_dir.name)

        train_data = self.cached_loader.load_train()
        self.assertEqual(len(train_data), 25)
        self.assertEqual(len(train_data[0]), 10)
        self.assertEqual(len(list(cache.glob("*_train.dataset"))), 1)

        self.cached_loader.cache_interim_results = True
        test_data = self.cached_loader.load_test()
        self.assertEqual(len(test_data), 5)
        self.assertEqual(len(test_data[0]), 12)
        self.assertEqual(len(list(cache.glob("*_test.dataset"))), 2)

        _ = self.cached_loader.load_train()
        self.assertEqual(len(list(cache.glob("*_train.dataset"))), 1)

    def test_load_uncached(self):
        """ Test loading of uncached train and validation data """
        train_data = self.uncached_loader.load_train()
        self.assertEqual(len(train_data), 25)
        self.assertEqual(len(train_data[0]), 8)

        test_data = self.uncached_loader.load_test()
        self.assertEqual(len(test_data), 5)
        self.assertEqual(len(test_data[0]), 8)

    def test_channels(self):
        self.assertListEqual(self.base_loader.channels,
                             [SimpleChannel.Test1, SimpleChannel.Test2])
        self.assertListEqual(self.cached_loader.channels,
                             [SimpleChannel.Test1, SimpleChannel.Test2])

    def test_labels(self):
        self.assertListEqual(self.cached_loader.labels,
                             [SimpleLabel.Yay, SimpleLabel.Nay])

    def test_path(self):
        self.assertEqual(self.cached_loader.path, Path("test/path"))


class TestFunctions(unittest.TestCase):
    """
    Tests the functions of the data module.
    """
    def test_split_dataset(self):
        """
        Test the split_dataset function.
        """
        dataset = data.Dataset([TestTimeseriesSample.samples[0]] * 10
                               + [TestSpikingSample.samples[1]] * 15)
        train_data, val_data = data.split_dataset(dataset, 0.2, seed=0x5EED)

        self.assertEqual(len(train_data), 20)
        self.assertEqual(len(val_data), 5)
        self.assertEqual(
            len([s for s in val_data if s.label == SimpleLabel.Yay]), 2)


if __name__ == "__main__":
    unittest.main()
