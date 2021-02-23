"""
Tests of the ecg module.
"""
import unittest
from pathlib import Path
import numpy as np
from timedatasets import data, ecg
from timedatasets.ecg import EcgChannel


class TestEcg(unittest.TestCase):
    """
    Test the ecg module.
    """

    def test_hdbioai_loader(self):
        """
        Test loading of the competition data.
        """

        path = Path(__file__).parent.joinpath("test_datasets/hdbioai")

        loader = ecg.HdbioaiLoader(path)
        self.assertListEqual(loader.channels, [EcgChannel.III, EcgChannel.I])
        self.assertEqual(len(loader.load_test()), 0)

        dataset = loader.load_train()
        self.assertIsInstance(dataset, data.Dataset)
        self.assertIsInstance(dataset[0], data.TimeseriesSample)
        self.assertEqual(dataset[0].data[EcgChannel.III].shape, (50,))
        self.assertEqual(len(dataset[0].channels), 2)
        self.assertTrue(np.allclose(dataset[0].data[EcgChannel.III],
                                    np.arange(0, 0.05, 0.001)))
        self.assertTrue(np.allclose(dataset[0].data[EcgChannel.I],
                                    np.arange(0.05, 0.1, 0.001)))
        self.assertEqual(dataset[0].label, ecg.EcgLabel.Normal)
        self.assertEqual(dataset[0].sample_rate, 100.)

        self.assertRaises(ValueError, ecg.HdbioaiLoader,
                          path, channels=[EcgChannel.AVF])
        self.assertRaises(ValueError, ecg.NSRDBLoader,
                          path, labels=[ecg.EcgLabel.Noise])

    def test_physionet2017_loader(self):
        """
        Test loading of the physionet 2017 training data.
        """

        path = Path(__file__).parent.joinpath("test_datasets/physionet2017")

        loader = ecg.Physionet2017Loader(path)
        self.assertListEqual(loader.channels, [EcgChannel.I])

        dataset = loader.load_train()
        self.assertIsInstance(dataset, data.Dataset)
        self.assertIsInstance(dataset[0], data.TimeseriesSample)
        self.assertEqual(dataset[0].data[EcgChannel.I].shape, (2714,))
        self.assertEqual(len(dataset[0].channels), 1)
        self.assertAlmostEqual(
            dataset[0].data[EcgChannel.I].mean(), 0, delta=20)
        self.assertEqual(dataset[0].label, ecg.EcgLabel.Normal)
        self.assertEqual(dataset[0].sample_rate, 300.)

        self.assertRaises(ValueError, ecg.Physionet2017Loader,
                          path, channels=[EcgChannel.AVF])

    def test_nsrdb_loader(self):
        """
        Test loading of the nsrdb.
        """
        path = Path(__file__).parent.joinpath("test_datasets/nsrdb")

        loader = ecg.NSRDBLoader(path)
        self.assertListEqual(loader.channels, [EcgChannel.I, EcgChannel.III])
        self.assertEqual(len(loader.load_test()), 0)

        dataset = loader.load_train()
        self.assertEqual(len(dataset), 1)
        self.assertIsInstance(dataset[0], data.TimeseriesSample)
        self.assertEqual(dataset[0].sample_rate, 128)
        self.assertEqual(dataset[0].label, ecg.EcgLabel.Normal)
        self.assertEqual(dataset[0].data[EcgChannel.I].shape, (62000,))
        self.assertEqual(len(dataset[0].channels), 2)

        beat_dataset = loader.load_beats()
        self.assertEqual(len(beat_dataset), 1)
        self.assertIsInstance(beat_dataset[0], data.SpikingSample)
        self.assertEqual(beat_dataset[0].sample_rate, 128)
        self.assertEqual(beat_dataset[0].label, ecg.EcgLabel.Normal)
        self.assertLess(beat_dataset[0].data[EcgChannel.BeatAnnotation].max(),
                        62000)
        self.assertEqual(len(beat_dataset[0].data[EcgChannel.BeatAnnotation]),
                         348)

        self.assertRaises(ValueError, ecg.NSRDBLoader,
                          path, channels=[EcgChannel.AVR])

    def test_mitdb_loader(self):
        """
        Test loading of the mitdb.
        """
        path = Path(__file__).parent.joinpath("test_datasets/nsrdb")

        loader = ecg.MITDBLoader(path)

        dataset = loader.load_train()
        self.assertEqual(dataset[0].label, ecg.EcgLabel.Unknown)

        beat_dataset = loader.load_beats()
        self.assertEqual(beat_dataset[0].label, ecg.EcgLabel.Unknown)

    def test_afdb_loader(self):
        """
        Test loading of the afdb.
        """
        path = Path(__file__).parent.joinpath("test_datasets/afdb")

        loader = ecg.AFDBLoader(path)
        self.assertListEqual(loader.channels, [EcgChannel.I, EcgChannel.III])

        dataset = loader.load_train()
        self.assertEqual(len(dataset), 1)
        self.assertIsInstance(dataset[0], data.TimeseriesSample)
        self.assertEqual(dataset[0].sample_rate, 250)
        self.assertEqual(dataset[0].label, ecg.EcgLabel.AFib)
        self.assertEqual(dataset[0].data[EcgChannel.I].shape, (76000,))
        self.assertEqual(len(dataset[0].channels), 2)

        beat_dataset = loader.load_beats()
        self.assertEqual(len(beat_dataset), 1)
        self.assertIsInstance(beat_dataset[0], data.SpikingSample)
        self.assertEqual(beat_dataset[0].sample_rate, 250)
        self.assertEqual(beat_dataset[0].label, ecg.EcgLabel.AFib)
        self.assertLess(beat_dataset[0].data[EcgChannel.BeatAnnotation].max(),
                        76000)
        self.assertEqual(len(beat_dataset[0].data[EcgChannel.BeatAnnotation]),
                         563)

        self.assertRaises(ValueError, ecg.AFDBLoader,
                          path, channels=[EcgChannel.AVL])


if __name__ == "__main__":
    unittest.main()
