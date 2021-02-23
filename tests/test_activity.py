"""
Tests of the activity module.
"""
from enum import auto
import unittest
from pathlib import Path
from timedatasets import data, activity


class TestActivity(unittest.TestCase):
    """
    Test the activity module.
    """

    def test_har_loader(self):
        """
        Test loading of the HAR dataset.
        """

        path = Path(__file__).parent.joinpath("test_datasets/har")

        loader = activity.HARLoader(path, [activity.Sensor.BodyAccX])

        dataset = loader.load_test()
        self.assertIsInstance(dataset, data.Dataset)
        self.assertEqual(len(dataset), 5)
        self.assertIsInstance(dataset[0], data.TimeseriesSample)
        self.assertEqual(len(dataset[0].data[activity.Sensor.BodyAccX]), 128)
        self.assertEqual(len(dataset[0].channels), 1)
        self.assertEqual(dataset[0].label, activity.ActivityLabel.Standing)
        self.assertEqual(dataset[0].sample_rate, 50.)

        class TestSensor(data.Channel):
            Unknown = auto()

        with self.assertRaises(ValueError):
            activity.HARLoader(path, channels=[TestSensor.Unknown])


if __name__ == "__main__":
    unittest.main()
