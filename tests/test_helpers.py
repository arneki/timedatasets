"""
Tests of the helpers module.
"""
import unittest
import numpy as np
import matplotlib.pyplot as plt

from timedatasets import helpers


class TestHelpers(unittest.TestCase):
    """
    Test the helpers module.
    """

    def test_array_safe_equal(self):
        """
        Test the array_safe_equal function.
        """

        self.assertTrue(helpers.array_safe_equal(1, 1))
        self.assertTrue(helpers.array_safe_equal(np.arange(2), np.arange(2)))
        self.assertFalse(helpers.array_safe_equal(1, 2))
        self.assertTrue(helpers.array_safe_equal(np.array(["A", "B"]),
                                                 np.array(["A", "B"])))
        arr = np.array([2, 3])
        self.assertTrue(helpers.array_safe_equal(arr, arr))

    def test_plot_confusion_matrix(self):
        """
        Tests the plotting function.
        """
        confusion_matrix = np.array([[3, 1], [1, 2]])
        _, axes = plt.subplots(1, 1)
        helpers.plot_confusion_matrix(
            axes, confusion_matrix, ["A", "B"], badge="special")

        self.assertEqual(axes.get_xlabel(), "predicted label")
        self.assertEqual(axes.get_ylabel(), "true label")
        self.assertEqual(axes.get_xticklabels()[1].properties()["text"], "A")
        self.assertEqual(axes.get_xlim(), (-0.5, 1.5))


if __name__ == "__main__":
    unittest.main()
