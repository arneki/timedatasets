from __future__ import annotations
from typing import List, Optional
import numpy as np
import matplotlib as mpl
import matplotlib.patheffects as path_effects


def array_safe_equal(value1, value2) -> bool:
    """
    Check if the two values are equal, numpy.ndarray instances have to be equal
    elementwise.

    :param value1: First value to compare.
    :param value2: Second value to compare.

    :return: Boolean equality.
    """

    if value1 is value2:
        return True
    if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
        try:
            return value1.shape == value2.shape \
                and value1.dtype == value2.dtype \
                and (np.allclose(value1, value2) or value1.size == 0)
        except TypeError:
            return value1.shape == value2.shape \
                and value1.dtype == value2.dtype \
                and (value1 == value2).all()
    return value1 == value2


def plot_confusion_matrix(axes: mpl.axes.Axes, conf_matrix: np.ndarray,
                          label_names: Optional[List[str]] = None,
                          badge: Optional[str] = None):
    """
    Plot a confusion matrix.
    :param axes: The matplotlib axes.
    :param conf_matrix: Confusion matrix.
    :param label_names: String representations of the labels.
    :param badge: Text that will be added as a badge in the top right.
    """
    n_samples = np.sum(conf_matrix, axis=1)
    conf_matrix_percent = (conf_matrix.T / n_samples).T * 100
    corners = np.arange(-.5, len(conf_matrix))
    axes.pcolormesh(
        *np.meshgrid(corners, corners), conf_matrix_percent, vmin=0, vmax=100)
    axes.set_aspect('equal')
    axes.invert_yaxis()
    axes.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    axes.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    axes.set_ylabel('true label')
    axes.set_xlabel('predicted label')
    if label_names:
        axes.set_xticklabels([""] + label_names)
        axes.set_yticklabels([""] + label_names)
    text_args = {'color': 'w', 'ha': 'center'}
    for i, row in enumerate(conf_matrix):
        for j, num in enumerate(row):
            num_text = axes.text(j, i, f"{num}",
                                 fontsize=12, va='bottom', **text_args)
            percent_text = axes.text(j, i + 0.02,
                                     f"{num / n_samples[i] * 100:.1f}%",
                                     fontsize=8, va='top', **text_args)
            for text in [num_text, percent_text]:
                text.set_path_effects([
                    path_effects.Stroke(linewidth=2, foreground='k', alpha=.5),
                    path_effects.Normal()])
    if badge:
        axes.text(1.05, 1.05, badge,
                  horizontalalignment='right', verticalalignment='top',
                  transform=axes.transAxes,
                  bbox=dict(fc='white', ec="k"))
