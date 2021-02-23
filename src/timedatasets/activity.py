"""
Classes and functions to deal with activity data.
"""
from __future__ import annotations
from enum import auto
from pathlib import Path
from typing import Dict
import numpy as np

from timedatasets import data


class ActivityLabel(data.Label):
    """
    Labels for human activities.
    """
    Walking = 0
    WalkingUpstairs = 1
    WalkingDownstairs = 2
    Sitting = 3
    Standing = 4
    Laying = 5


class Sensor(data.Channel):
    """
    Sensor channels.
    """
    BodyAccX = auto()
    BodyAccY = auto()
    BodyAccZ = auto()
    BodyGyroX = auto()
    BodyGyroY = auto()
    BodyGyroZ = auto()
    TotalAccX = auto()
    TotalAccY = auto()
    TotalAccZ = auto()


class HARLoader(data.DatasetLoader):
    """
    Loads the UCI Human Activity Recognition dataset.

    C.f. https://archive.ics.uci.edu/ml/\
datasets/Human+Activity+Recognition+Using+Smartphones
    """

    _available_channels = [
        Sensor.BodyAccX, Sensor.BodyAccY, Sensor.BodyAccZ,
        Sensor.BodyGyroX, Sensor.BodyGyroY, Sensor.BodyGyroZ,
        Sensor.TotalAccX, Sensor.TotalAccY, Sensor.TotalAccZ
    ]
    _available_labels = [
        ActivityLabel.Walking, ActivityLabel.WalkingUpstairs,
        ActivityLabel.WalkingDownstairs, ActivityLabel.Sitting,
        ActivityLabel.Standing, ActivityLabel.Laying
    ]

    def _load(self, path: Path) -> data.Dataset:
        sample_labels = np.loadtxt(path.joinpath(f"y_{path.stem}.txt")) - 1
        sample_idxs = []
        for label in self.labels:
            sample_idxs += np.nonzero(sample_labels == label)[0].tolist()

        signals = []
        signals_path = path.joinpath("Inertial Signals")
        channel_file_prefix: Dict[data.Channel, str] = {
            Sensor.BodyAccX: "body_acc_x",
            Sensor.BodyAccY: "body_acc_y",
            Sensor.BodyAccZ: "body_acc_z",
            Sensor.BodyGyroX: "body_gyro_x",
            Sensor.BodyGyroY: "body_gyro_y",
            Sensor.BodyGyroZ: "body_gyro_z",
            Sensor.TotalAccX: "total_acc_x",
            Sensor.TotalAccY: "total_acc_y",
            Sensor.TotalAccZ: "total_acc_z"
        }
        for channel in self.channels:
            channel_path = signals_path.joinpath(
                f"{channel_file_prefix[channel]}_{path.stem}.txt")
            signals.append(
                np.loadtxt(channel_path, dtype=np.float32)[sample_idxs]
            )
        signals = np.stack(signals, axis=1)

        samples = []
        for sample_idx, label, signal \
                in zip(sample_idxs, sample_labels[sample_idxs], signals):
            samples.append(data.TimeseriesSample(
                data=dict(zip(self.channels, signal)),
                label=ActivityLabel(label),
                filename=f"{path.stem}/{sample_idx}",
                sample_rate=50.,
            ))

        return data.Dataset(samples)

    def load_train(self) -> data.Dataset:
        return self._load(self.path.joinpath("train"))

    def load_test(self) -> data.Dataset:
        return self._load(self.path.joinpath("test"))
