"""
Classes and functions to deal with ecg data.
"""
from __future__ import annotations
from abc import abstractmethod
import enum
from pathlib import Path
from typing import Dict, List, Iterable
import numpy as np
from scipy.io import loadmat
import wfdb

from timedatasets import data


class EcgLabel(data.Label):
    """
    Possible labels for an ecg trace.
    """
    Normal = 0
    AFib = 1
    Other = 2
    Noise = 3
    Unknown = 4


class EcgChannel(data.Channel):
    # pylint: disable=invalid-name,blacklisted-name
    I = enum.auto()
    II = enum.auto()
    III = enum.auto()
    AVL = enum.auto()
    AVR = enum.auto()
    AVF = enum.auto()
    BeatAnnotation = enum.auto()
    Other = enum.auto()


class MITBIHAnnotationSymbol(enum.Enum):
    """
    Beat annotation symbols of the MIT-BIH databases.
    Cf. https://archive.physionet.org/physiobank/annotations.shtml
    """
    # pylint: disable=invalid-name
    normal_beat = "N"
    left_bundle_branch_block_beat = "L"
    right_bundle_branch_block_beat = "R"
    bundle_branch_block_beat = "B"
    atrial_premature_beat = "A"
    aberrated_atrial_premature_beat = "a"
    nodal_premature_beat = "J"
    supraventricular_premature_or_ectopic_beat = "S"
    premature_ventricular_contraction = "V"
    r_on_t_premature_ventricular_contraction = "r"
    fusion_of_ventricular_and_normal_beat = "F"
    atrial_escape_beat = "e"
    nodal_escape_beat = "j"
    supraventricular_escape_beat = "n"
    ventricular_escape_beat = "E"
    paced_beat = "/"
    fusion_of_paced_and_normal_beat = "f"
    unclassifiable_beat = "Q"
    beat_not_classified = "?"


class HdbioaiLoader(data.DatasetLoader):
    """
    Load the ecg training dataset of the competition
    `Energieeffizientes KI-System`.
    See `https://www.bmbf.de/foerderungen/bekanntmachung-2371.html`.
    """

    _available_channels: List[data.Channel] = [EcgChannel.III, EcgChannel.I]
    _available_labels: List[data.Label] = [EcgLabel.Normal, EcgLabel.AFib]

    def load_train(self) -> data.Dataset:
        chan_idxs = [self._available_channels.index(c) for c in self.channels]
        samples = []
        for label in self.labels:
            data_dir = "sinus_rhythm_8k" if label == EcgLabel.Normal \
                       else "atrial_fibrillation_8k"
            files = sorted(self.path.joinpath(data_dir).glob("*.ecg"))

            for fpath in files:
                with fpath.open("rb") as f:
                    header_raw, _, data_raw = f.read().partition(b'\x00')
                header_lines = header_raw.decode("UTF-8").strip().split("\r\n")
                ecg_header = dict((line.split(": ") for line in header_lines))
                ecg_data = np.frombuffer(data_raw, dtype=np.uint16)\
                    .reshape(-1, int(ecg_header["Channels"]))
                factor = np.float32(ecg_header['Factor'])
                traces = ecg_data.T.astype(np.int16)[chan_idxs]
                traces -= int(float(ecg_header['Offset']))
                samples.append(data.TimeseriesSample(
                    data=dict(zip(self.channels, traces * factor)),
                    label=label,
                    sample_rate=float(ecg_header['Sample Rate']),
                    filename=str(fpath.relative_to(self.path)),
                ))

        return data.Dataset(samples)

    def load_test(self) -> data.Dataset:
        # there is no test data yet
        return data.Dataset([])


class Physionet2017Loader(data.DatasetLoader):
    """
    Load the ecg data, labels and filenames of the physionet 2017 training set.
    """

    _available_channels: List[data.Channel] = [EcgChannel.I]
    _available_labels: List[data.Label] = [
        EcgLabel.Normal, EcgLabel.AFib, EcgLabel.Noise, EcgLabel.Other
    ]

    def _load(self, data_path: Path) -> data.Dataset:
        # magic numbers from https://archive.physionet.org/challenge/2017/
        sample_rate = 300.
        factor = np.float32(0.001)

        label_idx_file: Dict[data.Label, str] = {
            EcgLabel.Normal: "RECORDS-normal",
            EcgLabel.AFib: "RECORDS-af",
            EcgLabel.Other: "RECORDS-other",
            EcgLabel.Noise: "RECORDS-noisy",
        }
        samples = []
        for label in self.labels:
            with data_path.joinpath(label_idx_file[label]).open("r") as f_idx:
                file_names = list(filter(None,
                                         (name.rstrip() for name in f_idx)))
            for file_name in file_names:
                file_path = data_path.joinpath(f"{file_name}.mat")
                trace = loadmat(file_path)['val']
                samples.append(data.TimeseriesSample(
                    data=dict(zip(self.channels, trace * factor)),
                    label=label,
                    sample_rate=sample_rate,
                    filename=str(file_path.relative_to(self.path)),
                ))

        return data.Dataset(samples)

    def load_train(self) -> data.Dataset:
        return self._load(self.path.joinpath("training"))

    def load_test(self) -> data.Dataset:
        return self._load(self.path.joinpath("validation"))


class MITBIHLoader(data.DatasetLoader):
    """
    Load datasets from the MIT-BIH databases.
    E.g. https://www.physionet.org/content/nsrdb/1.0.0/
    """

    @property
    @abstractmethod
    def annotation_extensions(self) -> List[str]:
        """
        Possible extensions for a file that contains beat annotations.
        The beats will be read form the first extension with existing file.
        """
        raise NotImplementedError

    @property
    def _records(self) -> List[Path]:
        return sorted(
            [rec_p.with_suffix("") for rec_p in self.path.glob('*.dat')])

    def load_train(self) -> data.Dataset:
        chan_idxs = [self._available_channels.index(c) for c in self.channels]
        samples = []
        for record_path in self._records:
            # get header and trace signals
            signals, header = wfdb.rdsamp(str(record_path),
                                          channels=chan_idxs)
            assert signals.shape[0] == header['sig_len']
            samples.append(data.TimeseriesSample(
                data=dict(zip(self.channels, signals.T.astype(np.float32))),
                label=self._available_labels[0],
                sample_rate=header['fs'],
                filename=str(record_path.relative_to(self.path)),
            ))

        return data.Dataset(samples)

    def load_test(self) -> data.Dataset:
        # there is no test data yet
        return data.Dataset([])

    def load_beats(self, symbols: Iterable[MITBIHAnnotationSymbol]
                   = MITBIHAnnotationSymbol) -> data.Dataset:
        """
        Load the annotated beat positions of the dataset as spiking samples.

        :param symbols: The symbols of the beat annotations to load.
            Defaults to all possible codes. A full description of them can be
            found at:
            :url:`https://archive.physionet.org/physiobank/annotations.shtml`
        """
        spiking_samples = []
        for record_path in self._records:
            # get beats from annotations
            for annotation_extension in self.annotation_extensions:
                try:
                    annotation = wfdb.rdann(
                        str(record_path),
                        extension=annotation_extension)
                    filename = str(record_path.relative_to(
                        self.path).with_suffix(f".{annotation_extension}"))
                    beat_mask = np.any(
                        [np.array(annotation.symbol) == symbol.value
                         for symbol in symbols],
                        axis=0)
                    beats = annotation.sample[beat_mask]
                except FileNotFoundError:
                    if annotation_extension == self.annotation_extensions[-1]:
                        raise FileNotFoundError(
                            f"No annotations file found for {record_path}.")

            header = wfdb.rdheader(str(record_path))
            spiking_samples.append(data.SpikingSample(
                data={
                    EcgChannel.BeatAnnotation: beats[beats < header.sig_len]},
                label=self._available_labels[0],
                sample_rate=header.fs,
                filename=filename,
                length=header.sig_len))

        return data.Dataset(spiking_samples)


class NSRDBLoader(MITBIHLoader):
    """
    Load the data of the MIT-BIH Normal Sinus Rhythm Database (nsrdb).
    """

    _available_channels: List[data.Channel] = [EcgChannel.I, EcgChannel.III]
    _available_labels: List[data.Label] = [EcgLabel.Normal]
    annotation_extensions = ['atr']


class AFDBLoader(MITBIHLoader):
    """
    Load the data of the MIT-BIH Atrial Fibrillation Database (afdb)
    """

    _available_channels: List[data.Channel] = [EcgChannel.I, EcgChannel.III]
    _available_labels: List[data.Label] = [EcgLabel.AFib]
    annotation_extensions = ['qrsc', 'qrs']


class MITDBLoader(MITBIHLoader):
    """
    Load the data of the MIT-BIH Arrhythmia Database (mitdb)
    """
    _available_channels: List[data.Channel] = [EcgChannel.I, EcgChannel.III]
    _available_labels: List[data.Label] = [EcgLabel.Unknown]
    annotation_extensions = ['atr']
