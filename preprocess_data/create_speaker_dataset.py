import random
from pathlib import Path
from typing import Union
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class ClassificationDatasetSpkr(Dataset):
    """Sample utterances from speakers."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        speaker_infos: dict,
        n_utterances: int,
        seg_len: int,
    ):
        """
        Args:
            data_dir (string): path to the directory of pickle files.
            n_utterances (int): # of utterances per speaker to be sampled.
            seg_len (int): the minimum length of segments of utterances.
        """

        self.data_dir = data_dir
        self.n_utterances = n_utterances
        self.seg_len = seg_len
        self.speaker_infos = speaker_infos
        self.infos = []

        for spk_idx, uttr_infos in enumerate(self.speaker_infos.values()):
            feature_paths = [
                (uttr_info["feature_path"], spk_idx)
                for uttr_info in uttr_infos
                if uttr_info["mel_len"] > self.seg_len
            ]
            if len(feature_paths) > n_utterances:
                self.infos.append(feature_paths)

        self.infos_flatten = [i_ for s_ in self.infos for i_ in s_]

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        feature_paths_tuple = random.sample(self.infos[index], self.n_utterances)
        feature_path_unpacked, speaker_unpacked = zip(*feature_paths_tuple)

        uttrs_spkrs = [
            (torch.load(Path(self.data_dir, feature_path)), spkr)
            for feature_path, spkr in zip(
                list(feature_path_unpacked),
                list(speaker_unpacked),
            )
        ]
        lefts = [
            (random.randint(0, uttr[0].shape[0] - self.seg_len), uttr[1])
            for uttr in uttrs_spkrs
        ]
        segments = [
            (uttr[0][left[0] : left[0] + self.seg_len, :], uttr[1])
            for uttr, left in zip(uttrs_spkrs, lefts)
        ]
        feat, spk = zip(*segments)

        feat = torch.tensor(np.array(feat))

        return feat, spk


class ClassificationDatasetSpkrV2(Dataset):
    """Sample utterances from speakers."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        speaker_infos: dict,
        seg_len: int,
    ):
        """
        Args:
            data_dir (string): path to the directory of pickle files.
            n_utterances (int): # of utterances per speaker to be sampled.
            seg_len (int): the minimum length of segments of utterances.
        """

        self.data_dir = data_dir

        self.seg_len = seg_len
        self.speaker_infos = speaker_infos
        self.infos = []

        for spk_idx, uttr_infos in enumerate(self.speaker_infos.values()):
            feature_paths = [
                (uttr_info["feature_path"], spk_idx)
                for uttr_info in uttr_infos
                if uttr_info["mel_len"] > self.seg_len
            ]

            self.infos.append(feature_paths)

        self.infos_flatten = [i_ for s_ in self.infos for i_ in s_]

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        feature_paths_tuple = random.sample(self.infos[index], len(self.infos[index]))
        feature_path_unpacked, speaker_unpacked = zip(*feature_paths_tuple)

        uttrs_spkrs = [
            (torch.load(Path(self.data_dir, feature_path)), spkr)
            for feature_path, spkr in zip(
                list(feature_path_unpacked),
                list(speaker_unpacked),
            )
        ]
        lefts = [
            (random.randint(0, uttr[0].shape[0] - self.seg_len), uttr[1])
            for uttr in uttrs_spkrs
        ]
        segments = [
            (uttr[0][left[0] : left[0] + self.seg_len, :], uttr[1])
            for uttr, left in zip(uttrs_spkrs, lefts)
        ]
        feat, spk = zip(*segments)

        feat = torch.tensor(np.array(feat))

        return feat, spk


def collateSpkr(batch):
    """Collate a whole batch of utterances."""
    feat = [sample[0] for sample in batch]
    speaker = [sample[1] for sample in batch]

    flatten_speaker = [item for s in speaker for item in s]
    flatten_speaker = torch.tensor(flatten_speaker, dtype=torch.long)

    flatten = [u for s in feat for u in s]
    flatten = pad_sequence(flatten, batch_first=True, padding_value=0)
    return flatten, flatten_speaker


class SubDatasetSpk(Dataset):
    """To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units."""

    def __init__(self, original_dataset, sub_labels_spk):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []

        for index in range(len(self.dataset)):
            label_spk = np.unique(self.dataset[index][1])
            if label_spk in sub_labels_spk:
                self.sub_indeces.append(index)

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        return sample
