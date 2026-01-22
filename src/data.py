from __future__ import annotations

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

from .features import FeatureExtractor

class SCRDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        wav_dir: str,
        sample_rate: int,
        clip_seconds: float,
        feature_extractor: FeatureExtractor,
    ):
        self.df = pd.read_csv(csv_path)
        self.wav_dir = wav_dir
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * clip_seconds)
        self.fe = feature_extractor

        # compute number of classes from csv
        self.num_classes = int(self.df["class_id"].max()) + 1

    def __len__(self) -> int:
        return len(self.df)

    def _load_wav(self, filename: str) -> torch.Tensor:
        path = os.path.join(self.wav_dir, filename)
        wav, sr = torchaudio.load(path)  # wav: [C, T]
        # to mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # resample if needed
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # pad / trim to fixed length
        T = wav.size(1)
        if T < self.target_len:
            pad = self.target_len - T
            wav = torch.nn.functional.pad(wav, (0, pad))
        elif T > self.target_len:
            wav = wav[:, : self.target_len]

        return wav

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        filename = row["dst_filename"]
        y = int(row["class_id"])

        wav = self._load_wav(filename)          # [1, T]
        x = self.fe(wav)                        # [1, n_mels, time]
        return x, torch.tensor(y, dtype=torch.long)

def create_loader(
    csv_path: str,
    wav_dir: str,
    batch_size: int,
    sample_rate: int,
    clip_seconds: float,
    num_workers: int,
    shuffle: bool,
):
    fe = FeatureExtractor(
        sample_rate=sample_rate,
        n_mels=64,
        n_fft=400,
        hop_length=160,
    )
    ds = SCRDataset(
        csv_path=csv_path,
        wav_dir=wav_dir,
        sample_rate=sample_rate,
        clip_seconds=clip_seconds,
        feature_extractor=fe,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return ds, loader
