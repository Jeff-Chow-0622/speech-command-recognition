from __future__ import annotations

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import soundfile as sf
import librosa

# Optional torchaudio for loading/resampling
_TORCHAUDIO_OK = False
try:
    import torchaudio  # type: ignore
    _TORCHAUDIO_OK = True
except Exception:
    _TORCHAUDIO_OK = False

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
        self.num_classes = int(self.df["class_id"].max()) + 1

    def __len__(self) -> int:
        return len(self.df)

    def _load_wav_librosa(self, path: str) -> torch.Tensor:
        # soundfile reads faster and preserves original sr
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:  # stereo -> mono
            audio = audio.mean(axis=1).astype(np.float32)

        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate).astype(np.float32)

        # pad/trim
        T = audio.shape[0]
        if T < self.target_len:
            audio = np.pad(audio, (0, self.target_len - T), mode="constant")
        elif T > self.target_len:
            audio = audio[: self.target_len]

        return torch.from_numpy(audio).unsqueeze(0)  # [1, T]

    def _load_wav_torchaudio(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)  # [C, T]
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        T = wav.size(1)
        if T < self.target_len:
            wav = torch.nn.functional.pad(wav, (0, self.target_len - T))
        elif T > self.target_len:
            wav = wav[:, : self.target_len]
        return wav

    def _load_wav(self, filename: str) -> torch.Tensor:
        path = os.path.join(self.wav_dir, filename)
        if _TORCHAUDIO_OK:
            try:
                return self._load_wav_torchaudio(path)
            except Exception:
                # if torchaudio explodes at runtime, fallback
                return self._load_wav_librosa(path)
        return self._load_wav_librosa(path)

    def __getitem__(self, idx: int):
        # Robust loading: if a file is missing/corrupt, try the next sample
       	for _ in range(10):
            row = self.df.iloc[idx]
            filename = row["dst_filename"]
            y = int(row["class_id"])

            try:
                wav = self._load_wav(filename)   # [1, T]
                x = self.fe(wav)                 # [1, n_mels, time]
                return x, torch.tensor(y, dtype=torch.long)

            except Exception as e:
                # Optional: enable this line if you want to see which files are skipped
                # print(f"[WARN] Skipping unreadable/missing file: {filename} ({e})")
                idx = (idx + 1) % len(self.df)

        raise RuntimeError(
            f"Too many unreadable audio files around index {idx}. "
            f"Last attempted filename: {filename}"
        )



def create_loader(
    csv_path: str,
    wav_dir: str,
    batch_size: int,
    sample_rate: int,
    clip_seconds: float,
    num_workers: int,
    shuffle: bool,
    n_mels: int = 64,
    n_fft: int = 400,
    hop_length: int = 160,
):
    fe = FeatureExtractor(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
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
