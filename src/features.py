import torch
import torchaudio

class FeatureExtractor:
    """
    Converts waveform -> log-mel spectrogram tensor: [1, n_mels, time]
    """
    def __init__(self, sample_rate: int, n_mels: int, n_fft: int, hop_length: int):
        self.sample_rate = sample_rate
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [1, T]
        mel = self.mel(wav)              # [1, n_mels, time]
        log_mel = self.amp_to_db(mel)    # log scale
        # Normalize per-sample (optional but helps)
        mean = log_mel.mean()
        std = log_mel.std().clamp_min(1e-6)
        log_mel = (log_mel - mean) / std
        return log_mel
