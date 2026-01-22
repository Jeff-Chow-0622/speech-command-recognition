from dataclasses import dataclass

@dataclass
class Config:
    # Audio
    sample_rate: int = 16000
    clip_seconds: float = 1.0  # pad/trim to fixed length
    n_mels: int = 64
    n_fft: int = 400          # 25ms at 16kHz
    hop_length: int = 160     # 10ms at 16kHz

    # Training
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    seed: int = 42
    num_workers: int = 0  # Windows: keep 0 to avoid multiprocessing issues

    # Paths (relative to repo root)
    train_csv: str = "data/scr_train.csv"
    train_wav_dir: str = "data/train"
    val_csv: str = "data/scr_val.csv"
    val_wav_dir: str = "data/validation"
