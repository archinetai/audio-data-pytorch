import torchaudio
from torch import Tensor, nn


class Resample(nn.Module):
    """Resamples frequency of waveform"""

    def __init__(self, source: int, target: int):
        super().__init__()
        self.transform = torchaudio.transforms.Resample(
            orig_freq=source, new_freq=target
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x)
