import random

import torch
import torchaudio
from torch import Tensor, nn


class Crop(nn.Module):
    """Crops waveform to fixed size"""

    def __init__(self, size: int, start: int = 0) -> None:
        super().__init__()
        self.size = size
        self.start = start

    def forward(self, x: Tensor) -> Tensor:
        x = x[:, self.start :]
        channels, length = x.shape

        if length < self.size:
            padding_length = self.size - length
            padding = torch.zeros(channels, padding_length).to(x)
            return torch.cat([x, padding], dim=1)
        else:
            return x[:, 0 : self.size]


class RandomCrop(nn.Module):
    """Crops random chunk from the waveform"""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        # Pick start position
        length = x.shape[1]
        start = random.randint(0, max(length - self.size, 0))
        # Crop from random start
        x = x[:, start:]
        channels, length = x.shape
        # Pad to end if not large enough, else crop end
        if length < self.size:
            padding_length = self.size - length
            padding = torch.zeros(channels, padding_length).to(x)
            return torch.cat([x, padding], dim=1)
        else:
            return x[:, 0 : self.size]


class OverlapChannels(nn.Module):
    """Overlaps all channels into one"""

    def forward(self, x: Tensor) -> Tensor:
        return torch.sum(x, dim=0, keepdim=True)  # 'c l -> 1 l'


class Resample(nn.Module):
    """Resamples frequency of waveform"""

    def __init__(self, source: int, target: int):
        super().__init__()
        self.transform = torchaudio.transforms.Resample(
            orig_freq=source, new_freq=target
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x)


class Scale(nn.Module):
    """Scales waveform (change volume)"""

    def __init__(
        self,
        scale: float,
    ):
        super().__init__()
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale
