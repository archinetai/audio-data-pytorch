import random

import torch
from torch import Tensor, nn


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
