import torch
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
