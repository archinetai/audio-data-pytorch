import pyloudnorm as pyln
import torch
from torch import Tensor, nn


class Loudness(nn.Module):
    """Normalizes to target loudness using BS.1770-4, requires pyloudnorm"""

    def __init__(self, sampling_rate: int, target: float):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.target = target
        self.meter = pyln.Meter(sampling_rate)

    def forward(self, x: Tensor) -> Tensor:
        channels, length = x.shape
        # Measure sample loudness
        x_numpy = x.numpy().T
        loudness = self.meter.integrated_loudness(data=x_numpy)
        # Don't normalize zeros sample (i.e. silence)
        if loudness == -float("inf"):
            return x
        # Normalize sample loudness
        x_normalized = pyln.normalize.loudness(
            data=x_numpy, input_loudness=loudness, target_loudness=self.target
        )
        # Return normalized as torch Tensor
        return torch.from_numpy(x_normalized.T)
