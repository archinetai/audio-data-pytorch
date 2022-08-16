from torch import Tensor, nn


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
