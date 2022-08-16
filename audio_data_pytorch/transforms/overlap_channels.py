import torch
from torch import Tensor, nn


class OverlapChannels(nn.Module):
    """Overlaps all channels into one"""

    def forward(self, x: Tensor) -> Tensor:
        return torch.sum(x, dim=0, keepdim=True)  # 'c l -> 1 l'
