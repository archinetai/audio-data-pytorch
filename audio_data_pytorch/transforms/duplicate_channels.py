from torch import Tensor, nn


class DuplicateChannels(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        c = x.shape[0]
        if c < 2:
            return x.repeat(2, 1)
        return x
