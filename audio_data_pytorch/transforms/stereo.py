from torch import Tensor, nn


class Stereo(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        channels = shape[0]
        if len(shape) == 1:  # s -> 2, s
            x = x.unsqueeze(0).repeat(2, 1)
        elif len(shape) == 2:
            if channels == 1:  # 1, s -> 2, s
                x = x.repeat(2, 1)
            elif channels > 2:  # ?, s -> 2,s
                x = x[:2, :]
        return x
