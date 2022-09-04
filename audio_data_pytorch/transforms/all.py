from typing import Optional

from torch import Tensor, nn

from ..utils import exists
from .crop import Crop
from .loudness import Loudness
from .overlap_channels import OverlapChannels
from .randomcrop import RandomCrop
from .resample import Resample
from .scale import Scale
from .stereo import Stereo


class AllTransform(nn.Module):
    def __init__(
        self,
        source_rate: Optional[int] = None,
        target_rate: Optional[int] = None,
        crop_size: Optional[int] = None,
        random_crop_size: Optional[int] = None,
        loudness: Optional[int] = None,
        scale: Optional[float] = None,
        use_stereo: bool = False,
        overlap_channels: bool = False,
    ):
        super().__init__()

        message = "Both source_rate and target_rate must be provided"
        assert not exists(source_rate) ^ exists(target_rate), message

        message = "Loudness requires target_rate"
        assert not exists(loudness) or exists(target_rate), message

        self.transform = nn.Sequential(
            Resample(source=source_rate, target=target_rate)  # type: ignore
            if exists(source_rate) and source_rate != target_rate
            else nn.Identity(),
            RandomCrop(random_crop_size) if exists(random_crop_size) else nn.Identity(),
            Crop(crop_size) if exists(crop_size) else nn.Identity(),
            OverlapChannels() if overlap_channels else nn.Identity(),
            Stereo() if use_stereo else nn.Identity(),
            Loudness(sampling_rate=target_rate, target=loudness)  # type: ignore
            if exists(loudness)
            else nn.Identity(),
            Scale(scale) if exists(scale) else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x)
