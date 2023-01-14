import math
import random
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torchaudio
from tinytag import TinyTag
from torch import Tensor
from torch.utils.data import Dataset

from ..utils import fast_scandir, is_silence


def get_all_wav_filenames(paths: Sequence[str], recursive: bool) -> List[str]:
    extensions = [".wav", ".flac"]
    filenames = []
    for path in paths:
        _, files = fast_scandir(path, extensions, recursive=recursive)
        filenames.extend(files)
    return filenames


class WAVDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Sequence[str]],
        recursive: bool = False,
        transforms: Optional[Callable] = None,
        sample_rate: Optional[int] = None,
        random_crop_size: int = None,
        check_silence: bool = True,
        with_ID3: bool = False,
    ):
        self.paths = path if isinstance(path, (list, tuple)) else [path]
        self.wavs = get_all_wav_filenames(self.paths, recursive=recursive)
        self.transforms = transforms
        self.sample_rate = sample_rate
        self.check_silence = check_silence
        self.with_ID3 = with_ID3
        self.random_crop_size = random_crop_size
        assert (
            not random_crop_size or sample_rate
        ), "Optimized random crop requires sample_rate to be set."

    # Instead of loading the whole file and chopping out our crop,
    # we only load what we need.
    def optimized_random_crop(self, idx: int) -> Tuple[Tensor, int]:
        # Get length/audio info
        info = torchaudio.info(self.wavs[idx])
        length = info.num_frames
        sample_rate = info.sample_rate

        # Calculate correct number of samples to read based on actual
        # and intended sample rate
        ratio = 1 if (self.sample_rate is None) else sample_rate / self.sample_rate
        crop_size = length if (self.random_crop_size is None) else math.ceil(self.random_crop_size * ratio)  # type: ignore
        frame_offset = random.randint(0, max(length - crop_size, 0))

        # Load the samples
        waveform, sample_rate = torchaudio.load(
            filepath=self.wavs[idx], frame_offset=frame_offset, num_frames=crop_size
        )

        # Pad with zeroes if the sizes aren't quite right
        # (e.g., rates aren't exact multiples)
        if len(waveform[0]) < crop_size:
            waveform = torch.nn.functional.pad(
                waveform,
                pad=(0, crop_size - len(waveform[0])),
                mode="constant",
                value=0,
            )

        return waveform, sample_rate

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tensor,
        Tuple[Tensor, int],
        Tuple[Tensor, Tensor],
        Tuple[Tensor, List[str], List[str]],
        Tuple[Tensor, TinyTag],
    ]:  # type: ignore
        invalid_audio = False

        # Loop until we find a valid audio sample
        while True:
            # If last sample was invalid, use a new random one.
            if invalid_audio:
                idx = random.randrange(len(self))

            # Catch invalid audio files
            try:
                # Read ID3 tags if specified
                if self.with_ID3:
                    tag = TinyTag.get(self.wavs[idx])

                # Read with optimized crop if needed
                if hasattr(self, "random_crop_size"):
                    waveform, sample_rate = self.optimized_random_crop(int(idx))
                else:
                    waveform, sample_rate = torchaudio.load(filepath=self.wavs[idx])
            except Exception:
                invalid_audio = True
                continue

            # Apply sample rate transform if necessary
            if self.sample_rate and sample_rate != self.sample_rate:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.sample_rate
                )(waveform)

                # Downsampling can result in slightly different sizes.
                if hasattr(self, "random_crop_size"):
                    waveform = waveform[:, : self.random_crop_size]

            # Apply other transforms
            if self.transforms:
                waveform = self.transforms(waveform)

            # Check silence after transforms (useful for random crops)
            if self.check_silence and is_silence(waveform):
                invalid_audio = True
                continue

            # Return with TinyTag ID3 object if specified
            if self.with_ID3:
                return waveform, tag

            return waveform

    def __len__(self) -> int:
        return len(self.wavs)
