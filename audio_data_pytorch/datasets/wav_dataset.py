import random
import math
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import torchaudio
import random
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
        optimized_random_crop_size: int = None,
        check_silence: bool = True,
        with_idx: bool = False,
    ):
        self.paths = path if isinstance(path, (list, tuple)) else [path]
        self.wavs = get_all_wav_filenames(self.paths, recursive=recursive)
        self.transforms = transforms
        self.sample_rate = sample_rate
        self.check_silence = check_silence
        self.with_idx = with_idx
        self.optimized_random_crop_size = optimized_random_crop_size
        assert (not optimized_random_crop_size or sample_rate), "Optimized random cropr requires sample_rate to be set."

    # Instead of loading the whole file and chopping out our crop, we only load what we need.
    def optimized_random_crop(self, idx):
        # Get length/audio info
        info = torchaudio.info(self.wavs[idx])
        length = info.num_frames
        sample_rate = info.sample_rate

        # Calculate correct number of samples to read based on actual and intended sample rate
        ratio = math.ceil(sample_rate/self.sample_rate)
        crop_size = self.optimized_random_crop_size * ratio
        frame_offset = random.randint(0, max(length - crop_size, 0))

        # Load the samples
        waveform, sample_rate = torchaudio.load(
            filepath=self.wavs[idx], frame_offset=frame_offset, num_frames=crop_size)

        # Pad with zeroes if the sizes aren't quite right (e.g., rates aren't exact multiples)
        if len(waveform[0]) < crop_size:
            waveform = torch.nn.functional.pad(waveform, pad=(
                0, crop_size-len(waveform[0])), mode='constant', value=0)

        return waveform, sample_rate

    def __getitem__(
        self, idx: Union[Tensor, int]
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        idx = idx.tolist() if torch.is_tensor(idx) else idx  # type: ignore
        invalid_audio = False

        # Loop until we find a valid audio sample
        while (True):
            # If last sample was invalid, use a new random one.
            if invalid_audio:
                idx = random.randrange(len(self))

            if (self.optimized_random_crop_size > 0):
                waveform, sample_rate = self.optimized_random_crop(idx)
            else:
                # If no crop, just load everything
                try:
                    waveform, sample_rate = self.optimized_random_crop(idx)
                except Exception:
                    invalid_audio = True
                    continue

            # Apply sample rate transform if necessary
            if self.sample_rate and sample_rate != self.sample_rate:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.sample_rate
                )(waveform)

            # Apply other transforms
            if self.transforms:
                waveform = self.transforms(waveform)

            # Check silence after transforms (useful for random crops)
            if self.check_silence and is_silence(waveform):
                invalid_audio = True
                continue

            if self.with_idx:
                return waveform, idx
            # Otherwise, return sample without metadata
            return waveform

    def __len__(self) -> int:
        return len(self.wavs)
