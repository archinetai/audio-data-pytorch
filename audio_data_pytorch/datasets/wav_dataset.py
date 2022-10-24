import random
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torchaudio
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
        check_silence: bool = True,
    ):
        self.paths = path if isinstance(path, (list, tuple)) else [path]
        self.wavs = get_all_wav_filenames(self.paths, recursive=recursive)
        self.transforms = transforms
        self.sample_rate = sample_rate
        self.check_silence = check_silence

    def __getitem__(
        self, idx: Union[Tensor, int]
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        idx = idx.tolist() if torch.is_tensor(idx) else idx  # type: ignore
        invalid_audio = False

        # Check that we can load audio properly
        try:
            waveform, sample_rate = torchaudio.load(self.wavs[idx])
        except Exception:
            invalid_audio = True

        # Check that the sample is not silent
        if not invalid_audio and self.check_silence and is_silence(waveform):
            invalid_audio = True

        # Get new sample if audio is invalid
        if invalid_audio:
            return self[random.randrange(len(self))]

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
            return self[random.randrange(len(self))]

        return waveform

    def __len__(self) -> int:
        return len(self.wavs)
