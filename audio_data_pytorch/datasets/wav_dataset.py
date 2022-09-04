import glob
import os
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


def get_all_wav_filenames(paths: Sequence[str], recursive: bool) -> List[str]:
    extensions = ["wav", "flac"]
    filenames = []
    for ext_name in extensions:
        ext = f"**/*.{ext_name}" if recursive else f"*.{ext_name}"
        for path in paths:
            filenames.extend(glob.glob(os.path.join(path, ext), recursive=recursive))
    return filenames


class WAVDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Sequence[str]],
        recursive: bool = False,
        transforms: Optional[Callable] = None,
        sample_rate: Optional[int] = None,
    ):
        self.paths = path if isinstance(path, (list, tuple)) else [path]
        self.wavs = get_all_wav_filenames(self.paths, recursive=recursive)
        self.transforms = transforms
        self.sample_rate = sample_rate

    def __getitem__(
        self, idx: Union[Tensor, int]
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        idx = idx.tolist() if torch.is_tensor(idx) else idx  # type: ignore
        waveform, sample_rate = torchaudio.load(self.wavs[idx])
        if self.transforms:
            waveform = self.transforms(waveform)

        if self.sample_rate and sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )(waveform)

        return waveform

    def __len__(self) -> int:
        return len(self.wavs)
