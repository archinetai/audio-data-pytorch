import glob
import os
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


def get_all_wav_filenames(paths: Sequence[str], recursive: bool) -> List[str]:
    ext = "**/*.wav" if recursive else "*.wav"
    filenames = []
    for path in paths:
        filenames.extend(glob.glob(os.path.join(path, ext), recursive=recursive))
    return filenames


class WAVDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Sequence[str]],
        recursive: bool = False,
        with_sample_rate: bool = False,
        transforms: Optional[Callable] = None,
    ):
        self.paths = path if isinstance(path, (list, tuple)) else [path]
        self.wavs = get_all_wav_filenames(self.paths, recursive=recursive)
        self.transforms = transforms
        self.with_sample_rate = with_sample_rate

    def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        waveform, sample_rate = torchaudio.load(self.wavs[idx])
        if self.transforms:
            waveform = self.transforms(waveform)
        return (waveform, sample_rate) if self.with_sample_rate else waveform

    def __len__(self) -> int:
        return len(self.wavs)
