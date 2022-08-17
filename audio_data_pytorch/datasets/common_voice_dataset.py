from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from datasets import interleave_datasets, load_dataset
from torch import Tensor
from torch.utils.data import Dataset


class CommonVoiceDataset(Dataset):
    def __init__(
        self,
        root: str = "./data",
        languages: Sequence[str] = ["en"],
        with_sample_rate: bool = False,
        transforms: Optional[Callable] = None,
    ):
        self.root = root
        self.with_sample_rate = with_sample_rate
        self.transforms = transforms

        self.dataset = interleave_datasets(
            [
                load_dataset("common_voice", language, split="train", cache_dir=root)
                for language in languages
            ]
        )

    def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        data = self.dataset[idx]
        waveform = torch.tensor(data["audio"]["array"]).view(1, -1)
        sample_rate = data["audio"]["sampling_rate"]

        if self.transforms:
            waveform = self.transforms(waveform)
        return (waveform, sample_rate) if self.with_sample_rate else waveform

    def __len__(self) -> int:
        return len(self.dataset)
