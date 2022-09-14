import os
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset


class LibriSpeechDataset(Dataset):
    def __init__(
        self,
        root: str = "./data",
        with_info: bool = False,
        transforms: Optional[Callable] = None,
    ):
        self.with_info = with_info
        self.transforms = transforms

        from datasets import load_dataset

        self.dataset = load_dataset(
            "librispeech_asr",
            "clean",
            split="train.100",
            cache_dir=os.path.join(root, "librispeech_dataset"),
        )

    def __getitem__(
        self, idx: Union[Tensor, int]
    ) -> Union[Tensor, Tuple[Tensor, Dict]]:
        idx = idx.tolist() if torch.is_tensor(idx) else idx  # type: ignore
        data = self.dataset[idx]
        waveform = torch.tensor(data["audio"]["array"]).view(1, -1)
        info = dict(
            sample_rate=data["audio"]["sampling_rate"],
            text=data["text"],
            speaker_id=data["speaker_id"],
        )
        if self.transforms:
            waveform = self.transforms(waveform)
        return (waveform, info) if self.with_info else waveform

    def __len__(self) -> int:
        return len(self.dataset)
