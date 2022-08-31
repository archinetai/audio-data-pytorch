import os
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CommonVoiceDataset(Dataset):
    def __init__(
        self,
        auth_token: str,
        version: int,
        sub_version: int = 0,
        root: str = "./data",
        languages: Sequence[str] = ["en"],
        with_sample_rate: bool = False,
        transforms: Optional[Callable] = None,
    ):
        self.with_sample_rate = with_sample_rate
        self.transforms = transforms

        from datasets import interleave_datasets, load_dataset

        self.dataset = interleave_datasets(
            [
                load_dataset(
                    f"mozilla-foundation/common_voice_{version}_{sub_version}",
                    language,
                    split="train",
                    cache_dir=os.path.join(root, "common_voice_dataset"),
                    use_auth_token=auth_token,
                )
                for language in languages
            ]
        )

    def __getitem__(
        self, idx: Union[Tensor, int]
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        idx = idx.tolist() if torch.is_tensor(idx) else idx  # type: ignore
        data = self.dataset[idx]
        waveform = torch.tensor(data["audio"]["array"]).view(1, -1)
        sample_rate = data["audio"]["sampling_rate"]

        if self.transforms:
            waveform = self.transforms(waveform)
        return (waveform, sample_rate) if self.with_sample_rate else waveform

    def __len__(self) -> int:
        return len(self.dataset)
