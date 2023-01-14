import os
import tarfile
from typing import Callable, List, Optional

import pandas as pd
from tqdm import tqdm

from ..utils import Decompressor, Downloader, camel_to_snake, run_async
from .audio_web_dataset import AudioWebDataset
from .audio_web_dataset_preprocessing import AudioProcess


class ClothoDataset(AudioWebDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        preprocess_sample_rate: Optional[int] = None,
        preprocess_transforms: Optional[Callable] = None,
        reset: bool = False,
        **kwargs,
    ):
        self.root = root
        self.split = self.split_conversion(split)
        self.preprocess_sample_rate = preprocess_sample_rate
        self.preprocess_transforms = preprocess_transforms

        if not os.path.exists(self.tar_file_name) or reset:
            run_async(self.preprocess())

        super().__init__(urls=self.tar_file_name, **kwargs)

    def split_conversion(self, split: str) -> str:
        return {"train": "development", "valid": "evaluation"}[split]

    @property
    def urls(self) -> List[str]:
        return [
            f"https://zenodo.org/record/4783391/files/clotho_audio_{self.split}.7z",
            f"https://zenodo.org/record/4783391/files/clotho_captions_{self.split}.csv",
        ]

    @property
    def data_path(self) -> str:
        return os.path.join(self.root, camel_to_snake(self.__class__.__name__))

    @property
    def tar_file_name(self) -> str:
        return os.path.join(self.data_path, f"clotho_{self.split}.tar")

    async def preprocess(self):
        urls, path = self.urls, self.data_path
        waveform_id = 0

        async with Downloader(urls, path=path) as files:
            to_decompress = [f for f in files if f.endswith(".7z")]
            caption_csv_file = [f for f in files if f.endswith(".csv")][0]
            async with Decompressor(
                to_decompress, path=path, remove_on_exit=True
            ) as folders:
                captions = pd.read_csv(caption_csv_file)
                length = len(captions.index)

                with tarfile.open(self.tar_file_name, "w") as archive:
                    for i, caption in tqdm(captions.iterrows(), total=length):
                        wav_file_name = caption.file_name
                        wav_path = os.path.join(folders[0], self.split, wav_file_name)
                        wav_captions = [caption[f"caption_{i}"] for i in range(1, 6)]
                        info = dict(text=wav_captions)

                        with AudioProcess(
                            path=wav_path,
                            sample_rate=self.preprocess_sample_rate,
                            transforms=self.preprocess_transforms,
                            info=info,
                        ) as (wav, json):
                            archive.add(wav, arcname=f"{waveform_id:06d}.wav")
                            archive.add(json, arcname=f"{waveform_id:06d}.json")

                        waveform_id += 1
