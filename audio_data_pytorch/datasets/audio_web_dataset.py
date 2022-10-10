import glob
import json
import os
import re
import tarfile
from typing import Callable, List, Optional, Sequence, Union

import torchaudio
from torch import nn
from tqdm import tqdm
from webdataset import WebDataset, torch_audio

from ..transforms import Crop, Loudness, Mono, Resample, Stereo
from ..utils import Decompressor, Downloader, exists, run_async

"""
Preprocessing
"""


class AudioWebDatasetPreprocess:
    def __init__(
        self,
        name: str,
        root: Union[str, Sequence[str]],
        urls: List[str],
        crop_length: float,
        sample_rate: int = 48000,
        stereo: bool = True,
        mono: bool = False,
        loudness: Optional[int] = None,
    ):
        self.name = name
        self.root = root
        self.urls = urls
        self.sample_rate = sample_rate
        self.transform = nn.Sequential(
            Crop(int(crop_length * sample_rate)),
            Mono() if mono else nn.Identity(),
            Stereo() if stereo else nn.Identity(),
            Loudness(sampling_rate=sample_rate, target=loudness)
            if exists(loudness)
            else nn.Identity(),
        )

        run_async(self.preprocess())

    def str_to_tags(self, str: str) -> List[str]:
        return re.split(r"\s*[.,;_/]+\s*|\s+[-]+\s+", str)

    async def preprocess(self):
        urls, path = self.urls, self.root
        tarfile_name = os.path.join(path, f"{self.name}.tar")
        waveform_id = 0

        async with Downloader(urls, path=path) as files:
            async with Decompressor(files, path=path) as folders:
                with tarfile.open(tarfile_name, "w") as archive:
                    for folder in tqdm(folders):
                        for wav in tqdm(glob.glob(folder + "/**/*.wav")):
                            waveform, rate = torchaudio.load(wav)
                            resample = Resample(source=rate, target=self.sample_rate)
                            waveform = self.transform(resample(waveform))
                            txt = os.path.splitext(os.path.relpath(wav, folder))[0]
                            info = dict(
                                tags=self.str_to_tags(txt), sample_rate=self.sample_rate
                            )

                            file_name = f"{waveform_id:06d}"
                            wav_name = f"{file_name}.wav"
                            json_name = f"{file_name}.json"

                            wav_path = os.path.join(path, wav_name)
                            json_path = os.path.join(path, json_name)

                            torchaudio.save(wav_path, waveform, self.sample_rate)
                            with open(json_path, "w") as f:
                                json.dump(info, f)

                            archive.add(wav_path, arcname=wav_name)
                            archive.add(json_path, arcname=json_name)

                            os.remove(wav_path)
                            os.remove(json_path)

                            waveform_id += 1


"""
Dataset
"""


def get_all_tar_filenames(paths: Sequence[str], recursive: bool) -> List[str]:
    extensions = ["tar", "tar.gz"]
    filenames = []
    for ext_name in extensions:
        ext = f"**/*.{ext_name}" if recursive else f"*.{ext_name}"
        for path in paths:
            filenames.extend(glob.glob(os.path.join(path, ext), recursive=recursive))
    return filenames


def identity(x):
    return x


class AudioWebDataset(WebDataset):

    # Why batch_size in a dataset constructor?
    # https://webdataset.github.io/webdataset/gettingstarted/#webdataset-and-dataloader

    def __init__(
        self,
        path: Union[str, Sequence[str]],
        transforms: Optional[Callable] = None,
        batch_size: Optional[int] = None,
        recursive: bool = True,
        shuffle: int = 128,
        **kwargs,
    ):
        paths = path if isinstance(path, (list, tuple)) else [path]
        tars = get_all_tar_filenames(paths, recursive=recursive)
        super().__init__(urls=tars, **kwargs)

        (
            self.shuffle(shuffle)
            .decode(torch_audio)
            .to_tuple("wav", "json")
            .map_tuple((lambda tuple: tuple[0]), identity)
        )

        if exists(transforms):
            self.map_tuple(transforms, identity)

        if exists(batch_size):
            self.batched(batch_size)
