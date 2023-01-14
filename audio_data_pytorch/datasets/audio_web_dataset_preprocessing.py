import glob
import json
import os
import re
import tarfile
from typing import Callable, Dict, List, Optional, Sequence, Union

import torchaudio
from torch import nn
from tqdm import tqdm

from ..transforms import Crop, Loudness, Mono, Resample, Stereo
from ..utils import Decompressor, Downloader, exists, run_async

"""
Preprocessing
"""


class AudioProcess:
    def __init__(
        self,
        path: str,
        info: Dict,
        sample_rate: Optional[int] = None,
        transforms: Optional[Callable] = None,
    ):
        self.path = path
        self.sample_rate = sample_rate
        self.transforms = transforms
        self.info = info
        self.path_prefix = f"{os.path.splitext(self.path)[0]}_processed"
        self.wav_dest_path = None
        self.json_dest_path = None

    def process_wav(self):
        waveform, rate = torchaudio.load(self.path)

        if exists(self.sample_rate):
            resample = Resample(source=rate, target=self.sample_rate)
            waveform = resample(waveform)
            rate = self.sample_rate

        if exists(self.transforms):
            waveform = self.transforms(waveform)

        wav_dest_path = f"{self.path_prefix}.wav"
        torchaudio.save(wav_dest_path, waveform, rate)

        self.wav_dest_path = wav_dest_path
        return wav_dest_path

    def process_info(self):
        json_dest_path = f"{self.path_prefix}.json"
        with open(json_dest_path, "w") as f:
            json.dump(self.info, f)

        self.json_dest_path = json_dest_path
        return json_dest_path

    def __enter__(self):
        wav_processed_path = self.process_wav()
        json_processed_path = self.process_info()
        return wav_processed_path, json_processed_path

    def __exit__(self, *args):
        os.remove(self.wav_dest_path)
        os.remove(self.json_dest_path)


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
            async with Decompressor(files, path=path, remove_on_exit=True) as folders:
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
