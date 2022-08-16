import os
import tarfile

import requests  # type: ignore
from tqdm import tqdm

from ..utils import camel_to_snake
from .wav_dataset import WAVDataset


class LJSpeechDataset(WAVDataset):

    data_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    data_tar_file = "LJSpeech-1.1.tar.bz2"
    data_waws_path = "LJSpeech-1.1/wavs"

    def __init__(self, root: str = "./data", **kwargs) -> None:
        self.root = root

        if not os.path.exists(self.data_path):
            print(
                f"Data not found in {self.data_path}, downloading {self.data_tar_file}"
            )
            self.download()

        super().__init__(path=self.wavs_path, **kwargs)

    @property
    def data_path(self) -> str:
        return os.path.join(self.root, camel_to_snake(self.__class__.__name__))

    @property
    def file_path(self) -> str:
        return os.path.join(self.data_path, self.data_tar_file)

    @property
    def wavs_path(self) -> str:
        return os.path.join(self.data_path, self.data_waws_path)

    def download(self) -> None:
        os.makedirs(self.data_path, exist_ok=True)
        response = requests.get(self.data_url, stream=True)
        block_size = 1024  # Kibibyte
        progress_bar = tqdm(total=block_size, unit="iB", unit_scale=True)

        with open(self.file_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        self.decompress()

    def decompress(self) -> None:
        print(f"Decompressing {self.data_tar_file} to {self.data_path}")
        file = tarfile.open(self.file_path)
        file.extractall(self.data_path)
        file.close()
