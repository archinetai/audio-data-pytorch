import os
import shutil
from typing import Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import torchaudio
import yt_dlp
from torch.nn import functional as F
from tqdm import tqdm

from ..utils import camel_to_snake, exists
from .wav_dataset import WAVDataset


class YoutubeDataset(WAVDataset):
    def __init__(
        self,
        urls: Sequence[str],
        root: str = "./data",
        crop_length: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.root = root
        paths = self.download_and_process(urls, crop_length)
        super().__init__(path=paths, **kwargs)

    @property
    def data_path(self) -> str:
        return os.path.join(self.root, camel_to_snake(self.__class__.__name__))

    def download_and_process(
        self, urls: Sequence[str], crop_length: Optional[int]
    ) -> Sequence[str]:
        # Create folder if not existent
        os.makedirs(self.data_path, exist_ok=True)
        paths = []
        # Download audio tracks
        for url in urls:
            # Get folder for current url, and append to paths
            youtube_id = self.youtube_url_to_id(url)
            processed_path = self.get_processed_path(youtube_id, crop_length)
            paths.append(processed_path)
            # If already exists, continue without downloading and processing
            if os.path.isdir(processed_path):
                print(f"URL {url} aldready processed in {processed_path}")
                continue
            # Download song to data path
            file_path, youtube_id = self.download(url, youtube_id, processed_path)
            # Crop or copy song to folder
            self.process(file_path, processed_path, crop_length)
            # Remove file from data path
            os.remove(file_path)
        return paths

    def download(
        self, url: str, youtube_id: str, processed_path: str
    ) -> Tuple[str, str]:
        file_path = os.path.join(self.data_path, f"{youtube_id}.wav")
        uncropped_processed_path = self.get_processed_path(youtube_id)
        uncropped_file_path = self.get_processed_file_path(uncropped_processed_path, 0)

        # Download audio track if not existent
        if not os.path.isfile(uncropped_file_path):
            options = {
                "format": "bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                        "preferredquality": "192",
                    }
                ],
                "outtmpl": os.path.join(self.data_path, f"{youtube_id}.%(ext)s"),
            }
            with yt_dlp.YoutubeDL(options) as youtube_dl:
                youtube_dl.download([url])
        else:
            # Copy file to data path if exists uncropped
            print(f"Uncropped song with id {youtube_id} found, skipping download.")
            shutil.copy2(uncropped_file_path, file_path)

        return file_path, youtube_id

    def process(
        self, file_path: str, processed_path: str, crop_length: Optional[int]
    ) -> None:
        if not exists(crop_length):
            os.makedirs(processed_path, exist_ok=True)
            shutil.copy2(file_path, self.get_processed_file_path(processed_path, 0))
        else:
            print(f"Cropping track into chunks of {crop_length}s.")
            # Load audio file
            waveform, sample_rate = torchaudio.load(file_path)
            # Pad file with zeros so that we can divide it in chuncks
            length = waveform.shape[1]
            chunk_size = sample_rate * crop_length
            pad_length = chunk_size - length % chunk_size
            waveform_padded = F.pad(waveform, (0, pad_length), "constant", 0)
            # Chunk waveform
            waveforms = waveform_padded.chunk(
                (length + pad_length) // chunk_size, dim=1
            )
            # Make path after waveforms processed in case it crashes
            os.makedirs(processed_path, exist_ok=True)
            # Save crops if not new
            progress_bar = tqdm(waveforms)
            for idx, waveform in enumerate(progress_bar):
                progress_bar.set_description(f"Processed crop {idx}")
                crop_path = self.get_processed_file_path(processed_path, idx)
                if not os.path.isfile(crop_path):
                    torchaudio.save(
                        filepath=crop_path,
                        src=waveform,
                        sample_rate=sample_rate,
                        encoding="PCM_S",
                        bits_per_sample=16,
                    )

    def get_processed_path(
        self, youtube_id: str, crop_length: Optional[int] = None
    ) -> str:
        if exists(crop_length):
            return os.path.join(self.data_path, f"{youtube_id}_{crop_length}")
        else:
            return os.path.join(self.data_path, f"{youtube_id}")

    def get_processed_file_path(self, processed_path: str, idx: int):
        return os.path.join(processed_path, f"{idx}.wav")

    def youtube_url_to_id(self, url: str) -> str:
        url_data = urlparse(url)
        query = parse_qs(url_data.query)
        video_id = query["v"][0]
        return video_id
