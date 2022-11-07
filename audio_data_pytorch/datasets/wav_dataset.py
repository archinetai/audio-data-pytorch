import glob
import os
import json
from json.decoder import JSONDecodeError
from typing import Callable, List, Optional, Sequence, Tuple, Union
import math

import torch
import torchaudio
import random
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from tinytag import TinyTag


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
        source_rate: Optional[int] = None,
        metadata_mapping_path: Optional[str] = None,
    ):
        self.paths = path if isinstance(path, (list, tuple)) else [path]
        self.wavs = get_all_wav_filenames(self.paths, recursive=recursive)
        self.transforms = transforms
        self.sample_rate = sample_rate
        self.source_rate = source_rate
        self.metadata_mapping_path = metadata_mapping_path
        self.mappings = {}

        if metadata_mapping_path:
            # Create or load genre/artist -> id mapping file.
            with open(metadata_mapping_path, 'r') as openfile:
                # Reading from json file
                try:
                    self.mappings = json.load(openfile)
                    print("Mappings loaded.")
                    print("Artists:", len(self.mappings['artists']))
                    print("Genres:", len(self.mappings['genres']))
                except JSONDecodeError as e:
                    print("No mappings found on disk:", e)
            if self.mappings == {}:
                with open(metadata_mapping_path, 'w') as openfile:
                    print("Generating mappings")
                    # Generate data -> number mappings for dataset
                    artist_id = 1
                    genre_id = 1
                    for wav in self.wavs:
                        tag = TinyTag.get(wav)
                        artists = (tag.artist or '').split(', ')
                        genres = (tag.genre or '').split(', ')
                        for artist in artists:
                            # cringe
                            if not ('artists' in self.mappings and artist in self.mappings['artists']):
                                self.mappings.setdefault('artists', {}).setdefault(artist, artist_id)
                                artist_id += 1
                        for genre in genres:
                            if not ('genres' in self.mappings and genre in self.mappings['genres']):
                                self.mappings.setdefault('genres', {}).setdefault(genre, genre_id)
                                genre_id += 1
                    json.dump(self.mappings, openfile)
                    



    def __getitem__(
        self, idx: Union[Tensor, int]
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        idx = idx.tolist() if torch.is_tensor(idx) else idx  # type: ignore
        waveform, sample_rate = (0, 0)

        source_rate = self.source_rate
        target_rate = self.transforms.target_rate

        ratio = math.ceil(source_rate/target_rate)
        crop_size = self.transforms.random_crop_size * ratio

        if(self.transforms.random_crop_size > 0):
            length = torchaudio.info(self.wavs[idx]).num_frames
            frame_offset = random.randint(0, max(length - crop_size, 0))
            waveform, sample_rate = torchaudio.load(filepath=self.wavs[idx], frame_offset=frame_offset, num_frames=crop_size)
        else:
            waveform, sample_rate = torchaudio.load(self.wavs[idx])

        if self.sample_rate and sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )(waveform)

        if self.transforms:
            waveform = self.transforms(waveform)

        if self.metadata_mapping_path:
            tag = TinyTag.get(self.wavs[idx])
            artists = (tag.artist or '').split(', ')
            genres = (tag.genre or '').split(', ')
            # Map artist/genre strings to their ids in mappings. Ignore if not found.
            artist_ids = np.array(list(filter(lambda item: item != -1, map(lambda artist: self.mappings['artists'][artist], artists))))
            genre_ids = np.array(list(filter(lambda item: item != -1,map(lambda genre: self.mappings['genres'][genre], genres))))
            artist_ids.resize(4)
            genre_ids.resize(4)
            return waveform, Tensor([artist_ids, genre_ids]).int()
        return waveform

    def __len__(self) -> int:
        return len(self.wavs)
