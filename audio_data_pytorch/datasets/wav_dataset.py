import glob
import os
import json
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torchaudio
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
        metadata_mapping_path: Optional[str] = None,
    ):
        self.paths = path if isinstance(path, (list, tuple)) else [path]
        self.wavs = get_all_wav_filenames(self.paths, recursive=recursive)
        self.transforms = transforms
        self.sample_rate = sample_rate

        if metadata_mapping_path:
            # Create or load genre/artist -> id mapping file.
            with open(metadata_mapping_path, 'w+') as openfile:
                # Reading from json file
                self.mappings = json.load(openfile)
                if not self.mappings:
                    # Generate data -> number mappings for dataset
                    artist_id = 0
                    genre_id = 0
                    for wav in self.wavs:
                        tag = TinyTag.get(wav)
                        artists = (tag.artist or '').split(', ')
                        genres = (tag.genre or '').split(', ')
                        for artist in artists:
                            if not self.mappings['artists'][artist]:
                                self.mappings['artists'][artist] = artist_id
                                artist_id += 1
                        for genre in genres:
                            if not self.mappings['genres'][genre]:
                                self.mappings['genres'][genre] = genre_id
                                genre_id += 1
                    json.dump(self.mappings, openfile)
                    



    def __getitem__(
        self, idx: Union[Tensor, int]
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        idx = idx.tolist() if torch.is_tensor(idx) else idx  # type: ignore
        waveform, sample_rate = torchaudio.load(self.wavs[idx])

        if self.sample_rate and sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )(waveform)

        if self.transforms:
            waveform = self.transforms(waveform)

        if self.mappings:
            tag = TinyTag.get(self.wavs[idx])
            artists = (tag.artist or '').split(', ')
            genres = (tag.genre or '').split(', ')
            # Map artist/genre strings to their ids in mappings. Ignore if not found.
            artist_ids = filter(lambda item: item != -1, map(lambda artist: self.mappings['artists'][artist], artists))
            genre_ids = filter(lambda item: item != -1,map(lambda genre: self.mappings['genres'][genre], genres))
            return waveform, (artist_ids, genre_ids)
        return waveform

    def __len__(self) -> int:
        return len(self.wavs)
