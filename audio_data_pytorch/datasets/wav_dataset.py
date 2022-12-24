import glob
import os
import random
import math
import json
from json.decoder import JSONDecodeError
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import torchaudio
import random
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from tinytag import TinyTag
from ..utils import fast_scandir, is_silence, split_artists


def get_all_wav_filenames(paths: Sequence[str], recursive: bool) -> List[str]:
    extensions = [".wav", ".flac"]
    filenames = []
    for path in paths:
        _, files = fast_scandir(path, extensions, recursive=recursive)
        filenames.extend(files)
    return filenames


class TensorBackedImmutableStringArray:
    def __init__(self, strings, encoding = 'utf-8'):
        encoded = [torch.ByteTensor(torch.ByteStorage.from_buffer(s.encode(encoding))) for s in strings]
        self.cumlen = torch.cat((torch.zeros(1, dtype = torch.int64), torch.as_tensor(list(map(len, encoded)), dtype = torch.int64).cumsum(dim = 0)))
        self.data = torch.cat(encoded)
        self.encoding = encoding

    def __getitem__(self, i):
        return bytes(self.data[self.cumlen[i] : self.cumlen[i + 1]]).decode(self.encoding)

    def __len__(self):
        return len(self.cumlen) - 1

    def __list__(self):
        return [self[i] for i in range(len(self))]

class WAVDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Sequence[str]],
        recursive: bool = False,
        transforms: Optional[Callable] = None,
        sample_rate: Optional[int] = None,
        check_silence: bool = True,
        metadata_mapping_path: Optional[str] = None,
        max_artists: int = 4,
        max_genres: int = 4
    ):
        self.paths = path if isinstance(path, (list, tuple)) else [path]
        self.wavs = TensorBackedImmutableStringArray(get_all_wav_filenames(self.paths, recursive=recursive))
        self.transforms = transforms
        self.sample_rate = sample_rate
        self.check_silence = check_silence
        self.metadata_mapping_path = metadata_mapping_path
        self.max_artists = max_artists
        self.max_genres = max_genres
        self.mappings = {}

        if metadata_mapping_path:
            # Create or load genre/artist -> id mapping file.
            if os.path.isfile(metadata_mapping_path):
                with open(metadata_mapping_path, 'r') as openfile:
                    # Reading from json file
                    try:
                        self.mappings = json.load(openfile)
                        print("Mappings loaded.")
                    except JSONDecodeError as e:
                        print("Found invalid mapping file:", e)
            if self.mappings == {}:
                with open(metadata_mapping_path, 'w') as openfile:
                    print("Generating mappings")
                    # Generate data -> number mappings for dataset
                    artist_id = 1
                    genre_id = 1
                    for wav in self.wavs:
                        # Try to get ID3 tags via TinyTag
                        try:
                            tag = TinyTag.get(wav)
                        except:
                            print("broken file")
                            continue
                        artists = split_artists(tag.artist or '')
                        genres = (tag.genre or '').split(', ')

                        # Assign ids to new genres
                        for artist in artists:
                            if not ('artists' in self.mappings and artist in self.mappings['artists']):
                                self.mappings.setdefault('artists', {}).setdefault(artist, artist_id)
                                artist_id += 1
                        # Assign ids to new genres
                        for genre in genres:
                            if not ('genres' in self.mappings and genre in self.mappings['genres']):
                                self.mappings.setdefault('genres', {}).setdefault(genre, genre_id)
                                genre_id += 1
                    json.dump(self.mappings, openfile)
        print("Artists:", len(self.mappings['artists']))
        print("Genres:", len(self.mappings['genres']))

    def __getitem__(
        self, idx: Union[Tensor, int]
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        idx = idx.tolist() if torch.is_tensor(idx) else idx  # type: ignore
        invalid_audio = False

        # Loop until we find a valid audio sample
        while(True):
            # If last sample was invalid, use a new random one.
            if invalid_audio:
                idx = random.randrange(len(self))            
            
            # Instead of loading the whole file before crop, only load crop
            if(self.transforms.random_crop_size > 0):
                # Get length/audio info
                try:
                    info = torchaudio.info(self.wavs[idx])
                except Exception:
                    invalid_audio = True
                    continue
                length = info.num_frames
                sample_rate = info.sample_rate
                
                # Calculate correct number of samples to read based on actual and intended sample rate
                ratio = math.ceil(sample_rate/self.sample_rate)
                crop_size = self.transforms.random_crop_size * ratio
                frame_offset = random.randint(0, max(length - crop_size, 0))

                # Load the samples
                try:
                    waveform, sample_rate = torchaudio.load(filepath=self.wavs[idx], frame_offset=frame_offset, num_frames=crop_size)
                except Exception:
                    print("Unable to load sample... but was able to load info.")
                    invalid_audio = True
                    continue

                # Pad with zeroes if the sizes aren't quite right (e.g., rates aren't exact multiples)
                if len(waveform[0]) < crop_size:
                    waveform = torch.nn.functional.pad(waveform, pad=(0, crop_size-len(waveform[0])), mode='constant', value=0)
            else:
                # If no crop, just load everything
                waveform, sample_rate = torchaudio.load(self.wavs[idx])

            # Apply sample rate transform if necessary
            if self.sample_rate and sample_rate != self.sample_rate:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.sample_rate
                )(waveform)

            # Apply other transforms
            if self.transforms:
                waveform = self.transforms(waveform)

            # Check silence after transforms (useful for random crops)
            if self.check_silence and is_silence(waveform):
                invalid_audio = True
                continue

            # Return tuple with genre and artist ID3 tags if necessary
            if self.metadata_mapping_path:
                tag = TinyTag.get(self.wavs[idx])
                artists = split_artists(tag.artist or '')
                genres = (tag.genre or '').split(', ')
                # Map artist/genre strings to their ids in mappings. Ignore if not found.
                artist_ids = np.array(list(filter(lambda item: item != -1, map(lambda artist: self.mappings['artists'][artist], artists))))
                genre_ids = np.array(list(filter(lambda item: item != -1,map(lambda genre: self.mappings['genres'][genre], genres))))
                # Resize to requested sizes
                artist_ids.resize(self.max_artists)
                genre_ids.resize(self.max_genres)
                return waveform, Tensor(np.array([artist_ids, genre_ids])).int()

            # Otherwise, return sample without metadata
            return waveform

    def __len__(self) -> int:
        return len(self.wavs)
