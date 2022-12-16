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
        metadata_mapping_path: Optional[str] = None,
    ):
        self.paths = path if isinstance(path, (list, tuple)) else [path]
        self.wavs = TensorBackedImmutableStringArray(get_all_wav_filenames(self.paths, recursive=recursive))
        self.transforms = transforms
        self.sample_rate = sample_rate
        self.metadata_mapping_path = metadata_mapping_path
        self.mappings = {}

        if metadata_mapping_path:
            # Create or load genre/artist -> id mapping file.
            if os.path.isfile(metadata_mapping_path):
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
                        try:
                            tag = TinyTag.get(wav)
                        except:
                            print("broken file")
                            continue
                        artists = (tag.artist or '').replace(' w. ', ', ').replace(' vs. ', ', ').replace(' feat. ', ', ').replace(' featuring ', ', ').replace(' & ', ', ').replace(' ft. ', ', ').replace(' with ', ', ').split(', ')
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
        info = ""
        
        for i in range(100):
            try:
                info = torchaudio.info(self.wavs[idx])
                break
            except:
                print("broken file: ", self.wavs[idx])
                idx += 1
                pass

        
        

        if(self.transforms.random_crop_size > 0):
            info = torchaudio.info(self.wavs[idx])
            length = info.num_frames
            sample_rate = info.sample_rate
            
            ratio = math.ceil(sample_rate/self.sample_rate)
            
            crop_size = self.transforms.random_crop_size * ratio
            
            frame_offset = random.randint(0, max(length - crop_size, 0))
            waveform, sample_rate = torchaudio.load(filepath=self.wavs[idx], frame_offset=frame_offset, num_frames=crop_size)
            if len(waveform[0]) < crop_size:
                waveform = torch.nn.functional.pad(waveform, pad=(0, crop_size-len(waveform[0])), mode='constant', value=0)
            if sample_rate != 44100 or len(waveform[0]) != crop_size:
                print(self.wavs[idx])
                print(sample_rate)
                print("ratio", ratio)
                print("crop_size", crop_size)
                print("crop sample length", waveform.shape)
            
        else:
            waveform, sample_rate = torchaudio.load(self.wavs[idx])
            #print("NON-crop sample length", waveform.shape)

        if self.sample_rate and sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )(waveform)
        if sample_rate != 44100:
            print("resampled length", waveform.shape)

        if self.transforms:
            waveform = self.transforms(waveform)
        if sample_rate != 44100:
            print("transformed length", waveform.shape)

        if self.metadata_mapping_path:
            tag = TinyTag.get(self.wavs[idx])
            artists = (tag.artist or '').replace(' w. ', ', ').replace(' vs. ', ', ').replace(' feat. ', ', ').replace(' featuring ', ', ').replace(' & ', ', ').replace(' ft. ', ', ').replace(' with ', ', ').split(', ')
            genres = (tag.genre or '').split(', ')
            # Map artist/genre strings to their ids in mappings. Ignore if not found.
            artist_ids = np.array(list(filter(lambda item: item != -1, map(lambda artist: self.mappings['artists'][artist], artists))))
            genre_ids = np.array(list(filter(lambda item: item != -1,map(lambda genre: self.mappings['genres'][genre], genres))))
            artist_ids.resize(4)
            genre_ids.resize(4)
            return waveform, Tensor(np.array([artist_ids, genre_ids])).int()
        return waveform

    def __len__(self) -> int:
        return len(self.wavs)
