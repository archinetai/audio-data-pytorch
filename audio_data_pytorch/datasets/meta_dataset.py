import os
import json
from json.decoder import JSONDecodeError
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import numpy as np
from torch import Tensor
from .wav_dataset import WAVDataset
from tinytag import TinyTag
from ..utils import split_artists
from bidict import bidict



class MetaDataset(WAVDataset):
    def __init__(
        self,
        metadata_mapping_path: Optional[str] = None,
        max_artists: int = 4,
        max_genres: int = 4,
        **kwargs
    ):
        self.max_artists = max_artists
        self.max_genres = max_genres
        self.metadata_mapping_path = metadata_mapping_path

        super().__init__(with_idx=True, **kwargs)

        if metadata_mapping_path:
            # Create or load genre/artist -> id mapping file.
            if os.path.isfile(metadata_mapping_path):
                with open(metadata_mapping_path, 'r') as openfile:
                    # Reading from json file
                    try:
                        mappings = json.load(openfile)
                        self.mappings = {'artists' : bidict(mappings['artists']), 'genres': bidict(mappings['genres'])}
                        print("Mappings loaded.")
                    except JSONDecodeError as e:
                        print("Found invalid mapping file:", e)
            if self.mappings == {}:
                self.mappings = self.generate_mappings(metadata_mapping_path)
            print("Artists:", len(self.mappings['artists']))
            print("Genres:", len(self.mappings['genres']))

        

    def generate_mappings(self, metadata_mapping_path):
        mappings = {'artists': bidict(), 'genres': bidict()}
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
                # Create artist/genre arrays from ID3 artist/genre strings.
                artists = split_artists(tag.artist or '')
                genres = (tag.genre or '').split(', ')

                # Assign ids to new genres
                for artist in artists:
                    if not ('artists' in mappings and artist in mappings['artists']):
                        mappings.setdefault('artists', {}).setdefault(artist, artist_id)
                        artist_id += 1
                # Assign ids to new genres
                for genre in genres:
                    if not ('genres' in mappings and genre in mappings['genres']):
                        mappings.setdefault('genres', {}).setdefault(genre, genre_id)
                        genre_id += 1
            # Save newly generated mappings to disk
            json.dump(mappings, openfile)
        return mappings

    def __getitem__(
        self, idx: Union[Tensor, int]
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # Get waveform
        waveform, idx = super().__getitem__(idx)
        # Get ID3 data
        tag = TinyTag.get(self.wavs[idx])
        # Split artists by separators like "feat"
        artists = split_artists(tag.artist or '')
        # Split genres by ","
        genres = (tag.genre or '').split(', ')
        if self.metadata_mapping_path:
            # Map artist/genre strings to their ids in mappings. Ignore if not found.
            artist_ids = np.array(list(filter(lambda item: item != -1, map(lambda artist: self.mappings['artists'][artist], artists))))
            genre_ids = np.array(list(filter(lambda item: item != -1,map(lambda genre: self.mappings['genres'][genre], genres))))
            # Resize to requested sizes
            artist_ids.resize(self.max_artists)
            genre_ids.resize(self.max_genres)
            return waveform, torch.from_numpy(np.array([artist_ids, genre_ids])).int()
        else:
            return waveform, artists, genres

    def __len__(self) -> int:
        return len(self.wavs)
