
# Audio Data - PyTorch

A collection of useful audio datasets and transforms for PyTorch.

## Install

```bash
pip install audio-data-pytorch
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/audio-data-pytorch?style=flat&colorA=0f0f0f&colorB=0f0f0f)](https://pypi.org/project/audio-data-pytorch/)

## Datasets

### WAV Dataset

Load one or multiple folders of `.wav` files as dataset.

```py
from audio_data_pytorch import WAVDataset

dataset = WAVDataset(path=['my/path1', 'my/path2'])
```

#### Full API:
```py
WAVDataset(
    path: Union[str, Sequence[str]], # Path or list of paths from which to load files
    recursive: bool = False # Recursively load files from provided paths
    sample_rate: bool = False, # Specify sample rate to convert files to on read
    random_crop_size: int = None, # Load small portions of files randomly
    transforms: Optional[Callable] = None, # Transforms to apply to audio files
    check_silence: bool = True # Discards silent samples if true
)
```


### AudioWebDataset
A [`WebDataset`](https://webdataset.github.io/webdataset/) extension for audio data. Assumes that the `.tar` file comes with pairs of `.wav` (or `.flac`) and `.json` data.
```py
from audio_data_pytorch import AudioWebDataset

dataset = AudioWebDataset(
    urls='mywebdataset.tar'
)

waveform, info = next(iter(dataset))

print(waveform.shape) # torch.Size([2, 480000])
print(info.keys()) # dict_keys(['text'])
```

#### Full API:
```py
dataset = AudioWebDataset(
    urls: Union[str, Sequence[str]],
    shuffle: Optional[int] = None,
    batch_size: Optional[int] = None,
    transforms: Optional[Callable] = None,# Transforms to apply to audio files
    use_wav_processor: bool = False, # Set this to True if your tar files only use .wav
    crop_size: Optional[int] = None,
    max_crops: Optional[int] = None,
    **kwargs, # Forwarded to WebDataset class

)
```

### LJSpeech Dataset
An unsupervised dataset for LJSpeech with voice-only data.
```py
from audio_data_pytorch import LJSpeechDataset

dataset = LJSpeechDataset(root='./data')

dataset[0] # (1, 158621)
dataset[1] # (1, 153757)
```

#### Full API:
```py
LJSpeechDataset(
    root: str = "./data", # The root where the dataset will be downloaded
    transforms: Optional[Callable] = None, # Transforms to apply to audio files
)
```

### LibriSpeech Dataset
Wrapper for the [LibriSpeech](https://www.openslr.org/12) dataset (EN only). Requires `pip install datasets`. Note that this dataset requires several GBs of storage.

```py
from audio_data_pytorch import LibriSpeechDataset

dataset = LibriSpeechDataset(
    root="./data",
)

dataset[0] # (1, 222336)
```

#### Full API:
```py
LibriSpeechDataset(
    root: str = "./data", # The root where the dataset will be downloaded
    with_info: bool = False, # Whether to return info (i.e. text, sampling rate, speaker_id)
    transforms: Optional[Callable] = None, # Transforms to apply to audio files
)
```

### Common Voice Dataset
Multilanguage wrapper for the [Common Voice](https://commonvoice.mozilla.org/). Requires `pip install datasets`. Note that each language requires several GBs of storage, and that you have to confirm access for each distinct version you use e.g. [here](https://huggingface.co/datasets/mozilla-foundation/common_voice_10_0), to validate your Huggingface access token. You can provide a list of `languages` and to avoid an unbalanced dataset the values will be interleaved by downsampling the majority language to have the same number of samples as the minority language.

```py
from audio_data_pytorch import CommonVoiceDataset

dataset = CommonVoiceDataset(
    auth_token="hf_xxx",
    version=1,
    root="../data",
    languages=['it']
)
```

#### Full API:
```py
CommonVoiceDataset(
    auth_token: str, # Your Huggingface access token
    version: int, # Common Voice dataset version
    sub_version: int = 0, # Subversion: common_voice_{version}_{sub_version}
    root: str = "./data", # The root where the dataset will be downloaded
    languages: Sequence[str] = ['en'], # List of languages to include in the dataset
    with_info: bool = False,  #  Whether to return info (i.e. text, sampling rate, age, gender, accent, locale)
    transforms: Optional[Callable] = None, # Transforms to apply to audio files
)
```

### Youtube Dataset
A wrapper around yt-dlp that automatically downloads the audio source of Youtube videos. Requires `pip install yt-dlp`.

```py
from audio_data_pytorch import YoutubeDataset

dataset = YoutubeDataset(
    root='./data',
    urls=[
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=BZ-_KQezKmU",
    ],
    crop_length=10 # Crop source in 10s chunks (optional but suggested)
)
dataset[0] # (2, 480000)
```

#### Full API:
```py
dataset = YoutubeDataset(
    urls: Sequence[str], # The list of youtube urls
    root: str = "./data", # The root where the dataset will be downloaded
    crop_length: Optional[int] = None, # Crops the source into chunks of `crop_length` seconds
    with_sample_rate: bool = False, # Returns sample rate as second argument
    transforms: Optional[Callable] = None, # Transforms to apply to audio files
)
```

### Clotho Dataset
A wrapper for the [Clotho](https://zenodo.org/record/3490684#.Y0VVVOxBwR0) dataset extending `AudioWebDataset`. Requires `pip install py7zr` to decompress `.7z` archive.

```py
from audio_data_pytorch import ClothoDataset, Crop, Stereo, Mono

dataset = ClothoDataset(
    root='./data/',
    preprocess_sample_rate=48000, # Added to all files during preprocessing
    preprocess_transforms=nn.Sequential(Crop(48000*10), Stereo()), # Added to all files during preprocessing
    transforms=Mono() # Added dynamically at iteration time
)
```


#### Full API:
```py
dataset = ClothoDataset(
    root: str, # Path where the dataset is saved
    split: str = 'train', # Dataset split, one of: 'train', 'valid'
    preprocess_sample_rate: Optional[int] = None, # Preprocesses dataset to this sample rate
    preprocess_transforms: Optional[Callable] = None, # Preprocesses dataset with the provided transfomrs
    reset: bool = False, # Re-compute preprocessing if `true`
    **kwargs # Forwarded to `AudioWebDataset`
)
```

### MetaDataset
Extends `WAVDataset` with artist and genres read from ID3 tags and returned as string arrays or optionally mapped to integers stored in a json file at `metadata_mapping_path`.


```py
from audio_data_pytorch import MetaDataset

dataset = MetaDataset(
    path: Union[str, Sequence[str]], # Path or list of paths from which to load files
    metadata_mapping_path: Optional[str] = None, # Path where mapping from artist/genres to numbers will be saved
)

waveform, artists, genres = next(iter(dataset))

# Convert an artist ID back to a string
artist_name = dataset.mappings['artists'].invert[insert_artist_id]

# Convert a genre ID back to a string
genre_name = dataset.mappings['genres'].invert[insert_genre_id]

# If given a metadata_mapping_path, metadata is returned as an int Tensor
waveform, artist_genre_tensor = next(iter(dataset))
```


#### Full API:
```py
dataset = MetaDataset(
    path: Union[str, Sequence[str]], # Path or list of paths from which to load files
    metadata_mapping_path: Optional[str] = None, # Path where mapping from artist/genres to numbers will be saved
    max_artists: int = 4, # Max number of artists to return
    max_genres: int = 4, # Max number of artists to return
    **kwargs # Forwarded to `WAVDataset`
)
```


## Transforms

You can use the following individual transforms, or merge them with `nn.Sequential()`:

```py
from audio_data_pytorch import Crop
crop = Crop(size=22050*2, start=0) # Crop 2 seconds at 22050 Hz from the start of the file

from audio_data_pytorch import RandomCrop
random_crop = RandomCrop(size=22050*2) # Crop 2 seconds at 22050 Hz from a random position

from audio_data_pytorch import Resample
resample = Resample(source=48000, target=22050), # Resamples from 48kHz to 22kHz

from audio_data_pytorch import Mono
overlap = Mono() # Overap channels by sum to get mono soruce (C, N) -> (1, N)

from audio_data_pytorch import Stereo
stereo = Stereo() # Duplicate channels (1, N) -> (2, N) or (2, N) -> (2, N)

from audio_data_pytorch import Scale
scale = Scale(scale=0.8) # Scale waveform amplitude by 0.8

from audio_data_pytorch import Loudness
loudness = Loudness(sampling_rate=22050, target=-20) # Normalize loudness to -20dB, requires `pip install pyloudnorm`
```

Or use this wrapper to apply a subset of them in one go, API:
```py
from audio_data_pytorch import AllTransform

transform = AllTransform(
    source_rate: Optional[int] = None,
    target_rate: Optional[int] = None,
    crop_size: Optional[int] = None,
    random_crop_size: Optional[int] = None,
    loudness: Optional[int] = None,
    scale: Optional[float] = None,
    mono: bool = False,
    stereo: bool = False,
)
```
