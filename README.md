
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
    with_sample_rate: bool = False, # Returns sample rate as second argument
    transforms: Optional[Callable] = None, # Transforms to apply to audio files
)
```

### LJSpeech Dataset
An unsupervised dataset for LJSpeech with voice only data
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
    with_sample_rate: bool = False, # Returns sample rate as second argument
    transforms: Optional[Callable] = None, # Transforms to apply to audio files
)
```

### Youtube Dataset
A wrapper around yt-dlp that automatically downloads the audio source of Youtube videos.

```py
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


## Transforms

An example

```py

crop = Crop(22050) # Crop start of audio track

transforms = nn.Sequential(
    Resample(source=48000, target=22050), # Resample from 48kHz to 22kHz
    OverlapChannels(), # Overap channels by sum (C, N) -> (1, N)
    RandomCrop(22050 * 3), # Random crop from file
    Scale(0.8) # Scale waveform
)

```
