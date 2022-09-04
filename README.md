
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
An unsupervised dataset for LJSpeech with voice-only data.
```py
from audio_data_pytorch import LJSpeechDataset

dataset = LJSpeechDataset(root='./data')

dataset[0] # (1, 158621)
dataset[1] # (1, 153757)
```

### Common Voice Dataset
Multilanguage wrapper for the [Common Voice](https://commonvoice.mozilla.org/) dataset with voice-only data. Requires `pip install datasets`. Note that each language requires several GBs of storage, and that you have to confirm access for each distinct version you use e.g. [here](https://huggingface.co/datasets/mozilla-foundation/common_voice_10_0), to validate your Huggingface access token. You can provide a list of `languages` and to avoid an unbalanced dataset the values will be interleaved by downsampling the majority language to have the same number of samples as the minority language.

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
    with_sample_rate: bool = False,  # Returns sample rate as second argument
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


## Transforms

You can use the following individual transforms, or merge them with `nn.Sequential()`:

```py
from audio_data_pytorch import Crop
crop = Crop(size=22050*2, start=0) # Crop 2 seconds at 22050 Hz from the start of the file

from audio_data_pytorch import RandomCrop
random_crop = RandomCrop(size=22050*2) # Crop 2 seconds at 22050 Hz from a random position

from audio_data_pytorch import Resample
resample = Resample(source=48000, target=22050), # Resamples from 48kHz to 22kHz

from audio_data_pytorch import OverlapChannels
overlap = OverlapChannels() # Overap channels by sum (C, N) -> (1, N)

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
    overlap_channels: bool = False,
    duplicate_channels: bool = False,
)
```
