import io
import json
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import scipy
import torch
import webdataset as wds
from torch import Tensor


def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else d


def first(x):
    return x[0]


def identity(x):
    return x


def log_and_continue(exn):
    print(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def preprocess(item):
    """Optimized joint processing of .wav audio file and .json metadata"""
    # Json
    metadata_file = item["json"]
    metadata = json.loads(metadata_file.decode("utf-8"))
    # Audio
    data = item["wav"]
    with io.BytesIO(data) as stream:
        rate, array = scipy.io.wavfile.read(stream)  # wav only but the fastest
        wave = torch.from_numpy(np.copy(array.T))
    return wave, metadata


def crop_and_pad(
    tensor: Tensor,
    crop_size: int,
    max_crops: Optional[int] = None,
) -> List[Tensor]:
    """Crops a tensor in chunks and returns each chunk"""
    channels, length = tensor.shape
    num_crops = length // crop_size
    max_crops = min(default(max_crops, num_crops), num_crops)
    crops = []
    # Iterate over the crops
    for i in range(max_crops):  # type: ignore
        crop = tensor[:, i * crop_size : (i + 1) * crop_size]  # Crop the tensor
        crops.append(crop)
    # No zero padding needed in this cases
    if max_crops < num_crops or length % crop_size == 0:  # type: ignore
        return crops
    else:
        # Pad the last crop with zeros
        last_crop = tensor[:, num_crops * crop_size :]
        padding = torch.zeros(channels, crop_size - last_crop.shape[-1])
        padded_crop = torch.cat([last_crop, padding], dim=1)
        crops.append(padded_crop)
        return crops


def _crop_audio(data, crop_size: int, max_crops: Optional[int] = None, handler=None):
    """WebDataset crop filter, yields sequential crops"""
    for sample in data:
        audio, info = sample
        try:
            # Crop audio in sequential chunks
            crops = crop_and_pad(audio, crop_size=crop_size, max_crops=max_crops)
            # Yield each crop
            for crop in crops:
                yield (crop, info)
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


crop_audio = wds.filters.RestCurried(_crop_audio)


class AudioWebDataset(wds.WebDataset):
    def __init__(
        self,
        urls: Union[str, Sequence[str]],
        shuffle: Optional[int] = None,
        batch_size: Optional[int] = None,
        transforms: Optional[Callable] = None,
        use_wav_processor: bool = False,
        crop_size: Optional[int] = None,
        max_crops: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(urls=urls, resampled=True, handler=log_and_continue, **kwargs)

        # Decode audio
        if use_wav_processor:
            # More efficient but for wav only
            self.map(preprocess, handler=log_and_continue)
        else:
            self.decode(wds.torch_audio, handler=log_and_continue)
            self.to_tuple("wav;flac", "json", handler=log_and_continue)
            self.map_tuple(first, identity, handler=log_and_continue)

        # Transform audio
        if exists(transforms):
            self.map_tuple(transforms, identity, handler=log_and_continue)

        # Crop by yielding each crop sequentially
        if exists(crop_size):
            self.compose(
                crop_audio(
                    crop_size=crop_size, max_crops=max_crops, handler=log_and_continue
                )
            )

        # Shuffle
        if exists(shuffle):
            self.shuffle(shuffle)

        # Batch items
        if exists(batch_size):
            self.batched(batch_size)


class AudioWebDataloader(wds.WebLoader):
    def __init__(
        self,
        urls: Union[str, Sequence[str]],
        num_workers: int,
        batch_size: int,
        shuffle: int,
        epoch_length: Optional[int] = None,
        **kwargs,
    ):
        # Build dataset
        dataset = AudioWebDataset(urls=urls, shuffle=shuffle, batch_size=None, **kwargs)

        super().__init__(
            dataset=dataset,
            batch_size=None,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            prefetch_factor=2,
        )

        # Shuffle between workers
        self.shuffle(shuffle)

        # Batching
        self.batched(batch_size)

        # Epoched
        if exists(epoch_length):
            self.with_epoch(epoch_length)


if __name__ == "__main__":

    batch_size = 32

    loader = AudioWebDataloader(
        urls="pipe:aws s3 cp s3://thebucket/mix-{000000..000012}.tar -",
        batch_size=batch_size,
        shuffle=256,
        num_workers=12,
        use_wav_processor=True,
        # crop_size=2**18,
        # max_crops=100
    )

    # Or AudioWebDataset using torch dataloader
    # loader = torch.utils.data.DataLoader(
    #     dataset=AudioWebDataset(
    #         urls='pipe:aws s3 cp s3://thebucket/mix-{000000..000012}.tar -',
    #         batch_size=batch_size,
    #         shuffle=256,
    #     ),
    #     num_workers=12,
    #     batch_size=None,
    #     shuffle=False,
    #     pin_memory=True,
    #     prefetch_factor=2,
    # )

    """ Logging """

    # def dict_hash(d) -> str:
    #     import hashlib
    #     dhash = hashlib.md5()
    #     dhash.update(json.dumps(d, sort_keys=True).encode())
    #     return dhash.hexdigest()

    # for i, batch in enumerate(loader):
    #     audios, infos = batch
    #     for j, (audio, info) in enumerate(zip(audios, infos)):
    #         # We hash the dict to check the occurrence of repeating items by info
    #         print(i, j, audio.shape, dict_hash(info)[0:8])

    """ Timing """

    # import time
    # count, t0 = 0, time.time()
    # for batch in loader:
    #     count += 1
    #     if count > 180:
    #         break
    # tf = time.time()
    # print(f"BATCH/S: {count*batch_size/(tf-t0)}")

    # BATCH/S: 114.28585607357242, with crop_size=2**18 from 2**21 length items
    # BATCH/S: 30.297565637487082, without cropping
