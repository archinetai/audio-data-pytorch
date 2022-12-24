import asyncio
import concurrent
import math
import os
import re
import shutil
import tarfile
import threading
import zipfile
from typing import List, Optional, Sequence, Tuple, TypeVar

import aiohttp
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset, Subset
from tqdm import tqdm
from typing_extensions import TypeGuard

T = TypeVar("T")

def split_artists(artists: str) -> List[str]:
    return (artists
            .replace(' w. ', ', ')
            .replace(' vs. ', ', ')
            .replace(' feat. ', ', ')
            .replace(' featuring ', ', ')
            .replace(' & ', ', ')
            .replace(' ft. ', ', ')
            .replace(' with ', ', ')
            .split(', '))

def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None


def camel_to_snake(name: str) -> str:
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def fractional_random_split(
    dataset: Dataset[T], fractions: Sequence[int]
) -> List[Subset[T]]:
    """Fractional split that follows the same convention as random_split"""
    assert sum(fractions) == 1.0, "Fractions must sum to 1.0"

    length = len(dataset)  # type: ignore[arg-type]
    indices = torch.randperm(length)
    splits = []
    cursor = 0

    for fraction in fractions:
        next_cursor = math.ceil(length * fraction + cursor)
        splits += [Subset(dataset, indices[cursor:next_cursor])]  # type: ignore[arg-type] # noqa
        cursor = next_cursor

    return splits


"""
Audio utils
"""


def is_silence(audio: Tensor, thresh: int = -60):
    dBmax = 20 * torch.log10(torch.flatten(audio.abs()).max())
    return dBmax < thresh


"""
Data/async utils
"""


def fast_scandir(path: str, exts: List[str], recursive: bool = False):
    # Scan files recursively faster than glob
    # From github.com/drscotthawley/aeiou/blob/main/aeiou/core.py
    subfolders, files = [], []

    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(path):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    if os.path.splitext(f.name)[1].lower() in exts:
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    if recursive:
        for path in list(subfolders):
            sf, f = fast_scandir(path, exts, recursive=recursive)
            subfolders.extend(sf)
            files.extend(f)  # type: ignore

    return subfolders, files


class RunThread(threading.Thread):
    def __init__(self, func):
        self.func = func
        self.result = None
        super().__init__()

    def run(self):
        self.result = asyncio.run(self.func)


def run_async(func):
    """Allows to run asyncio in an already running loop, e.g. Jupyter notebooks"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        thread = RunThread(func)
        thread.start()
        thread.join()
        return thread.result
    else:
        return asyncio.run(func)


class Downloader:
    def __init__(
        self,
        urls: List[str],
        path: str = ".",
        remove_on_exit: bool = False,
        check_exists: bool = True,
        description: str = "Downloading",
    ):
        self.urls = urls
        self.path = path
        self.files: List[str] = []
        self.remove_on_exit = remove_on_exit
        self.check_exists = check_exists
        self.description = description

    def get_file_path(self, url: str) -> str:
        os.makedirs(self.path, exist_ok=True)
        filename = url.split("/")[-1]
        return os.path.join(self.path, filename)

    async def download(self, url: str, session):

        async with session.get(url) as response:
            file_path = self.get_file_path(url)
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            chunk_size = 1024

            progress_bar = tqdm(
                desc=f"{self.description}: {file_path}",
                total=total_size_in_bytes,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            )

            with open(file_path, "wb") as file:
                async for chunk in response.content.iter_chunked(chunk_size):
                    size = file.write(chunk)
                    progress_bar.update(size)

            return file_path

    async def download_all_async(self) -> List[str]:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None)  # Unlimited timeout time
        ) as session:
            tasks = []
            for url in self.urls:
                file_path = self.get_file_path(url)
                # Check if file already exists
                if self.check_exists and os.path.exists(file_path):
                    self.files += [file_path]
                else:
                    tasks += [self.download(url, session)]
            self.files += await asyncio.gather(*tasks, return_exceptions=True)
        return self.files

    def download_all(self) -> List[str]:
        return run_async(self.download_all_async())

    def remove_files(self):
        for file in self.files:
            os.remove(file)

    def __enter__(self):
        return self.download_all()

    def __exit__(self, *args):
        if self.remove_on_exit:
            self.remove_files()

    async def __aenter__(self):
        return await self.download_all_async()

    async def __aexit__(self, *args):
        if self.remove_on_exit:
            self.remove_files()


def is_tar(file_name: str) -> bool:
    return file_name.lower().endswith((".tar", ".gz", ".bz2", ".xz"))


def is_zip(file_name: str) -> bool:
    return file_name.lower().endswith(".zip")


def is_7zip(file_name: str) -> bool:
    return file_name.lower().endswith(".7z")


class Decompressor:
    def __init__(
        self,
        files: List[str],
        path: str = ".",
        remove_on_exit: bool = False,
        check_exists: bool = True,
        description: str = "Decompressing",
    ):
        self.files = files
        self.path = path
        self.paths: List[str] = []
        self.remove_on_exit = remove_on_exit
        self.check_exists = check_exists
        self.description = description

    def get_path_and_folder(self, file_name: str) -> Tuple[str, str]:
        path = os.path.splitext(file_name)[0]  # Remove extension
        folder = os.path.split(path)[1]
        return path, folder

    def extract_all(self, archive, path: str):
        for member in tqdm(archive.infolist(), desc=f"{self.description}: {path}"):
            archive.extract(member, path)

    def decompress(self, file_name: str):
        path, _ = self.get_path_and_folder(file_name)
        if is_zip(file_name):
            with zipfile.ZipFile(file_name, "r") as archive:
                self.extract_all(archive, path)
        elif is_tar(file_name):
            with tarfile.open(file_name) as archive:
                self.extract_all(archive, path)
        elif is_7zip(file_name):
            import py7zr

            print(f"{self.description}: {path}")
            with py7zr.SevenZipFile(file_name, mode="r") as archive:
                archive.extractall(path=path)
        else:
            raise ValueError(f"Unsupported file extension: {file_name}")
        return path

    async def decompress_all_async(self):
        files_to_decompress = []
        # Remove already existing paths from files to decompress if requested
        if self.check_exists:
            for file_name in self.files:
                path, _ = self.get_path_and_folder(file_name)
                if os.path.exists(path):
                    self.paths += [path]
                else:
                    files_to_decompress += [file_name]
        else:
            files_to_decompress = self.files
        # Decompress with multiprocessing
        loop = asyncio.get_running_loop()
        with concurrent.futures.ProcessPoolExecutor() as pool:
            tasks = [
                loop.run_in_executor(pool, self.decompress, file_name)
                for file_name in files_to_decompress
            ]
            self.paths += await asyncio.gather(*tasks)
        return self.paths

    def decompress_all(self):
        return run_async(self.decompress_all_async())

    def remove_paths(self):
        for path in self.paths:
            shutil.rmtree(path)

    def __enter__(self):
        return self.decompress_all()

    def __exit__(self, *args):
        if self.remove_on_exit:
            self.remove_paths()

    async def __aenter__(self):
        return await self.decompress_all_async()

    async def __aexit__(self, *args):
        if self.remove_on_exit:
            self.remove_paths()


if __name__ == "__main__":

    """
    Downloader and decompressor usage
    """

    urls = ["https://zenodo.org/record/7135077/files/hashtags.zip"]

    # Async example
    async def run(urls):
        async with Downloader(urls, remove_on_exit=True) as files:
            async with Decompressor(files, remove_on_exit=True) as paths:
                print(paths)

    asyncio.run(run(urls))

    # Sync example
    with Downloader(urls, remove_on_exit=True) as files:
        with Decompressor(files, remove_on_exit=True) as paths:
            print(paths)
