import asyncio
import concurrent
import math
import os
import re
import shutil
import tarfile
import zipfile
from typing import List, Optional, Sequence, TypeVar

import aiohttp
import torch
from torch.utils.data.dataset import Dataset, Subset
from tqdm import tqdm
from typing_extensions import TypeGuard

T = TypeVar("T")


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
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in urls:
                file_path = self.get_file_path(url)
                # Check if file already exists
                if self.check_exists and os.path.exists(file_path):
                    self.files += [file_path]
                else:
                    tasks += [self.download(url, session)]
            self.files += await asyncio.gather(*tasks, return_exceptions=True)
        return self.files

    def download_all(self) -> List[str]:
        return asyncio.run(self.download_all_async())

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


class Decompressor:
    def __init__(
        self,
        files: List[str],
        ext: str = "tar",
        path: str = ".",
        remove_on_exit: bool = False,
        description: str = "Decompressing",
    ):
        self.files = files
        self.ext = ext
        self.path = path
        self.paths: List[str] = []
        self.remove_on_exit = remove_on_exit
        self.description = description

    def extract_all(self, archive, path: str):
        for member in tqdm(archive.infolist(), desc=f"{self.description}: {path}"):
            archive.extract(member, path)

    def decompress(self, file_name: str):
        extract_path = os.path.splitext(file_name)[0]
        if self.ext == "zip":
            with zipfile.ZipFile(file_name, "r") as archive:
                self.extract_all(archive, extract_path)
        else:
            with tarfile.open(file_name) as archive:
                self.extract_all(archive, extract_path)
        return extract_path

    async def decompress_all_async(self):
        loop = asyncio.get_running_loop()
        with concurrent.futures.ProcessPoolExecutor() as pool:
            results = [
                loop.run_in_executor(pool, self.decompress, file) for file in self.files
            ]
            self.paths += await asyncio.gather(*results)
        return self.paths

    def decompress_all(self):
        return asyncio.run(self.decompress_all_async())

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
            async with Decompressor(files, ext="zip", remove_on_exit=True) as paths:
                print(paths)

    asyncio.run(run(urls))

    # Sync example
    with Downloader(urls, remove_on_exit=True) as files:
        with Decompressor(files, ext="zip", remove_on_exit=True) as paths:
            print(paths)
