import math
import re
from typing import List, Optional, Sequence, TypeVar

import torch
from torch.utils.data.dataset import Dataset, Subset
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
