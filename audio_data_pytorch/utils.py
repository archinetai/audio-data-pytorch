import re
from typing import Optional, TypeVar

from typing_extensions import TypeGuard

T = TypeVar("T")


def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None


def camel_to_snake(name: str) -> str:
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
