"""Small typed registries used to keep the sim2sim bricks pluggable."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name -> factory/class mapping with explicit, readable failures."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T | None = None) -> Callable[[T], T] | T:
        """Register directly, or as a decorator when ``item`` is omitted."""

        def _do(value: T) -> T:
            key = name.strip().lower()
            if not key:
                raise ValueError(f"{self.kind} name must not be empty")
            if key in self._items and self._items[key] is not value:
                raise ValueError(f"Duplicate {self.kind} '{name}'")
            self._items[key] = value
            return value

        if item is None:
            return _do
        return _do(item)

    def get(self, name: str) -> T:
        key = name.strip().lower()
        try:
            return self._items[key]
        except KeyError as exc:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {', '.join(sorted(self._items))}"
            ) from exc

    def create(self, name: str, /, **kwargs: object) -> object:
        factory = self.get(name)
        if not callable(factory):
            raise TypeError(f"{self.kind} '{name}' is not callable")
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __contains__(self, name: str) -> bool:
        return name.strip().lower() in self._items

    def __iter__(self) -> Iterable[str]:
        return iter(self.names())
