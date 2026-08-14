"""二阶段实验框架使用的小型显式组件注册表。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


class RegistryError(ValueError):
    """组件名称缺失、重复或未知时抛出的异常。"""


class ComponentRegistry:
    """将稳定的配置名称映射到工厂，避免运行器耦合具体实现。"""

    def __init__(self, domain: str):
        self.domain = domain
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any], *, replace: bool = False) -> None:
        key = str(name).strip()
        if not key:
            raise RegistryError(f"{self.domain} component name cannot be empty")
        if key in self._factories and not replace:
            raise RegistryError(f"{self.domain} component already registered: {key}")
        self._factories[key] = factory

    def decorator(self, name: str):
        """返回一个装饰器，以 ``name`` 注册类或工厂。"""

        def register(factory: Callable[..., Any]):
            self.register(name, factory)
            return factory

        return register

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._factories[name]
        except KeyError as error:
            available = ", ".join(self.names()) or "<none>"
            raise RegistryError(
                f"Unknown {self.domain} component '{name}'. Available: {available}"
            ) from error

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.get(name)(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def require(self, name: str) -> None:
        self.get(name)

    def validate(self, name: str, *args: Any, **kwargs: Any) -> None:
        """运行由组件自行提供的可选配置校验器。"""
        factory = self.get(name)
        validator = getattr(factory, "validate_config", None)
        if validator is not None:
            validator(*args, **kwargs)

    def update(self, entries: Iterable[tuple[str, Callable[..., Any]]]) -> None:
        for name, factory in entries:
            self.register(name, factory)
