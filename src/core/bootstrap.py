"""仅导入一次内置组件模块，以触发其注册装饰器。"""

from __future__ import annotations

_LOADED = False


def load_builtin_components() -> None:
    global _LOADED
    if _LOADED:
        return
    from ..knowledge import providers as _knowledge  # noqa: F401
    from ..models import pu as _pu  # noqa: F401
    from ..models import cnn as _cnn  # noqa: F401
    from ..models import rf as _rf  # noqa: F401
    from ..models import spe as _spe  # noqa: F401
    from ..operators.features import builtins as _operators  # noqa: F401
    from ..predicates import builtins as _predicates  # noqa: F401
    from ..tasks import deep_edge as _deep_edge  # noqa: F401
    from ..tasks import target_area as _target_area  # noqa: F401
    from ..tuning import bayes as _bayes  # noqa: F401
    from ..tuning import none as _none  # noqa: F401
    from ..validation import metrics as _metrics  # noqa: F401
    from ..validation import splitters as _splitters  # noqa: F401
    _LOADED = True
