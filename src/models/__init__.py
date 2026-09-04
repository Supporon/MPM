"""主模型与旧版标签细化适配器。"""

from .registry import LABEL_REFINER_REGISTRY, MODEL_REGISTRY

# 导入模型适配器以确保它们被注册
from . import cnn  # noqa: F401
from . import cnn2d  # noqa: F401
from . import mlp  # noqa: F401
from . import rf  # noqa: F401
from . import rf_constrained  # noqa: F401
from . import spe  # noqa: F401
from . import spe_constrained  # noqa: F401

__all__ = ["MODEL_REGISTRY", "LABEL_REFINER_REGISTRY"]
