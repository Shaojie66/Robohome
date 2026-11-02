"""入库模块：识别新买的物品 → 按温区、类别、容量分配容器 → 更新物品台账。

也就是把不规范的东西装进规范的封装里，之后机器人只需要记住容器在哪。
"""

from .core import IntakeResult, NewItem, Placement, assign, choose_container
from .recognize import KeywordRecognizer, LLMRecognizer

__all__ = ["IntakeResult", "KeywordRecognizer", "LLMRecognizer", "NewItem", "Placement", "assign", "choose_container"]
