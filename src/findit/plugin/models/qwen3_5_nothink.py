"""Qwen3.5 with thinking disabled (no `<think>…</think>` block)."""

from lmms_eval.models.simple.qwen3_5 import Qwen3_5


class Qwen3_5NoThink(Qwen3_5):
    def __init__(self, pretrained: str = "Qwen/Qwen3.5-9B", **kwargs):
        kwargs["enable_thinking"] = False
        super().__init__(pretrained=pretrained, **kwargs)
