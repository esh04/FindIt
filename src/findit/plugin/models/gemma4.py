"""Gemma 4 wrapper — reuses lmms-eval's Gemma3 loop, swaps in the Gemma4 HF class.
Also patches processor.apply_chat_template to interleave user-message content at
`<image N>` placeholder positions. Without this hook, placeholders survive as raw
text and images stack at the front of the user message."""

import transformers
from transformers import Gemma4ForConditionalGeneration

import lmms_eval.models.simple.gemma3 as _gemma3_mod
from lmms_eval.models.simple.gemma3 import Gemma3

from ._interleave import interleave_batched_messages


class Gemma4(Gemma3):
    def __init__(self, pretrained="google/gemma-4-E4B-it", **kwargs):
        orig = _gemma3_mod.Gemma3ForConditionalGeneration
        _gemma3_mod.Gemma3ForConditionalGeneration = Gemma4ForConditionalGeneration
        try:
            super().__init__(pretrained=pretrained, **kwargs)
        finally:
            _gemma3_mod.Gemma3ForConditionalGeneration = orig

        orig_apply = self.processor.apply_chat_template

        def apply_with_interleave(messages, *args, **kw):
            return orig_apply(interleave_batched_messages(messages), *args, **kw)

        self.processor.apply_chat_template = apply_with_interleave
