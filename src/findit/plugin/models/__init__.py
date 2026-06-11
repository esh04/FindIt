from lmms_eval.models.chat.internvl_hf import InternVLHf

from ._interleave import interleave_chat_messages

# Monkey-patch upstream InternVLHf so its processor.apply_chat_template receives
# user-message content interleaved at `<image N>` placeholder positions. We
# don't subclass because the legacy plugin loader registers entries into
# AVAILABLE_SIMPLE_MODELS, which would conflict with InternVLHf being chat-class.
_orig_internvl_init = InternVLHf.__init__


def _patched_internvl_init(self, *args, **kwargs):
    _orig_internvl_init(self, *args, **kwargs)
    orig_apply = self.processor.apply_chat_template

    def apply_with_interleave(messages, *a, **k):
        return orig_apply(interleave_chat_messages(messages), *a, **k)

    self.processor.apply_chat_template = apply_with_interleave


InternVLHf.__init__ = _patched_internvl_init


AVAILABLE_MODELS = {
    "gemma4": "Gemma4",
    "glm4v_nothink": "GLM4VNoThink",
    "qwen3_5_nothink": "Qwen3_5NoThink",
    "openrouter": "OpenRouterNoThink",
}
