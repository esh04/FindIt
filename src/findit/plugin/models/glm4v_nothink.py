"""GLM4V with thinking disabled — adds /nothink and prefills <think></think> in the template.
Also rearranges user-message content into interleaved form when the prompt text
carries `<image N>` placeholders (the GLM4V wrapper builds images-first-then-text by
default; this hook puts each image at its placeholder position)."""

from lmms_eval.models.simple.glm4v import GLM4V

from ._interleave import interleave_batched_messages


class GLM4VNoThink(GLM4V):
    def __init__(self, pretrained="zai-org/GLM-4.6V-Flash", **kwargs):
        super().__init__(pretrained=pretrained, **kwargs)
        orig_apply = self.processor.apply_chat_template

        def apply_no_think(messages, *args, **kw):
            kw.setdefault("enable_thinking", False)
            return orig_apply(interleave_batched_messages(messages), *args, **kw)

        self.processor.apply_chat_template = apply_no_think
