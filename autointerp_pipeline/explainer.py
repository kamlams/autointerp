"""Explanation generation."""
import re

from . import prompts
from .llm import LLMClient
from .sampler import FeatureExamples, SamplerConfig


def explain(client: LLMClient, ex: FeatureExamples, cfg: SamplerConfig,
            max_tokens: int = 16384) -> dict:
    user = prompts.format_explainer_user(ex, cfg.dense_highlight_q)
    reply = client.chat(
        [{"role": "system", "content": prompts.EXPLAINER_SYSTEM},
         {"role": "user", "content": user}],
        max_tokens=max_tokens, tag="explain",
    )
    label = _extract(reply, "LABEL")
    explanation = _extract(reply, "EXPLANATION")
    if explanation is None:  # fall back to last non-empty line
        lines = [l.strip() for l in reply.strip().splitlines() if l.strip()]
        explanation = lines[-1] if lines else reply.strip()
    return {"label": label or "", "explanation": explanation, "raw": reply, "prompt": user}


def _extract(text: str, key: str) -> str | None:
    m = re.search(rf"^{key}:\s*(.+)$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None
