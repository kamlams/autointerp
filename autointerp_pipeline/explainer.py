"""Explanation generation."""
import re

from . import prompts
from .llm import LLMClient
from .sampler import FeatureExamples, SamplerConfig


def explain(client: LLMClient, ex: FeatureExamples, cfg: SamplerConfig,
            max_tokens: int = 16384, max_words: int = 0) -> dict:
    """One explainer call -> LABEL + EXPLANATION. With max_words > 0 the prompt asks for at most
    that many words, and an EXPLANATION still over the limit gets one follow-up turn in the same
    conversation asking the explainer to shorten it; both versions are returned."""
    user = prompts.format_explainer_user(ex, cfg.dense_highlight_q)
    messages = [{"role": "system", "content": prompts.explainer_system(max_words)},
                {"role": "user", "content": user}]
    reply = client.chat(messages, max_tokens=max_tokens, tag="explain")
    label, explanation = _parse(reply)
    out = {"label": label, "explanation": explanation, "raw": reply, "prompt": user}
    if max_words:
        n = out["explanation_words"] = _words(explanation)
        if n > max_words:
            out.update(label_first=label, explanation_first=explanation)
            try:
                again = client.chat(messages + [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": prompts.SHORTEN_USER.format(n=n, limit=max_words)}],
                    max_tokens=max_tokens, tag="shorten")
                label2, explanation2 = _parse(again)
                out.update(label=label2 or label, explanation=explanation2, raw_shorten=again,
                           explanation_words=_words(explanation2))
            except Exception as e:  # noqa: BLE001 - keep the long explanation rather than lose the feature
                out["shorten_error"] = str(e)[:300]
        out["over_limit"] = out["explanation_words"] > max_words
    return out


def _words(text: str) -> int:
    return len(text.split())


def _parse(reply: str) -> tuple[str, str]:
    label = _extract(reply, "LABEL")
    explanation = _extract(reply, "EXPLANATION")
    if explanation is None:  # fall back to last non-empty line
        lines = [l.strip() for l in reply.strip().splitlines() if l.strip()]
        explanation = lines[-1] if lines else reply.strip()
    return label or "", explanation


def _extract(text: str, key: str) -> str | None:
    m = re.search(rf"^{key}:\s*(.+)$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None
