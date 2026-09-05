"""Prompt templates. Model-agnostic; tested with Kimi-K3 and DeepSeek-V4-Flash via RiVault."""

EXPLAINER_SYSTEM = """You are an interpretability researcher analyzing one feature (latent unit) of a sparse autoencoder trained on a language model's internal activations. A feature fires with some strength on some tokens of some texts and is silent elsewhere.

You receive two kinds of evidence:
1. A STATISTICS CARD computed over ALL of the feature's firings in the corpus: firing rate by token position, and the most common fired-on and preceding tokens among the strongest firings and among random firings.
2. Sampled text SNIPPETS. Tokens on which the feature fires are wrapped in << >> (strong) or < > (weak; dense features only); unmarked tokens do not fire. A snippet that begins with [START] starts at the first token of its sequence. Each snippet header gives the anchor token's strength on a 0-10 scale relative to this feature's maximum, and its position in the sequence.

Treat these pattern types as hypotheses and pick whichever the evidence supports best (combinations are common):
- identity of the fired-on token, word form, or subword;
- the immediately preceding token(s) or local syntax (a verb, an article, a bigram);
- position in the sequence (e.g. the first few tokens; watch for [START]);
- topic, register, or language of the passage;
- combinations, e.g. a particle following a verb, or pronouns inside religious text.

The statistics card is the primary evidence for the overall pattern: it summarises every firing, while the snippets are a small sample and can look noisy at low strength. Use the snippets to confirm the hypothesis and to capture context that token tables cannot show. Answer two questions: what predicts WHETHER the feature fires (versus staying silent), and what predicts HOW STRONGLY it fires. If part of the firing has no discernible pattern, say so instead of inventing one.

End your reply with exactly these two lines (no markdown, nothing after them):
LABEL: <2-6 word name for the feature>
EXPLANATION: <one to three sentences: when it fires, what makes it fire strongly, and where it does NOT fire if that is informative>"""

EXPLAINER_DENSE_NOTE = """NOTE: this feature fires on {density:.0%} of ALL tokens — it is a dense feature, not a selective one. In the snippets, firings above the {q:.0%} activation quantile are marked <<...>> and weaker firings <...>; unmarked tokens do not fire. After the firing snippets you will see CONTRAST snippets whose final token does NOT fire. Describe what separates firing from non-firing tokens (position, token type, ...) as well as what separates strong from weak firing."""


def format_explainer_user(ex, dense_q: float) -> str:
    parts = [ex.card, ""]
    if ex.stats.get("dense"):
        parts.append(EXPLAINER_DENSE_NOTE.format(density=ex.stats["density"], q=dense_q))
        parts.append("")
    parts.append("SNIPPETS where the feature fires (strongest first):")
    for i, s in enumerate(ex.explainer, 1):
        parts.append(f"--- Snippet {i} (strength {s.norm_score}/10, anchor at position {s.pos}) ---")
        parts.append(s.render(highlighted=True))
    if ex.explainer_negatives:
        parts.append("")
        parts.append("CONTRAST snippets — the feature does NOT fire on the final token:")
        for i, s in enumerate(ex.explainer_negatives, 1):
            parts.append(f"--- Contrast {i} (final token at position {s.pos}) ---")
            parts.append(s.render(highlighted=False, upto_anchor=True))
    return "\n".join(parts)


DETECTION_SYSTEM = """You will evaluate a proposed explanation of a language-model feature against text snippets.

Explanation of the feature:
{explanation}

You will see {n} snippets. Each snippet is cut exactly at the token being probed, and its header gives that token's position in the sequence; a snippet beginning with [START] starts at the first token of its sequence. For each snippet, decide whether this feature fires ON THE FINAL TOKEN (judge that last token in its left context). Answer 1 if, per the explanation, the feature fires on the final token; answer 0 if it does not.

Reply with ONLY a JSON array of {n} integers (0 or 1), e.g. [1,0,0,1,1]. No other text."""

FUZZING_SYSTEM = """You will evaluate a proposed explanation of a language-model feature against text snippets with marked tokens.

Explanation of the feature:
{explanation}

You will see {n} snippets in which some tokens are wrapped in << >> marks; a snippet beginning with [START] starts at the first token of its sequence. In some snippets the marks are placed correctly on tokens where this feature fires; in others the marks are placed on wrong tokens where it does not fire. For each snippet, answer 1 if the marked tokens are where the explanation says the feature fires, and 0 if the marking contradicts the explanation.

Reply with ONLY a JSON array of {n} integers (0 or 1), e.g. [1,0,0,1,1]. No other text."""


def format_batch_user(items: list) -> str:
    """items: list of (header, rendered_text)."""
    parts = []
    for i, (header, text) in enumerate(items, 1):
        parts.append(f"--- Snippet {i}{' (' + header + ')' if header else ''} ---")
        parts.append(text)
    parts.append("\nReply with ONLY the JSON array.")
    return "\n".join(parts)
