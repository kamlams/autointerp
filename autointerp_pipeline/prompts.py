"""Prompt templates. Kept model-agnostic; tested with moonshotai/Kimi-K3 via RiVault."""

EXPLAINER_SYSTEM = """You are an interpretability researcher analyzing one feature (latent unit) of a sparse autoencoder trained on a language model's internal activations. A feature fires with some strength on some tokens of some texts and is silent elsewhere.

You will see text snippets where this feature fires. Inside each snippet, the tokens on which the feature fires are wrapped in << >> delimiters. Each snippet is preceded by the firing strength at its strongest marked token, on a 0-10 scale relative to this feature's maximum.

Find the most precise pattern that predicts where and how strongly the feature fires. Check, in order of preference:
1. the exact token identity or word form it fires on;
2. the immediately preceding tokens or local grammatical context;
3. position in the sentence or document;
4. broader topic or style of the text.
Prefer precise mechanistic descriptions ("fires on the token ' to' following a noun phrase") over vague topical ones ("related to instructions"). If strength varies systematically, say what drives strong vs weak firing. If the examples are heterogeneous, describe the dominant pattern and note the exceptions.

End your reply with exactly these two lines (no markdown, no extra text after them):
LABEL: <2-6 word name for the feature>
EXPLANATION: <one or two sentences stating when the feature fires>"""

EXPLAINER_DENSE_NOTE = """
NOTE: this feature fires on {density:.0%} of ALL tokens in the corpus — it is a dense/ubiquitous feature, not a selective one. The snippets below mark only its STRONGEST firings (above the {q:.0%} quantile). Describe what distinguishes strong firing from weak firing, and say explicitly that the feature is near-ubiquitous."""

EXPLAINER_EXAMPLE_HEADER = "Snippets (strongest first):\n"


DETECTION_SYSTEM = """You will evaluate a proposed explanation of a language-model feature against text snippets.

Explanation of the feature:
{explanation}

You will see {n} snippets. For each snippet, decide whether this feature fires ON THE FINAL TOKEN of the snippet (the snippet is cut exactly at the token being probed; judge that last token in its left context). Answer 1 if, per the explanation, the feature fires on the final token; answer 0 if it does not.

Reply with ONLY a JSON array of {n} integers (0 or 1), e.g. [1,0,0,1,1]. No other text."""

FUZZING_SYSTEM = """You will evaluate a proposed explanation of a language-model feature against text snippets with marked tokens.

Explanation of the feature:
{explanation}

You will see {n} snippets in which some tokens are wrapped in << >> marks. In some snippets the marks are placed correctly on tokens where this feature fires; in others the marks are placed on wrong tokens where it does not fire. For each snippet, answer 1 if the marked tokens are where the explanation says the feature fires, and 0 if the marking contradicts the explanation.

Reply with ONLY a JSON array of {n} integers (0 or 1), e.g. [1,0,0,1,1]. No other text."""


def format_explainer_user(snippets, density: float | None, dense_q: float | None) -> str:
    parts = []
    if density is not None:
        parts.append(EXPLAINER_DENSE_NOTE.format(density=density, q=dense_q))
    parts.append(EXPLAINER_EXAMPLE_HEADER)
    for i, s in enumerate(snippets, 1):
        parts.append(f"--- Snippet {i} (strength {s.norm_score}/10) ---")
        parts.append(s.render(highlighted=True))
    return "\n".join(parts)


def format_batch_user(rendered_snippets: list[str]) -> str:
    parts = []
    for i, text in enumerate(rendered_snippets, 1):
        parts.append(f"--- Snippet {i} ---")
        parts.append(text)
    parts.append("\nReply with ONLY the JSON array.")
    return "\n".join(parts)
