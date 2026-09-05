"""Example extraction: top-k + quantile-stratified activating snippets, negatives, fuzz sets.

Design notes (calibrated on the 5 hand-labeled sample features):
- Top-k alone over-cleans the story (feature12298 "after_a" looks perfect at top,
  messy at low quantiles) -> explainer and scorer examples are stratified over
  activation quantiles, not just top-k.
- Dense features exist (feature24792 fires on 92.7% of tokens). For those,
  highlighting every firing token is useless; we highlight only tokens above the
  P75 of the feature's activation distribution and tell the explainer the density.
- Detection scoring is anchored on the FINAL token of a left-context snippet
  ("does the feature fire on the last token?"). Unlike whole-window judgment
  (delphi), this stays well-defined for dense features and matches per-token
  routing decisions, which is what RouterInterp ultimately needs.
"""
import random
from dataclasses import dataclass, field

import torch

from .data import Corpus, FeatureRecord


@dataclass
class SamplerConfig:
    ctx_left: int = 31          # tokens of context to the left of the anchor
    ctx_right: int = 8          # right context (explainer/fuzz views only)
    n_explainer: int = 24       # examples shown to the explainer
    explainer_top_frac: float = 0.33  # fraction of explainer examples from the top bin
    n_test_act: int = 20        # activating detection probes
    n_test_neg: int = 20        # non-activating detection probes
    n_fuzz: int = 10            # per class (correct / incorrect highlighting)
    n_bins: int = 10            # quantile bins for stratification
    max_per_prompt: int = 2     # diversity cap per (batch, prompt)
    dense_threshold: float = 0.15    # density above which "dense" handling kicks in
    dense_highlight_q: float = 0.75  # highlight only acts above this quantile when dense
    seed: int = 0


@dataclass
class Snippet:
    batch: int
    prompt: int
    pos: int                    # anchor token position
    tokens: list                # window token strings
    anchor_offset: int          # anchor index within tokens
    highlights: set             # offsets within tokens to mark (may be empty)
    max_act: float              # anchor activation (0 for negatives)
    norm_score: int             # 0-10 vs feature max (0 for negatives)

    def render(self, highlighted: bool, upto_anchor: bool = False) -> str:
        toks = self.tokens[: self.anchor_offset + 1] if upto_anchor else self.tokens
        if not highlighted:
            return "".join(toks)
        out, i = [], 0
        while i < len(toks):
            if i in self.highlights:
                j = i
                while j + 1 < len(toks) and (j + 1) in self.highlights:
                    j += 1
                out.append("<<" + "".join(toks[i: j + 1]) + ">>")
                i = j + 1
            else:
                out.append(toks[i])
                i += 1
        return "".join(out)


@dataclass
class FeatureExamples:
    explainer: list
    test_act: list
    test_neg: list
    fuzz_correct: list
    fuzz_incorrect: list
    stats: dict = field(default_factory=dict)


def _make_snippet(feat: FeatureRecord, corpus: Corpus, b: int, p: int, pos: int,
                  cfg: SamplerConfig, highlight_thr: float, act: float) -> Snippet:
    toks = corpus.tokens[b][p]
    lo = max(0, pos - cfg.ctx_left)
    hi = min(len(toks), pos + cfg.ctx_right + 1)
    window = list(toks[lo:hi])
    anchor = pos - lo
    prompt_acts = feat.acts_by_prompt.get((b, p), {})
    highlights = {t - lo for t, a in prompt_acts.items() if lo <= t < hi and a > highlight_thr}
    highlights.add(anchor) if act > 0 else None
    norm = round(10 * act / feat.max_act) if feat.max_act > 0 else 0
    return Snippet(batch=b, prompt=p, pos=pos, tokens=window, anchor_offset=anchor,
                   highlights=highlights, max_act=act, norm_score=int(norm))


def _stratified_pick(order_rows: list, n_bins: int, n_want: int, rng: random.Random,
                     used: set, max_per_prompt: int, prompt_counts: dict) -> list:
    """Pick ~n_want row indices stratified over n_bins equal-population bins of
    the activation-sorted rows, respecting dedup constraints."""
    N = len(order_rows)
    picked = []
    per_bin = max(1, n_want // n_bins)
    for bin_i in range(n_bins):
        lo, hi = (N * bin_i) // n_bins, (N * (bin_i + 1)) // n_bins
        candidates = list(range(lo, hi))
        rng.shuffle(candidates)
        got = 0
        for c in candidates:
            if got >= per_bin or len(picked) >= n_want:
                break
            row = order_rows[c]
            key = (row[0], row[1], row[2])
            pkey = (row[0], row[1])
            if key in used or prompt_counts.get(pkey, 0) >= max_per_prompt:
                continue
            used.add(key)
            prompt_counts[pkey] = prompt_counts.get(pkey, 0) + 1
            picked.append(row)
            got += 1
    return picked


def build_examples(feat: FeatureRecord, corpus: Corpus, cfg: SamplerConfig) -> FeatureExamples:
    rng = random.Random(cfg.seed * 100003 + feat.feature_id)
    feat.build_index(corpus)
    density = feat.density(corpus)
    acts = feat.acts
    highlight_thr = 0.0
    if density >= cfg.dense_threshold:
        highlight_thr = float(torch.quantile(acts, cfg.dense_highlight_q))

    # rows sorted by activation ascending: (b, p, pos, act)
    order = torch.argsort(acts)
    rec = feat.records[order]
    rows = [(int(r[0]), int(r[1]), int(r[2]), float(r[3])) for r in
            rec[:: max(1, len(rec) // 200000)]]  # subsample huge features for speed, keeps order
    used, prompt_counts = set(), {}

    # --- explainer: top-heavy + stratified ---
    n_top = max(1, int(cfg.n_explainer * cfg.explainer_top_frac))
    top_rows = []
    for row in reversed(rows):  # highest act first
        key, pkey = (row[0], row[1], row[2]), (row[0], row[1])
        if key in used or prompt_counts.get(pkey, 0) >= cfg.max_per_prompt:
            continue
        used.add(key)
        prompt_counts[pkey] = prompt_counts.get(pkey, 0) + 1
        top_rows.append(row)
        if len(top_rows) >= n_top:
            break
    strat_rows = _stratified_pick(rows, cfg.n_bins, cfg.n_explainer - len(top_rows),
                                  rng, used, cfg.max_per_prompt, prompt_counts)
    explainer_rows = sorted(top_rows + strat_rows, key=lambda r: -r[3])

    # --- held-out activating probes for detection ---
    test_rows = _stratified_pick(rows, cfg.n_bins, cfg.n_test_act, rng, used,
                                 cfg.max_per_prompt, prompt_counts)

    # --- fuzzing probes (held out too) ---
    fuzz_rows = _stratified_pick(rows, cfg.n_bins, cfg.n_fuzz * 2, rng, used,
                                 cfg.max_per_prompt, prompt_counts)

    # --- negatives: uniform positions where the feature does not fire ---
    neg_rows = []
    tries = 0
    while len(neg_rows) < cfg.n_test_neg and tries < 200000:
        tries += 1
        b = rng.randrange(corpus.n_batches)
        p = rng.randrange(corpus.n_prompts)
        pos = rng.randrange(cfg.ctx_left // 2, corpus.n_tokens)  # keep some left context
        if feat.fires_at(corpus, b, p, pos):
            continue
        key = (b, p, pos)
        if key in used:
            continue
        used.add(key)
        neg_rows.append((b, p, pos, 0.0))
    if len(neg_rows) < cfg.n_test_neg:
        raise RuntimeError(f"feature {feat.feature_id}: could not sample "
                           f"{cfg.n_test_neg} negatives (density={density:.3f})")

    mk = lambda r: _make_snippet(feat, corpus, r[0], r[1], r[2], cfg, highlight_thr, r[3])
    explainer = [mk(r) for r in explainer_rows]
    test_act = [mk(r) for r in test_rows]
    test_neg = [mk(r) for r in neg_rows]

    fuzz_correct = [mk(r) for r in fuzz_rows[: cfg.n_fuzz]]
    fuzz_incorrect = []
    for r in fuzz_rows[cfg.n_fuzz: cfg.n_fuzz * 2]:
        s = mk(r)
        k = max(1, len(s.highlights))
        non_firing = [i for i in range(len(s.tokens)) if i not in s.highlights]
        if len(non_firing) < k:  # dense feature: fall back to lowest-act positions
            prompt_acts = feat.acts_by_prompt.get((s.batch, s.prompt), {})
            lo_off = s.pos - s.anchor_offset
            cand = sorted(range(len(s.tokens)),
                          key=lambda i: prompt_acts.get(lo_off + i, 0.0))
            non_firing = [i for i in cand if i not in s.highlights] or cand
        s.highlights = set(rng.sample(non_firing, min(k, len(non_firing))))
        fuzz_incorrect.append(s)

    stats = {
        "density": density,
        "n_firings": feat.n_firings,
        "max_act": feat.max_act,
        "act_quantiles": {q: float(torch.quantile(acts, q)) for q in (0.25, 0.5, 0.75, 0.9, 0.99)},
        "highlight_threshold": highlight_thr,
        "dense": density >= cfg.dense_threshold,
    }
    return FeatureExamples(explainer=explainer, test_act=test_act, test_neg=test_neg,
                           fuzz_correct=fuzz_correct, fuzz_incorrect=fuzz_incorrect, stats=stats)
