"""Example extraction: top + stratified activating snippets, negatives, fuzz sets.

Design notes (calibrated on the 5 hand-labeled sample features):
- Top-k alone over-cleans the story (feature12298 "after_a" looks perfect at top,
  messy at low quantiles) -> explainer and probe sets are stratified over activation.
  Half of each stratified set uses rank bins (equal population), half uses value bins
  (equal activation width), so heavy-tailed features also get their strong regime
  probed instead of 20/20 probes landing in the weak diffuse regime (feature14825).
- Diversity is enforced on the (preceding token, fired-on token) bigram, not on either
  token alone: capping the fired-on token would starve "to"-type features, capping the
  preceding token would starve "after a"-type features. Quotas are proportional to a
  bigram's share of the candidate pool (compressed), with a floor, so a dominant bigram
  is still represented but cannot fill the set (feature25337's top-8 were 7x cut/kick).
- Dense features exist (feature24792 fires on 92.7% of tokens). For those, firings above
  the P75 activation are marked <<strong>>, weaker firings <weak>, and a few non-firing
  contrast snippets are shown so the fire/no-fire boundary is describable.
- Position matters (feature14825 fires 2x more often at position 0): windows that begin
  at the first token are prefixed with [START], and snippet headers carry the position.
- Detection scoring is anchored on the FINAL token of a left-context snippet
  ("does the feature fire on the last token?"). Unlike whole-window judgment
  (delphi), this stays well-defined for dense features and matches per-token
  routing decisions, which is what RouterInterp ultimately needs.
"""
import bisect
import random
from collections import Counter
from dataclasses import dataclass, field

import torch

from .data import Corpus, FeatureRecord
from .stats import feature_stats_card

FRAG = "�"  # decoded byte fragments of split multi-byte characters


@dataclass
class SamplerConfig:
    ctx_left: int = 31          # tokens of context to the left of the anchor
    ctx_right: int = 8          # right context (explainer/fuzz views only)
    n_explainer: int = 24       # examples shown to the explainer
    explainer_top_frac: float = 0.33  # fraction of explainer examples from the top pool
    n_test_act: int = 20        # activating detection probes
    n_test_neg: int = 20        # non-activating detection probes
    n_fuzz: int = 20            # fuzzing PAIRS (one correct + one wrongly marked per pair), spread over rank and value bins
    n_bins: int = 10            # bins for stratification (rank and value modes)
    top_pool: int = 200         # strongest firings considered for the top set
    bigram_cap_min: int = 2     # floor on slots per (preceding, fired-on) bigram
    bigram_share_factor: float = 0.5  # dominant bigram gets share*n*factor slots (>= floor)
    max_per_prompt: int = 4     # loose diversity cap per (batch, prompt)
    dense_threshold: float = 0.10    # density above which "dense" handling kicks in
    dense_highlight_q: float = 0.75  # strong/weak split quantile for dense features
    n_dense_negatives: int = 6  # non-firing contrast snippets shown to the explainer (dense only)
    seed: int = 0


@dataclass
class Snippet:
    batch: int
    prompt: int
    pos: int                    # anchor token position in the sequence
    tokens: list                # window token strings (byte fragments merged)
    anchor_offset: int          # anchor index within tokens
    strong: set                 # offsets marked <<...>>
    weak: set                   # offsets marked <...> (dense features only)
    at_start: bool              # window begins at sequence position 0
    max_act: float              # anchor activation (0 for negatives)
    norm_score: int             # 0-10 vs feature max (0 for negatives)

    @property
    def anchor_token(self) -> str:
        return self.tokens[self.anchor_offset]

    def render(self, highlighted: bool, upto_anchor: bool = False,
               show_weak: bool = True, mark_anchor: bool = False) -> str:
        """mark_anchor wraps only the probed token in [[ ]] (detection view): the
        examiner must not have to count tokens to find it — weak examiners judged
        a "to" explanation against whichever "to" appeared anywhere in the snippet."""
        n = self.anchor_offset + 1 if upto_anchor else len(self.tokens)
        marks = {}
        if highlighted:
            for i in self.strong:
                marks[i] = ("<<", ">>")
            if show_weak:
                for i in self.weak:
                    marks.setdefault(i, ("<", ">"))
        if mark_anchor:
            marks[self.anchor_offset] = ("[[", "]]")
        # one mark per token (not merged runs): the explainer needs to see that
        # "a<< lot>><< of>>" is two firing tokens, not one firing phrase
        out = ["[START]"] if self.at_start else []
        for i in range(n):
            m = marks.get(i)
            out.append(self.tokens[i] if m is None else m[0] + self.tokens[i] + m[1])
        return "".join(out)


@dataclass
class FeatureExamples:
    explainer: list
    explainer_negatives: list   # non-firing contrast snippets (dense features only)
    test_act: list
    test_neg: list
    fuzz_correct: list
    fuzz_incorrect: list
    stats: dict = field(default_factory=dict)
    card: str = ""              # statistics card text for the explainer prompt


def _make_snippet(feat: FeatureRecord, corpus: Corpus, b: int, p: int, pos: int,
                  cfg: SamplerConfig, thr: float, act: float, dense: bool) -> Snippet:
    toks = corpus.tokens[b][p]
    lo = max(0, pos - cfg.ctx_left)
    hi = min(len(toks), pos + cfg.ctx_right + 1)
    pacts = feat.acts_by_prompt.get((b, p), {})
    merged = []  # [token, act, is_anchor]
    for i in range(lo, hi):
        t, a, an = toks[i], pacts.get(i, 0.0), i == pos
        if merged and FRAG in t and FRAG in merged[-1][0]:
            merged[-1][0] += t
            merged[-1][1] = max(merged[-1][1], a)
            merged[-1][2] = merged[-1][2] or an
        else:
            merged.append([t, a, an])
    tokens = [m[0] for m in merged]
    anchor = next(i for i, m in enumerate(merged) if m[2])
    strong = {i for i, m in enumerate(merged) if m[1] > thr}
    if act > 0:
        strong.add(anchor)  # every activating snippet carries at least its anchor mark
    weak = ({i for i, m in enumerate(merged) if 0 < m[1] <= thr} - strong) if dense else set()
    norm = int(round(10 * act / feat.max_act)) if feat.max_act > 0 else 0
    return Snippet(batch=b, prompt=p, pos=pos, tokens=tokens, anchor_offset=anchor,
                   strong=strong, weak=weak, at_start=(lo == 0), max_act=act, norm_score=norm)


def _bigram(corpus: Corpus, row) -> tuple:
    toks = corpus.tokens[row[0]][row[1]]
    q = row[2]
    return (toks[q - 1] if q > 0 else "[START]", toks[q])


def _quotas(corpus: Corpus, rows: list, n_slots: int, cfg: SamplerConfig) -> dict:
    """Proportional-with-floor quotas on three keys — the (preceding, fired-on) bigram,
    the preceding token alone, and the fired-on token alone. A key holding share s of
    the candidate pool gets max(floor, s*n*factor) slots; when every candidate is
    exhausted under quota the overflow pass ignores them, so single-token features
    ("to", "after a") still fill their set."""
    bigrams = [_bigram(corpus, r) for r in rows]
    total = max(1, len(bigrams))
    out = {}
    for name, keys in (("bigram", bigrams), ("prev", [b[0] for b in bigrams]),
                       ("cur", [b[1] for b in bigrams])):
        out[name] = {k: max(cfg.bigram_cap_min, round(c / total * n_slots * cfg.bigram_share_factor))
                     for k, c in Counter(keys).items()}
    return out


class _Picker:
    """Shared `used` set keeps all sample sets disjoint; per-set counters enforce
    the diversity quotas and the loose per-prompt cap."""

    def __init__(self, corpus: Corpus, cfg: SamplerConfig):
        self.corpus, self.cfg = corpus, cfg
        self.used = set()

    @staticmethod
    def new_counts() -> dict:
        return {"bigram": Counter(), "prev": Counter(), "cur": Counter(), "prompt": Counter()}

    def take(self, row, quotas: dict, counts: dict, overflow: bool = False) -> bool:
        key = (row[0], row[1], row[2])
        if key in self.used:
            return False
        bg = _bigram(self.corpus, row)
        keys = {"bigram": bg, "prev": bg[0], "cur": bg[1]}
        pk = (row[0], row[1])
        if not overflow:
            for name, k in keys.items():
                if counts[name][k] >= quotas[name].get(k, self.cfg.bigram_cap_min):
                    return False
            if counts["prompt"][pk] >= self.cfg.max_per_prompt:
                return False
        self.used.add(key)
        for name, k in keys.items():
            counts[name][k] += 1
        counts["prompt"][pk] += 1
        return True


def _bins(rows: list, n_bins: int, mode: str) -> list:
    """Index ranges into activation-ascending rows: equal population ('rank') or
    equal activation width ('value')."""
    N = len(rows)
    if mode == "rank":
        return [((N * i) // n_bins, (N * (i + 1)) // n_bins) for i in range(n_bins)]
    acts = [r[3] for r in rows]
    lo_v, hi_v = acts[0], acts[-1]
    edges = [lo_v + (hi_v - lo_v) * i / n_bins for i in range(n_bins + 1)]
    idx = [bisect.bisect_left(acts, e) for e in edges]
    idx[-1] = N
    return [(idx[i], idx[i + 1]) for i in range(n_bins)]


def _stratified(rows: list, n_want: int, modes: tuple, picker: _Picker, quotas: dict,
                cfg: SamplerConfig, rng: random.Random) -> list:
    """Round-robin over bins (half the slots per mode) so empty value-bins of
    heavy-tailed features simply yield their turn to populated ones."""
    picked = []
    counts = picker.new_counts()
    per_mode = [n_want // len(modes)] * len(modes)
    per_mode[-1] += n_want - sum(per_mode)
    for mode, n_mode in zip(modes, per_mode):
        cands = []
        for lo, hi in _bins(rows, cfg.n_bins, mode):
            c = list(range(lo, hi))
            rng.shuffle(c)
            cands.append(c)
        got = 0
        for overflow in (False, True):
            progress = True
            while got < n_mode and progress:
                progress = False
                for c in cands:
                    if got >= n_mode:
                        break
                    while c:
                        row = rows[c.pop()]
                        if picker.take(row, quotas, counts, overflow):
                            picked.append(row)
                            got += 1
                            progress = True
                            break
    return picked


def _stratified_pairs(rows: list, n_pairs: int, modes: tuple, picker: _Picker, quotas: dict,
                      cfg: SamplerConfig, rng: random.Random) -> list:
    """Pairs (a, b) drawn from the SAME bin, so that the two fuzzing classes (a keeps
    its true marks, b gets deliberately wrong marks) have identical strength
    distributions bin by bin. Pairs are spread over the bins of each mode; a bin that
    cannot supply two eligible rows yields its turn to the next bin."""
    pairs = []
    per_mode = [n_pairs // len(modes)] * len(modes)
    per_mode[-1] += n_pairs - sum(per_mode)
    for mode, n_mode in zip(modes, per_mode):
        bins = _bins(rows, cfg.n_bins, mode)
        cands = []
        for lo, hi in bins:
            c = list(range(lo, hi))
            rng.shuffle(c)
            cands.append(c)
        if n_mode < len(bins):  # evenly spaced bins over the strength range
            order = [round(i * (len(bins) - 1) / max(1, n_mode - 1)) for i in range(n_mode)]
        else:
            order = list(range(len(bins))) * ((n_mode + len(bins) - 1) // len(bins))
        counts = picker.new_counts()
        got = 0
        for overflow in (False, True):
            for bi in order:
                if got >= n_mode:
                    break
                pair = []
                c = cands[bi]
                while c and len(pair) < 2:
                    row = rows[c.pop()]
                    if picker.take(row, quotas, counts, overflow):
                        pair.append(row)
                if len(pair) == 2:
                    pairs.append(tuple(pair))
                    got += 1
                else:  # bin exhausted: release a lone pick so it is not wasted
                    for row in pair:
                        picker.used.discard((row[0], row[1], row[2]))
    return pairs


def _negatives(feat: FeatureRecord, corpus: Corpus, n: int, picker: _Picker,
               cfg: SamplerConfig, rng: random.Random) -> list:
    out, tries = [], 0
    while len(out) < n and tries < 200000:
        tries += 1
        b = rng.randrange(corpus.n_batches)
        p = rng.randrange(corpus.n_prompts)
        pos = rng.randrange(corpus.n_tokens)
        if feat.fires_at(corpus, b, p, pos) or (b, p, pos) in picker.used:
            continue
        picker.used.add((b, p, pos))
        out.append((b, p, pos, 0.0))
    if len(out) < n:
        raise RuntimeError(f"feature {feat.feature_id}: could not sample {n} negatives "
                           f"(density={feat.density(corpus):.3f})")
    return out


def build_examples(feat: FeatureRecord, corpus: Corpus, cfg: SamplerConfig) -> FeatureExamples:
    rng = random.Random(cfg.seed * 100003 + feat.feature_id)
    feat.build_index(corpus)
    density = feat.density(corpus)
    dense = density >= cfg.dense_threshold
    acts = feat.acts
    thr = float(torch.quantile(acts, cfg.dense_highlight_q)) if dense else 0.0

    order = torch.argsort(acts)
    rec = feat.records[order]
    rows = [(int(r[0]), int(r[1]), int(r[2]), float(r[3])) for r in
            rec[:: max(1, len(rec) // 200000)]]  # subsample huge features, keeps order
    if rows[-1][3] < float(acts.max()):  # keep the true maximum in the pool
        r = feat.records[int(torch.argmax(acts))]
        rows.append((int(r[0]), int(r[1]), int(r[2]), float(r[3])))
    picker = _Picker(corpus, cfg)

    # --- explainer: proportional-diversity top set + mixed stratified set ---
    n_top = max(1, int(cfg.n_explainer * cfg.explainer_top_frac))
    pool = rows[-cfg.top_pool:][::-1]  # strongest first
    top_quotas = _quotas(corpus, pool, n_top, cfg)
    top_rows, top_counts = [], picker.new_counts()
    for overflow in (False, True):
        for r in pool:
            if len(top_rows) >= n_top:
                break
            if picker.take(r, top_quotas, top_counts, overflow):
                top_rows.append(r)
    all_quotas = _quotas(corpus, rows[:: max(1, len(rows) // 5000)], cfg.n_explainer, cfg)
    strat_rows = _stratified(rows, cfg.n_explainer - len(top_rows), ("rank", "value"),
                             picker, all_quotas, cfg, rng)
    explainer_rows = sorted(top_rows + strat_rows, key=lambda r: -r[3])

    # --- held-out probes ---
    test_rows = _stratified(rows, cfg.n_test_act, ("rank", "value"), picker, all_quotas, cfg, rng)
    fuzz_pairs = _stratified_pairs(rows, cfg.n_fuzz, ("rank", "value"), picker, all_quotas, cfg, rng)
    neg_rows = _negatives(feat, corpus, cfg.n_test_neg, picker, cfg, rng)
    contrast_rows = _negatives(feat, corpus, cfg.n_dense_negatives, picker, cfg, rng) if dense else []

    mk = lambda r: _make_snippet(feat, corpus, r[0], r[1], r[2], cfg, thr, r[3], dense)
    fuzz_correct = [mk(a) for a, _ in fuzz_pairs]
    fuzz_incorrect = []
    for _, r in fuzz_pairs:
        s = mk(r)
        k = max(1, len(s.strong))
        non_firing = [i for i in range(len(s.tokens)) if i not in s.strong and i not in s.weak]
        if len(non_firing) < k:  # dense feature: fall back to the weakest positions
            pacts = feat.acts_by_prompt.get((s.batch, s.prompt), {})
            lo_off = s.pos - s.anchor_offset
            cand = sorted(range(len(s.tokens)), key=lambda i: pacts.get(lo_off + i, 0.0))
            non_firing = [i for i in cand if i not in s.strong] or cand
        s.strong = set(rng.sample(non_firing, min(k, len(non_firing))))
        s.weak = set()
        fuzz_incorrect.append(s)

    card, card_data = feature_stats_card(feat, corpus, seed=cfg.seed)
    stats = {
        "density": density,
        "n_firings": feat.n_firings,
        "max_act": feat.max_act,
        "act_quantiles": {q: float(torch.quantile(acts, q)) for q in (0.25, 0.5, 0.75, 0.9, 0.99)},
        "highlight_threshold": thr,
        "dense": dense,
        "card_data": card_data,
    }
    return FeatureExamples(
        explainer=[mk(r) for r in explainer_rows],
        explainer_negatives=[mk(r) for r in contrast_rows],
        test_act=[mk(r) for r in test_rows],
        test_neg=[mk(r) for r in neg_rows],
        fuzz_correct=fuzz_correct, fuzz_incorrect=fuzz_incorrect,
        stats=stats, card=card,
    )
