"""Corpus-wide statistics card for a feature, shown to the explainer.

Computed over ALL firings, not the ~24 sampled snippets, so aggregate patterns that a
small sample hides (phrasal-verb particles, deity references, early-position firing)
become visible at zero LLM cost. Calibration example: feature25337's 24 snippets were
dominated by "cut"/"kick", while the top-2000 table reads cut/pick/click/jump/dump/break
-> up/on/out/in — the phrasal-verb-particle pattern is obvious from the table alone.
"""
import collections
import random

import torch

from .data import Corpus, FeatureRecord

POS_BUCKETS = [(0, 1), (1, 4), (4, 16), (16, 64), (64, 10 ** 9)]


def _fmt_tok(t: str) -> str:
    return '"' + t.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"') + '"'


def _top(counter: collections.Counter, k: int) -> str:
    return ", ".join(f"{_fmt_tok(t)} {c}" for t, c in counter.most_common(k))


def feature_stats_card(feat: FeatureRecord, corpus: Corpus, n_strong: int = 2000,
                       n_random: int = 5000, top_k: int = 8, seed: int = 0) -> tuple[str, dict]:
    rec = feat.records
    acts = rec[:, 3]
    pos = rec[:, 2].long()
    T = corpus.n_tokens
    n_seq = corpus.n_batches * corpus.n_prompts
    density = feat.density(corpus)

    profile = []
    for lo, hi in POS_BUCKETS:
        if lo >= T:
            break
        hi = min(hi, T)
        m = (pos >= lo) & (pos < hi)
        n = int(m.sum())
        profile.append({"lo": lo, "hi": hi - 1, "rate": n / (n_seq * (hi - lo)),
                        "mean_act": float(acts[m].mean()) if n else 0.0})

    def counters(idxs):
        cur, prev = collections.Counter(), collections.Counter()
        for i in idxs:
            b, p, q = int(rec[i, 0]), int(rec[i, 1]), int(rec[i, 2])
            toks = corpus.tokens[b][p]
            cur[toks[q]] += 1
            prev[toks[q - 1] if q > 0 else "[START]"] += 1
        return cur, prev

    k = min(n_strong, len(rec))
    strong_idx = torch.topk(acts, k).indices.tolist()
    rng = random.Random(seed)
    rand_idx = rng.sample(range(len(rec)), min(n_random, len(rec)))
    s_cur, s_prev = counters(strong_idx)
    r_cur, r_prev = counters(rand_idx)
    top1_share = s_cur.most_common(1)[0][1] / k if k else 0.0

    def label(b):
        return f"{b['lo']}" if b["lo"] == b["hi"] else f"{b['lo']}-{b['hi']}"

    lines = [
        f"FEATURE STATISTICS (over all {feat.n_firings:,} firings in a "
        f"{corpus.total_tokens:,}-token corpus of {n_seq:,} sequences x {T} tokens)",
        f"- fires on {density:.1%} of all tokens; max activation {feat.max_act:.2f}",
        "- firing rate by token position: "
        + " | ".join(f"pos {label(b)}: {b['rate']:.1%}" for b in profile),
        "- mean activation by token position: "
        + " | ".join(f"{b['mean_act']:.2f}" for b in profile),
        f"- strongest {k:,} firings — fired-on token: {_top(s_cur, top_k)}",
        f"    (the single most common fired-on token covers {top1_share:.0%} of them)",
        f"- strongest {k:,} firings — preceding token: {_top(s_prev, top_k)}",
        f"- random {len(rand_idx):,} firings — fired-on token: {_top(r_cur, top_k)}",
        f"- random {len(rand_idx):,} firings — preceding token: {_top(r_prev, top_k)}",
        '  ("[START]" as preceding token = the fired-on token is the first of its sequence)',
    ]
    data = {"density": density, "position_profile": profile,
            "strong_fired_on": s_cur.most_common(top_k), "strong_preceding": s_prev.most_common(top_k),
            "random_fired_on": r_cur.most_common(top_k), "random_preceding": r_prev.most_common(top_k),
            "strong_top1_share": top1_share}
    return "\n".join(lines), data
