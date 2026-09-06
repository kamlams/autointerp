"""Corpus-wide statistics card for a feature, shown to the explainer.

Computed over ALL firings, not the ~24 sampled snippets, so aggregate patterns that a
small sample hides (phrasal-verb particles, deity references, early-position firing)
become visible at zero LLM cost. Calibration example: feature25337's 24 snippets were
dominated by "cut"/"kick", while the top-2000 table reads cut/pick/click/jump/dump/break
-> up/on/out/in.

Two kinds of numbers per token, because they answer different questions:
- share among firings, P(token = X | fire): what is common among firings;
- conditional rate, P(fire | token = X), i.e. how often ANY corpus occurrence of X is
  (followed by) a firing: how reliably X triggers the feature. " kick" is followed by a
  firing 98% of the time (mechanism); " the" appears often among firings but fires
  after only ~5% of its occurrences (background). Sampling alone cannot recover the
  second kind — that is what the card is for.
"""
import collections

import torch

from .data import Corpus, FeatureRecord

POS_BUCKETS = [(0, 1), (1, 4), (4, 16), (16, 64), (64, 10 ** 9)]


def _fmt_tok(t: str) -> str:
    return '"' + t.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"') + '"'


def _fmt_n(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 10000 else f"{n:,}"


def _top_with_rates(counter: collections.Counter, denom: dict, k: int, what: str) -> str:
    parts = []
    for t, c in counter.most_common(k):
        n = denom.get(t, 0)
        rate = f"{c / n:.0%}" if n else "?"
        parts.append(f"{_fmt_tok(t)} {c} ({what} {rate} of its {_fmt_n(n)} occurrences)")
    return ", ".join(parts)


def feature_stats_card(feat: FeatureRecord, corpus: Corpus, n_strong: int = 2000,
                       top_k: int = 8, seed: int = 0) -> tuple[str, dict]:
    rec = feat.records
    acts = rec[:, 3]
    pos = rec[:, 2].long()
    T = corpus.n_tokens
    n_seq = corpus.n_batches * corpus.n_prompts
    density = feat.density(corpus)
    count_all, count_prev = corpus.token_counts()

    profile = []
    for lo, hi in POS_BUCKETS:
        if lo >= T:
            break
        hi = min(hi, T)
        m = (pos >= lo) & (pos < hi)
        n = int(m.sum())
        profile.append({"lo": lo, "hi": hi - 1, "rate": n / (n_seq * (hi - lo)),
                        "mean_act": float(acts[m].mean()) if n else 0.0})

    # token identities for every firing (one pass, ~1 s per million rows)
    b_l, p_l, q_l = rec[:, 0].long().tolist(), rec[:, 1].long().tolist(), rec[:, 2].long().tolist()
    cur_all, prev_all = collections.Counter(), collections.Counter()
    cur_tok, prev_tok = [], []
    for b, p, q in zip(b_l, p_l, q_l):
        toks = corpus.tokens[b][p]
        c = toks[q]
        pr = toks[q - 1] if q > 0 else "[START]"
        cur_tok.append(c)
        prev_tok.append(pr)
        cur_all[c] += 1
        prev_all[pr] += 1
    count_prev = dict(count_prev)
    count_prev["[START]"] = n_seq  # every sequence has exactly one first token

    k = min(n_strong, len(rec))
    strong_idx = torch.topk(acts, k).indices.tolist()
    cur_strong = collections.Counter(cur_tok[i] for i in strong_idx)
    prev_strong = collections.Counter(prev_tok[i] for i in strong_idx)
    top1_share = cur_strong.most_common(1)[0][1] / k if k else 0.0

    # conditional rates use ALL firings in the numerator, corpus occurrences in the denominator
    def rate_line(counter_subset, counter_all, denom, what):
        parts = []
        for t, c in counter_subset.most_common(top_k):
            n = denom.get(t, 0)
            r = f"{counter_all[t] / n:.0%}" if n else "?"
            parts.append(f"{_fmt_tok(t)} {c} ({what} {r} of its {_fmt_n(n)} occurrences)")
        return ", ".join(parts)

    def label(bk):
        return f"{bk['lo']}" if bk["lo"] == bk["hi"] else f"{bk['lo']}-{bk['hi']}"

    lines = [
        f"FEATURE STATISTICS (over all {feat.n_firings:,} firings in a "
        f"{corpus.total_tokens:,}-token corpus of {n_seq:,} sequences x {T} tokens)",
        f"- fires on {density:.1%} of all tokens (baseline: a random token fires with "
        f"probability {density:.1%}); max activation {feat.max_act:.2f}",
        "- firing rate by token position: "
        + " | ".join(f"pos {label(bk)}: {bk['rate']:.1%}" for bk in profile),
        "- mean activation by token position: "
        + " | ".join(f"{bk['mean_act']:.2f}" for bk in profile),
        "- for each token below, the parenthesis gives how often ANY occurrence of that token in the "
        "corpus fires (fired-on token) or is followed by a firing (preceding token): a high rate "
        "means the token reliably triggers the feature, a low rate means it is merely frequent",
        f"- strongest {k:,} firings — fired-on token: "
        + rate_line(cur_strong, cur_all, count_all, "fires on"),
        f"    (the single most common fired-on token covers {top1_share:.0%} of them)",
        f"- strongest {k:,} firings — preceding token: "
        + rate_line(prev_strong, prev_all, count_prev, "followed by a firing"),
        "- all firings — fired-on token: "
        + rate_line(cur_all, cur_all, count_all, "fires on"),
        "- all firings — preceding token: "
        + rate_line(prev_all, prev_all, count_prev, "followed by a firing"),
        '  ("[START]" as preceding token = the fired-on token is the first of its sequence)',
    ]
    data = {"density": density, "position_profile": profile,
            "strong_fired_on": cur_strong.most_common(top_k),
            "strong_preceding": prev_strong.most_common(top_k),
            "all_fired_on": cur_all.most_common(top_k), "all_preceding": prev_all.most_common(top_k),
            "cond_rate_fired_on": {t: cur_all[t] / count_all[t] for t, _ in
                                   (cur_strong.most_common(top_k) + cur_all.most_common(top_k))
                                   if count_all.get(t)},
            "cond_rate_preceding": {t: prev_all[t] / count_prev[t] for t, _ in
                                    (prev_strong.most_common(top_k) + prev_all.most_common(top_k))
                                    if count_prev.get(t)},
            "strong_top1_share": top1_share}
    return "\n".join(lines), data
