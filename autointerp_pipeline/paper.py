"""Published autointerp protocol (Paulo, Mallen, Juang, Belrose 2024, arXiv 2410.13928), following the
reference implementation EleutherAI/delphi, plus the switches the ablation arms need.

The unit is a WINDOW: every corpus sequence is cut into non-overlapping `ctx_len`-token windows; a window
is an activating example if the latent fires anywhere inside it. The explainer sees highlighted windows
(with activation values), detection classifies plain windows, fuzzing classifies highlighted windows.
Prompts are delphi's, verbatim (`delphi_prompts/`). Declared deviations from the reference are listed in
results/paper_protocol_260920/README.md; the one that matters here: test windows are drawn first from a
seed that does not depend on the arm, and explainer windows come from the rest, so every arm of a latent
is scored on the same, unseen test set.
"""
import ast
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field

from .data import Corpus, FeatureRecord
from .delphi_prompts import detection as det_prompts
from .delphi_prompts import explainer as expl_prompts
from .delphi_prompts import fuzz as fuzz_prompts
from .llm import LLMClient, map_concurrent

class NotScoreable(RuntimeError):
    """Too few activating or non-activating windows for this protocol to say anything."""


THRESHOLD = 0.3   # delphi: highlight tokens above 0.3 x the example's max activation
N_SHOWN = 5       # paper App. D: five shuffled examples per scorer prompt


@dataclass
class PaperConfig:
    ctx_len: int = 32
    n_train: int = 40
    n_test: int = 100
    n_quantiles: int = 10
    train_type: str = "random"      # random | quantiles | top | bins_dedup
    test_high_only: bool = False    # probe only windows whose max is >= strong_floor x the latent
    match_mark_counts: bool = False # give each negative as many fake marks as a positive really has
    random_mark_pos: bool = False   # scatter the fake marks instead of delphi's fixed positions
    single_mark: bool = False       # one marked token per probe, and negatives of both kinds
    strong_floor: float = 0.30      # of the latent's maximum activation
    show_activations: bool = True
    stats_card: bool = False
    drop_pos0: bool = False
    seed: int = 22                  # train sampling seed (delphi uses 22); the test seed is fixed


@dataclass
class Window:
    batch: int
    prompt: int
    w: int
    tokens: list
    acts: list                      # one activation per token, 0.0 where silent
    quantile: int = -1              # decile of the window max among activating windows, 0 = strongest
    norm: list = field(default_factory=list)

    @property
    def key(self):
        return (self.batch, self.prompt, self.w)

    @property
    def max_act(self) -> float:
        return max(self.acts)


def build_windows(feat: FeatureRecord, corpus: Corpus, cfg: PaperConfig) -> list:
    """All activating windows of a latent, strongest first (delphi sorts by window max)."""
    L = cfg.ctx_len
    by_window = {}
    for b, p, pos, a in feat.records.tolist():
        b, p, pos = int(b), int(p), int(pos)
        if cfg.drop_pos0 and pos == 0:
            continue
        if pos >= (corpus.n_tokens // L) * L:
            continue
        by_window.setdefault((b, p, pos // L), {})[pos % L] = a
    out = []
    for (b, p, w), hits in by_window.items():
        acts = [0.0] * L
        for i, a in hits.items():
            acts[i] = a
        out.append(Window(b, p, w, corpus.tokens[b][p][w * L:(w + 1) * L], acts))
    out.sort(key=lambda x: (-x.max_act, x.key))
    return out


def _split_quantiles(examples: list, n_quantiles: int, n_samples: int, rng: random.Random) -> list:
    """delphi `split_quantiles`: n_samples // n_quantiles from each equal-count slice of the sorted list."""
    size = len(examples) // n_quantiles
    per = n_samples // n_quantiles
    picked = []
    for q in range(n_quantiles):
        chunk = examples[q * size:(q + 1) * size]
        take = chunk if len(chunk) < per else rng.sample(chunk, per)
        for e in take:
            e.quantile = q
        picked += take
    return picked


def _bins_dedup(examples: list, n: int, n_bins: int, rng: random.Random) -> list:
    """Arm A3: half of the examples from rank deciles, half from equal-width bins of the window max,
    with a cap on how often one fired-on token or one (preceding, fired-on) bigram may appear."""
    def key_of(e):
        i = max(range(len(e.acts)), key=e.acts.__getitem__)
        return (e.tokens[i - 1] if i else "[START]", e.tokens[i])

    keys = [key_of(e) for e in examples]
    total = max(1, len(keys))
    cap_bi = {k: max(2, round(c / total * n * 0.5)) for k, c in Counter(keys).items()}
    cap_tok = {k: max(2, round(c / total * n * 0.5)) for k, c in Counter(k[1] for k in keys).items()}
    lo, hi = examples[-1].max_act, examples[0].max_act
    width = (hi - lo) / n_bins or 1.0
    size = max(1, len(examples) // n_bins)
    rank_bins = [list(range(q * size, min(len(examples), (q + 1) * size))) for q in range(n_bins)]
    value_bins = [[] for _ in range(n_bins)]
    for i, e in enumerate(examples):
        value_bins[min(n_bins - 1, int((hi - e.max_act) / width))].append(i)
    used, n_bi, n_tok, picked = set(), Counter(), Counter(), []
    for strict in (True, False):
        for bins, quota in ((rank_bins, n // 2), (value_bins, n - n // 2)):
            have = sum(1 for _, tag in picked if tag is bins)
            pools = [rng.sample(b, len(b)) for b in bins]
            while have < quota and any(pools):
                for pool in pools:
                    while pool and have < quota:
                        i = pool.pop()
                        k = keys[i]
                        if i in used or (strict and (n_bi[k] >= cap_bi[k] or n_tok[k[1]] >= cap_tok[k[1]])):
                            continue
                        used.add(i); n_bi[k] += 1; n_tok[k[1]] += 1
                        picked.append((examples[i], bins)); have += 1
                        break
    return [e for e, _ in picked]


def sample(feat: FeatureRecord, corpus: Corpus, cfg: PaperConfig) -> tuple[list, list, list]:
    """(train, test, non_activating). Test and non-activating sets depend on the latent only."""
    windows = build_windows(feat, corpus, cfg)
    if len(windows) < 12:
        raise NotScoreable(f"feature {feat.feature_id}: only {len(windows)} activating windows")
    n_train = cfg.n_train if len(windows) >= cfg.n_train + cfg.n_test else min(cfg.n_train, len(windows) // 3)
    n_test = min(cfg.n_test, len(windows) - n_train)
    test_rng = random.Random(1_000_003 * 22 + feat.feature_id)
    if cfg.test_high_only:
        # TopK forces every token to fill k slots, so the weak tail of a latent is largely filler
        # an explanation cannot be expected to predict. Probe only where the latent really fires:
        # windows above strong_floor x its maximum, sampled at random among those.
        floor = cfg.strong_floor * windows[0].max_act
        pool = [e for e in windows if e.max_act >= floor] or windows[:n_test]
        n_test = min(n_test, len(pool))
        test = test_rng.sample(pool, n_test)
        for e in test:
            e.quantile = min(9, int((windows[0].max_act - e.max_act) / max(1e-9, windows[0].max_act - floor) * 10))
    else:
        test = _split_quantiles(windows, cfg.n_quantiles, n_test, test_rng)
    taken = {e.key for e in test}
    rest = [e for e in windows if e.key not in taken]
    rng = random.Random(1_000_003 * cfg.seed + feat.feature_id)
    if cfg.train_type == "random":
        train = rng.sample(rest, min(n_train, len(rest)))
    elif cfg.train_type == "quantiles":
        train = _split_quantiles(rest, cfg.n_quantiles, n_train, rng)
    elif cfg.train_type == "top":
        train = rest[:n_train]
    elif cfg.train_type == "bins_dedup":
        train = _bins_dedup(rest, min(n_train, len(rest)), cfg.n_quantiles, rng)
    else:
        raise ValueError(cfg.train_type)
    top = max(windows[0].max_act, 1e-6)
    for e in train + test:
        e.norm = [min(10, max(0, math.ceil(a * 10 / top))) for a in e.acts]

    # Non-activating windows: the complement of the activating set, enumerated rather than
    # rejection-sampled -- a latent dense enough to fire in almost every window (L0 latent 2565
    # fires on 97.8% of tokens) would make rejection sampling spin forever.
    L, n_w = cfg.ctx_len, corpus.n_tokens // cfg.ctx_len
    active = {e.key for e in windows}
    free = [(b, p, w) for b in range(corpus.n_batches) for p in range(corpus.n_prompts)
            for w in range(n_w) if (b, p, w) not in active]
    if len(free) < len(test):
        raise NotScoreable(f"feature {feat.feature_id}: {len(free)} non-activating windows for "
                           f"{len(test)} probes (fires in {len(active)} of {corpus.n_batches * corpus.n_prompts * n_w} "
                           f"windows) -- no contrast to score against")
    negatives = [Window(b, p, w, corpus.tokens[b][p][w * L:(w + 1) * L], [0.0] * L)
                 for b, p, w in test_rng.sample(free, len(test))]
    return train, test, negatives


def highlight(tokens: list, marked) -> str:
    """delphi `_highlight`: consecutive marked tokens share one << >> pair."""
    out, i = "", 0
    while i < len(tokens):
        if marked(i):
            out += "<<"
            while i < len(tokens) and marked(i):
                out += tokens[i]
                i += 1
            out += ">>"
        else:
            out += tokens[i]
            i += 1
    return out


def explainer_messages(train: list, cfg: PaperConfig, card: str | None = None) -> list:
    lines = []
    for n, e in enumerate(train, 1):
        thr = e.max_act * THRESHOLD
        lines.append(f"Example {n}: " + highlight(e.tokens, lambda i: e.acts[i] > thr))
        if cfg.show_activations:
            shown = [(t, v) for t, a, v in zip(e.tokens, e.acts, e.norm) if a > thr][:11]
            lines.append("Activations: " + "".join(f'("{t}" : {int(v)}), ' for t, v in shown))
    messages = expl_prompts.system(cot=False)
    for k in (1, 2, 3):
        q, a = expl_prompts.example(k, activations=cfg.show_activations, cot=False)
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    body = "\n" + "\n".join(lines) + "\n"
    if card:
        body = ("\nStatistics over ALL activations of this latent in the corpus (the examples below are a "
                "small sample of them):\n" + card + "\n" + body)
    return messages + [{"role": "user", "content": body}]


def explain(client: LLMClient, train: list, cfg: PaperConfig, card: str | None, max_tokens: int) -> dict:
    messages = explainer_messages(train, cfg, card)
    reply = client.chat(messages, max_tokens=max_tokens, tag="explain")
    found = re.findall(r"\[EXPLANATION\]:\s*(.+)", reply)   # delphi: the LAST marker wins
    return {"explanation": found[-1].strip() if found else "Explanation could not be parsed.",
            "parsed": bool(found), "raw": reply, "prompt": messages[-1]["content"]}


def _parse_list(reply: str, n: int):
    m = re.search(r"\[[^\]]*\]", reply, flags=re.DOTALL)
    if not m:
        return None
    try:
        arr = ast.literal_eval(m.group(0))
        return [1 if int(x) == 1 else 0 for x in arr] if len(arr) == n else None
    except (ValueError, SyntaxError, TypeError):
        return None


def _classify(client: LLMClient, prompt_fn, explanation: str, samples: list, workers: int, tag: str,
              max_tokens: int = 2048) -> dict:
    """samples: (text, label, quantile). delphi shuffles with seed 42 and batches N_SHOWN per prompt."""
    samples = samples[:]
    random.Random(42).shuffle(samples)
    batches = [samples[i:i + N_SHOWN] for i in range(0, len(samples), N_SHOWN)]

    def run(batch):
        text = "\n".join(f"Example {i}: {t}" for i, (t, _, _) in enumerate(batch))
        try:
            reply = client.chat(prompt_fn(text, explanation), max_tokens=max_tokens, tag=tag)
        except Exception:  # noqa: BLE001  a failed batch is excluded and counted, as in scorer.py
            return [None] * len(batch)
        return _parse_list(reply, len(batch)) or [None] * len(batch)

    preds = [p for out in map_concurrent(run, batches, workers) for p in out]
    labels = [y for _, y, _ in samples]
    ok = [(p, y) for p, y in zip(preds, labels) if p is not None]
    pos = [p for p, y in ok if y == 1]
    neg = [p for p, y in ok if y == 0]
    tpr = sum(pos) / len(pos) if pos else float("nan")
    tnr = sum(1 - p for p in neg) / len(neg) if neg else float("nan")
    by_q = {}
    for (_, y, q), p in zip(samples, preds):
        if y == 1 and p is not None:
            by_q.setdefault(q, []).append(p)
    return {"accuracy": sum(p == y for p, y in ok) / len(ok) if ok else float("nan"),
            "balanced_acc": (tpr + tnr) / 2, "tpr": tpr, "tnr": tnr,
            "n_pos": len(pos), "n_neg": len(neg), "n_parse_errors": len(preds) - len(ok),
            "tpr_by_quantile": {q: sum(v) / len(v) for q, v in sorted(by_q.items())},
            "preds": preds, "labels": labels}


def detection(client: LLMClient, explanation: str, test: list, negatives: list, workers: int,
              max_tokens: int = 2048) -> dict:
    samples = [("".join(e.tokens), 1, e.quantile) for e in test] + [("".join(e.tokens), 0, -1) for e in negatives]
    return _classify(client, det_prompts.prompt, explanation, samples, workers, "detect", max_tokens)


def fuzzing(client: LLMClient, explanation: str, test: list, negatives: list, workers: int,
            max_tokens: int = 2048, strong_abs: float | None = None,
            match_counts: bool = False, random_pos: bool = False) -> dict:
    """delphi `fuzz_type="default"`: true highlights on activating windows; on non-activating windows
    n_incorrect random tokens, n_incorrect = ceil(mean number of firing tokens per test window). The
    reference reseeds with 22 for every example and always includes position len - len // 4.

    `random_pos` scatters the fake marks. delphi pins the first one at index len - len//4 -- always
    satisfied in a window with no firing at all -- and reseeds `random.Random(22)` per example, so
    every negative of a latent carries the SAME mark pattern (e.g. {24, 31} for a 32-token window)
    while only 8% of positives have a mark at index 24. Mark position alone separates the classes,
    which is why a wrong explanation still reaches TNR 0.818. This flag prices that cue; it is a
    control, not a proposed fix.

    `match_counts` draws each negative's number of fake marks from the positives' own counts
    instead of delphi's single ceil(mean non-zero count). It is the one-variable control for the
    cue below: with the rest of the exam untouched, a drop in TNR means the examiner was following
    mark counts rather than the explanation.

    `strong_abs` replaces delphi's per-window threshold (0.3 x the window's own max) by one absolute
    cut for the whole latent. Two things follow, and they are why the strong-firing exam sets it:
    a weak window no longer has its noise promoted to "strong" just because nothing else fires in it,
    and n_incorrect is then counted above the same cut. Under delphi's rule the fake marks are drawn
    from ALL non-zero tokens while the true marks come only from those above 30%, so negatives carry
    systematically more marks (5.58 vs 4.30 over these 95 latents, in the same direction for every
    one of them) -- a cue a scorer can follow without reading the explanation at all."""
    thr_of = (lambda e: strong_abs) if strong_abs is not None else (lambda e: e.max_act * THRESHOLD)
    true_counts = [max(1, sum(1 for a in e.acts if a >= thr_of(e))) for e in test]
    samples = []
    for e in test:
        thr = thr_of(e)
        samples.append((highlight(e.tokens, lambda i: e.acts[i] >= thr), 1, e.quantile))
    # delphi gives every negative the same ceil(mean non-zero count) marks; with an absolute cut we
    # instead draw each negative's mark count from the positives' own counts, so the two classes are
    # indistinguishable by how many marks they carry.
    n_incorrect = math.ceil(sum(sum(1 for a in e.acts if a > 0) for e in test) / len(test))
    cnt_rng = random.Random(23)
    for e in negatives:
        L = len(e.tokens)
        n = cnt_rng.choice(true_counts) if (match_counts or strong_abs is not None) else n_incorrect
        if random_pos:
            rng = random.Random(hash(e.key) & 0xFFFFFFFF)
            marks = set(rng.sample(range(L), min(n, L)))
        else:
            anchor = L - L // 4
            rng = random.Random(22)
            marks = {anchor} | set(rng.sample([i for i in range(L) if i != anchor], min(n, L) - 1))
        samples.append((highlight(e.tokens, lambda i: i in marks), 0, -1))
    out = _classify(client, fuzz_prompts.prompt, explanation, samples, workers, "fuzz", max_tokens)
    out["n_incorrect"] = n_incorrect
    return out


def fuzzing_single(client: LLMClient, explanation: str, test: list, negatives: list,
                   windows: list, workers: int, max_tokens: int, seed: int) -> dict:
    """One marked token per probe, and two kinds of negative.

    delphi marks every token of a window that is above 30% of that window's maximum, then asks
    whether *every* mark is right. The explanation, though, is a per-token rule, so the question is
    a conjunction of k per-token judgements: a positive is accepted only if all k marks survive,
    while a negative is rejected as soon as one mark looks wrong. That is why TPR sits at 0.52 and
    TNR at 0.88, and why the hit rate falls with k (0.635 at one mark, 0.336 above six) -- dense
    latents, which put 7.8 marks in a window, are penalised for their density rather than for a bad
    explanation. One mark per probe makes the question the same shape as the explanation, equalises
    difficulty across densities, and removes the mark-count cue for free.

    Negatives come in two kinds, in equal numbers, and are reported separately:
      A  a token of a window where the latent never fires  -- is this text related at all?
      B  a silent token of a window where it does fire     -- which token exactly?
    """
    rng = random.Random(seed + 7)
    samples = []
    for e in test:
        i = max(range(len(e.acts)), key=e.acts.__getitem__)
        samples.append((highlight(e.tokens, lambda j: j == i), 1, e.quantile, "pos"))
    half = len(negatives) // 2
    for e in negatives[:half]:
        i = rng.randrange(len(e.tokens))
        samples.append((highlight(e.tokens, lambda j: j == i), 0, -1, "negA"))
    used = {e.key for e in test}
    pool = [e for e in windows if e.key not in used]
    rng.shuffle(pool)
    for e in pool[: len(negatives) - half]:
        silent = [j for j, a in enumerate(e.acts) if a == 0]
        if not silent:
            continue
        i = rng.choice(silent)
        samples.append((highlight(e.tokens, lambda j: j == i), 0, -1, "negB"))
    out = _classify(client, fuzz_prompts.prompt, explanation, [(t, y, q) for t, y, q, _ in samples],
                    workers, "fuzz", max_tokens)
    kinds = [k for _, _, _, k in samples]
    random.Random(42).shuffle(kinds)          # _classify shuffles with seed 42; mirror it
    for want in ("negA", "negB"):
        v = [1 - p for p, k in zip(out["preds"], kinds) if k == want and p is not None]
        out[f"tnr_{want}"] = sum(v) / len(v) if v else float("nan")
    out["n_negA"] = kinds.count("negA")
    out["n_negB"] = kinds.count("negB")
    return out
