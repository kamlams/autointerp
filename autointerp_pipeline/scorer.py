"""Detection and fuzzing scorers (cheap alternatives to simulation scoring,
following Paulo et al. 2024, adapted to final-token probing)."""
import json
import random
import re

from . import prompts
from .llm import LLMClient, map_concurrent
from .sampler import FeatureExamples

BATCH = 5


def _parse_array(reply: str, n: int) -> list[int] | None:
    m = re.search(r"\[[^\]]*\]", reply, flags=re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list) or len(arr) != n:
        return None
    try:
        return [1 if int(x) == 1 else 0 for x in arr]
    except (TypeError, ValueError):
        return None


def _score_batches(client: LLMClient, system: str, probes: list, workers: int,
                   tag: str, max_tokens: int) -> dict:
    """probes: list of (header, rendered_text, true_label, snippet). Returns metrics."""
    batches = [probes[i: i + BATCH] for i in range(0, len(probes), BATCH)]

    def run(batch):
        sys_msg = system.replace("{n}", str(len(batch)))
        user = prompts.format_batch_user([(h, t) for h, t, _, _ in batch])
        for _ in range(2):  # one extra attempt on unparseable reply
            reply = client.chat([{"role": "system", "content": sys_msg},
                                 {"role": "user", "content": user}],
                                max_tokens=max_tokens, tag=tag)
            preds = _parse_array(reply, len(batch))
            if preds is not None:
                return preds
        return [None] * len(batch)

    preds = [p for batch in map_concurrent(run, batches, workers) for p in batch]
    labels = [y for _, _, y, _ in probes]

    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    npos = sum(1 for y in labels if y == 1)
    nneg = len(labels) - npos
    tpr = tp / npos if npos else float("nan")
    tnr = tn / nneg if nneg else float("nan")
    return {
        "balanced_acc": (tpr + tnr) / 2,
        "tpr": tpr, "tnr": tnr,
        "n_pos": npos, "n_neg": nneg,
        "n_parse_errors": sum(1 for p in preds if p is None),
        "preds": preds, "labels": labels,
        "positions": [s.pos for _, _, _, s in probes],
        "strengths": [s.norm_score for _, _, _, s in probes],
    }


def detection(client: LLMClient, explanation: str, ex: FeatureExamples,
              workers: int = 4, seed: int = 0, max_tokens: int = 6144) -> dict:
    """Probe: does the feature fire on the FINAL token of each snippet?"""
    rng = random.Random(seed)
    probes = [(f"final token at position {s.pos}", s.render(False, upto_anchor=True), y, s)
              for s, y in [(s, 1) for s in ex.test_act] + [(s, 0) for s in ex.test_neg]]
    rng.shuffle(probes)
    system = prompts.DETECTION_SYSTEM.replace("{explanation}", explanation)
    out = _score_batches(client, system, probes, workers, "detect", max_tokens)
    hits = {}
    for (_, _, y, s), p in zip(probes, out["preds"]):
        if y == 1 and p is not None:
            hits.setdefault(s.norm_score, []).append(p)
    out["tpr_by_strength"] = {k: sum(v) / len(v) for k, v in sorted(hits.items())}
    return out


def fuzzing(client: LLMClient, explanation: str, ex: FeatureExamples,
            workers: int = 4, seed: int = 0, max_tokens: int = 6144) -> dict:
    """Probe: are the << >> marks placed where the explanation says the feature fires?"""
    rng = random.Random(seed + 1)
    probes = [("", s.render(True, show_weak=False), y, s)
              for s, y in [(s, 1) for s in ex.fuzz_correct] + [(s, 0) for s in ex.fuzz_incorrect]]
    rng.shuffle(probes)
    system = prompts.FUZZING_SYSTEM.replace("{explanation}", explanation)
    return _score_batches(client, system, probes, workers, "fuzz", max_tokens)
