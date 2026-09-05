"""Loading of the autointerp sample-data format.

Corpus: token_list_*.pkl, nested lists token_ls[batch][prompt][pos] -> decoded token str.
Feature: feature{ID}__{label}__*.pt, float tensor [N, 4] rows = (batch, prompt, pos, activation)
         listing every token position where the feature fired.
"""
import os
import pickle
import re
from dataclasses import dataclass, field

import torch


@dataclass
class Corpus:
    tokens: list  # tokens[batch][prompt][pos] -> str
    n_batches: int
    n_prompts: int
    n_tokens: int
    _count_all: dict = field(default_factory=dict)   # token -> occurrences anywhere
    _count_prev: dict = field(default_factory=dict)  # token -> occurrences at pos < T-1

    @property
    def total_tokens(self):
        return self.n_batches * self.n_prompts * self.n_tokens

    def token_counts(self) -> tuple[dict, dict]:
        """Corpus-wide occurrence counts, computed once (~2 s): denominators for
        P(fire | fired-on token = Y) and P(fire | preceding token = X)."""
        if not self._count_all:
            call, cprev = {}, {}
            last = self.n_tokens - 1
            for batch in self.tokens:
                for seq in batch:
                    for i, t in enumerate(seq):
                        call[t] = call.get(t, 0) + 1
                        if i < last:
                            cprev[t] = cprev.get(t, 0) + 1
            self._count_all, self._count_prev = call, cprev
        return self._count_all, self._count_prev


def load_corpus(path: str) -> Corpus:
    with open(path, "rb") as f:
        token_ls = pickle.load(f)
    return Corpus(
        tokens=token_ls,
        n_batches=len(token_ls),
        n_prompts=len(token_ls[0]),
        n_tokens=len(token_ls[0][0]),
    )


@dataclass
class FeatureRecord:
    feature_id: int
    human_label: str | None  # from filename; None for unlabeled dumps
    path: str
    records: torch.Tensor  # [N, 4] (batch, prompt, pos, act)
    acts_by_prompt: dict = field(default_factory=dict)  # (batch, prompt) -> {pos: act}
    firing_keys: set = field(default_factory=set)  # {batch*P*T + prompt*T + pos}

    @property
    def n_firings(self) -> int:
        return self.records.shape[0]

    @property
    def acts(self) -> torch.Tensor:
        return self.records[:, 3]

    @property
    def max_act(self) -> float:
        return float(self.acts.max())

    def density(self, corpus: Corpus) -> float:
        return self.n_firings / corpus.total_tokens

    def build_index(self, corpus: Corpus):
        if self.acts_by_prompt:
            return
        P, T = corpus.n_prompts, corpus.n_tokens
        b = self.records[:, 0].to(torch.int32).tolist()
        p = self.records[:, 1].to(torch.int32).tolist()
        pos = self.records[:, 2].to(torch.int32).tolist()
        a = self.records[:, 3].tolist()
        idx = {}
        keys = set()
        for bi, pi, ti, ai in zip(b, p, pos, a):
            idx.setdefault((bi, pi), {})[ti] = ai
            keys.add((bi * P + pi) * T + ti)
        self.acts_by_prompt = idx
        self.firing_keys = keys

    def fires_at(self, corpus: Corpus, batch: int, prompt: int, pos: int) -> bool:
        P, T = corpus.n_prompts, corpus.n_tokens
        return (batch * P + prompt) * T + pos in self.firing_keys


_FNAME_RE = re.compile(r"feature(\d+)(?:__(.+?))?__\d+\.pt$")


def load_feature(path: str) -> FeatureRecord:
    m = _FNAME_RE.search(os.path.basename(path))
    if m:
        feature_id = int(m.group(1))
        human_label = m.group(2)
    else:
        feature_id, human_label = -1, None
    records = torch.load(path, weights_only=True)
    assert records.ndim == 2 and records.shape[1] == 4, f"unexpected shape {records.shape}"
    return FeatureRecord(feature_id=feature_id, human_label=human_label, path=path, records=records)


def discover_feature_files(data_dir: str) -> list[str]:
    out = []
    for fn in sorted(os.listdir(data_dir)):
        if fn.startswith("feature") and fn.endswith(".pt"):
            out.append(os.path.join(data_dir, fn))
    return out
