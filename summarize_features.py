"""Build a readable per-feature report from an autointerp run over a feature-data directory.

For every feature: density, firing count, max activation, the tokens it fires on most often with
their conditional rate P(fire | token), the share of firings at sequence position 0, the LLM label
and explanation, and the detection / fuzzing balanced accuracies.

  python summarize_features.py FEATURE_DATA_DIR INTERP_DIR OUT_MD [--title "..."]
"""

from __future__ import annotations

import collections
import glob
import json
import math
import os
import pickle
import sys

import torch

feat_dir, interp_dir, out_md = sys.argv[1], sys.argv[2], sys.argv[3]
title = sys.argv[sys.argv.index("--title") + 1] if "--title" in sys.argv else feat_dir.rstrip("/").split("/")[-1]

# sampling.json (written by extract_features.py) is optional: density is firings / corpus tokens
sampling = f"{feat_dir}/sampling.json"
meta = json.load(open(sampling)) if os.path.exists(sampling) else {"features": {}}
tokens = pickle.load(open(glob.glob(f"{feat_dir}/token_list_*.pkl")[0], "rb"))
corpus_counts = collections.Counter(t for b in tokens for p in b for t in p)
n_tokens = sum(corpus_counts.values())

rows = []
for path in sorted(glob.glob(f"{interp_dir}/feature*.json")):
    d = json.load(open(path))
    fid = int(d["feature_id"])
    rec = torch.load(glob.glob(f"{feat_dir}/feature{fid}__*.pt")[0])
    idx = rec[:, :3].long().tolist()
    acts = rec[:, 3]
    strong = acts >= max(0.3 * float(acts.max()), float(acts.median()))
    fired = collections.Counter(tokens[b][p][q] for (b, p, q), s in zip(idx, strong.tolist()) if s)
    top = [(t, c, c / corpus_counts[t]) for t, c in fired.most_common(4)]
    det, fuz = d.get("detection") or {}, d.get("fuzzing") or {}
    fz = fuz.get("balanced_acc") if fuz.get("n_pos") else None
    if fz is not None and isinstance(fz, float) and math.isnan(fz):
        fz = None
    rows.append(dict(fid=fid, dens=meta["features"].get(str(fid), {}).get("density", rec.shape[0] / n_tokens), n=int(rec.shape[0]),
                     mx=float(acts.max()), pos0=float((rec[:, 2] == 0).float().mean()), top=top,
                     label=d.get("label"), expl=(d.get("explanation") or "").strip(),
                     det=det.get("balanced_acc"), fuz=fz))
rows.sort(key=lambda r: -r["dens"])


def fmt_tok(t: str) -> str:
    return "`" + t.replace("\n", "\\n").replace("\t", "\\t").replace("|", "\\|") + "`"


lines = [f"# {title}", "",
         f"{len(rows)} features, corpus {n_tokens:,} tokens. `density` = share of tokens the feature fires on; "
         "`top tokens` = most frequent tokens among the strong firings, with P(fire | token) in brackets; "
         "`pos0` = share of firings on the first token of a 128-token chunk; `det`/`fuzz` = balanced accuracy "
         "of the explanation under detection / fuzzing scoring (chance 0.5).", "",
         "| latent | density | firings | max act | pos0 | top tokens (P(fire\\|token)) | LLM label | det | fuzz |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    toks = " ".join(f"{fmt_tok(t)} {c}×[{p:.0%}]" for t, c, p in r["top"])
    lines.append(f"| {r['fid']} | {r['dens'] * 100:.3f}% | {r['n']:,} | {r['mx']:.2f} | {r['pos0']:.0%} | {toks} | "
                 f"{r['label']} | {r['det'] if r['det'] is None else round(r['det'], 2)} | "
                 f"{r['fuz'] if r['fuz'] is None else round(r['fuz'], 2)} |")
lines += ["", "## Explanations", ""]
for r in rows:
    lines.append(f"**{r['fid']}** (density {r['dens'] * 100:.3f}%, det {r['det']}, fuzz {r['fuz']}) — "
                 f"*{r['label']}*: {r['expl']}")
    lines.append("")
open(out_md, "w").write("\n".join(lines))
print(f"wrote {out_md}: {len(rows)} features")
