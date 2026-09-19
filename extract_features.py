"""Build an autointerp `feature-data` directory from a TopK SAE on OLMoE-1B-7B-0924.

Output layout (what `run_autointerp.py --data-dir` reads):
  <out-root>/<name>/token_list_<tag>.pkl      corpus, token_ls[batch][prompt][pos] -> decoded token str
  <out-root>/<name>/feature<ID>__<tag>.pt     float32 [N, 4] rows (batch, prompt, pos, activation)
  <out-root>/<name>/sampling.json             which latents were sampled, why (density bin), and the
                                              per-latent firing counts summary
  <out-root>/<name>/density.npy               firing density of every latent on the corpus

The corpus is built once from a held-out SmolLM2-135M-10B shard (default: the last one, never used
in any SAE training here or by the collaborator) with the collaborator's `chunk_and_tokenize`
(sparsify branch `collaborator-mods`, max_seq_len 128), exactly like the original feature-data
(825 batches x 16 prompts x 128 tokens = 1.69M tokens), and cached under <out-root>/corpus/ so
every SAE is evaluated on the same tokens.

Sampling: latents with at least --min-firings firings are binned by log-density on fixed edges
(--bins) and --n-features // n_bins are drawn per bin with --seed; a bin that is short is filled
from the other eligible latents. The same edges and seed for every SAE make the comparison fair
per density bin (dense features are intrinsically harder to explain).

Run from the repo root with HF_HOME exported and the paper2 python:
  python extract_features.py --sae <sae.safetensors> --layer 3 --input h --name moein_L3
  python extract_features.py --sae <sae.safetensors> --layer 3 --input m --name residmid_L3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from autointerp_pipeline.sae_tools import MODEL_NAME, ActivationTap, TopKSAE, load_truncated_model

SHARD_GLOB = "hub/datasets--EleutherAI--SmolLM2-135M-10B/snapshots/*/data/train-{shard:05d}-of-00085.parquet"


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sae", required=True, help="sae.safetensors (sparsify or W_enc layout)")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--input", choices=["h", "m"], required=True,
                   help="h = output of post_attention_layernorm (moein SAEs); m = its input (residmid SAEs)")
    p.add_argument("--name", required=True, help="output sub-directory name")
    p.add_argument("--out-root", default="outputs/feature-data")
    p.add_argument("--tag", default=time.strftime("%y%m%d"))
    p.add_argument("--shard", type=int, default=84, help="SmolLM2 parquet shard for the corpus")
    p.add_argument("--n-batches", type=int, default=825)
    p.add_argument("--prompts", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--n-features", type=int, default=100)
    p.add_argument("--latents", default="", help="comma/space-separated latent ids to extract verbatim "
                   "(skips the density-stratified sampling; ids that never fire on the corpus are reported and dropped)")
    p.add_argument("--bins", default="6e-5,3e-4,1e-3,1e-2,1.01",
                   help="density bin edges; the first edge is also the eligibility floor")
    p.add_argument("--min-firings", type=int, default=0, help="extra eligibility floor in firings (0 = edges only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude", default="", help="comma-separated sampling.json files whose features must not be drawn again")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_corpus(args, corpus_dir: Path) -> np.ndarray:
    """Return token ids [n_batches, prompts, seq_len]; build and cache the corpus if needed."""
    ids_file = corpus_dir / "ids.npy"
    if ids_file.exists():
        ids = np.load(ids_file)
        print(f"corpus: loaded {ids.shape} from {ids_file}")
        return ids
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from sparsify.data import chunk_and_tokenize  # collaborator's version (branch collaborator-mods)

    hf_home = os.environ["HF_HOME"]
    shards = glob.glob(str(Path(hf_home) / SHARD_GLOB.format(shard=args.shard)))
    assert len(shards) == 1, f"shard {args.shard} not found under {hf_home}"
    n_chunks = args.n_batches * args.prompts
    dataset = load_dataset("parquet", data_files=shards, split="train")
    # ~50 tokens per 128-token chunk per document on average; take a generous prefix of documents
    n_docs = min(len(dataset), n_chunks // 2)
    dataset = dataset.select(range(n_docs))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = chunk_and_tokenize(dataset, tokenizer, max_seq_len=args.seq_len, num_proc=16, format="numpy")
    assert len(tokenized) >= n_chunks, f"{len(tokenized)} chunks from {n_docs} docs, need {n_chunks}"
    ids = np.stack([np.asarray(tokenized[i]["input_ids"]) for i in range(n_chunks)]).astype(np.int64)
    ids = ids.reshape(args.n_batches, args.prompts, args.seq_len)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    np.save(ids_file, ids)
    # decoded token strings, one per id (what the original feature-data holds)
    vocab = {int(i): tokenizer.decode([int(i)]) for i in np.unique(ids)}
    token_ls = [[[vocab[int(t)] for t in prompt] for prompt in batch] for batch in ids]
    with (corpus_dir / f"token_list_{args.tag}.pkl").open("wb") as fh:
        pickle.dump(token_ls, fh)
    (corpus_dir / "corpus.json").write_text(json.dumps({
        "source": os.path.basename(shards[0]), "documents": n_docs, "n_batches": args.n_batches,
        "prompts": args.prompts, "seq_len": args.seq_len, "tokens": int(ids.size), "tokenizer": MODEL_NAME,
        "chunking": "sparsify collaborator-mods chunk_and_tokenize (eos-joined stream cut into fixed chunks)"}, indent=2))
    print(f"corpus: built {ids.shape} from {n_docs} documents of {os.path.basename(shards[0])}")
    return ids


@torch.no_grad()
def encode_corpus(args, ids: np.ndarray, sae: TopKSAE):
    """Run the truncated model over the corpus; return TopK indices [T, k] and values [T, k] (CPU)."""
    base = load_truncated_model(args.layer, args.device)
    tap = ActivationTap(base.layers, args.layer, args.input)
    idx_all, val_all = [], []
    for b in range(ids.shape[0]):
        x_ids = torch.from_numpy(ids[b]).to(args.device)
        base(input_ids=x_ids, use_cache=False)
        val, idx = sae.encode_topk(tap.value.reshape(-1, sae.d_x).float())
        idx_all.append(idx.to(torch.int32).cpu())
        val_all.append(val.cpu())
        if (b + 1) % 200 == 0:
            print(f"  encoded {b + 1}/{ids.shape[0]} batches", flush=True)
    return torch.cat(idx_all), torch.cat(val_all)


def sample_by_density(args, density, counts, edges):
    """Draw args.n_features latents, equal quota per density bin, excluding --exclude lists."""
    n_bins = len(edges) - 1
    eligible = (density >= edges[0]) & (counts >= args.min_firings)
    excluded = sorted({int(k) for f in args.exclude.split(",") if f for k in json.load(open(f))["features"]})
    if excluded:
        eligible[np.array(excluded)] = False
        print(f"excluded {len(excluded)} previously sampled latents")
    bin_of = np.digitize(density, edges) - 1  # 0..n_bins-1 inside the edges
    rng = np.random.default_rng(args.seed)
    quota = args.n_features // n_bins
    chosen: list[int] = []
    per_bin = {}
    for b in range(n_bins):
        pool = np.flatnonzero(eligible & (bin_of == b))
        take = rng.choice(pool, size=min(quota, len(pool)), replace=False) if len(pool) else np.array([], dtype=int)
        chosen += take.tolist()
        per_bin[b] = {"edges": [edges[b], edges[b + 1]], "eligible": int(len(pool)), "sampled": int(len(take))}
    if len(chosen) < args.n_features:
        rest = np.setdiff1d(np.flatnonzero(eligible), np.array(chosen, dtype=int))
        extra = rng.choice(rest, size=min(args.n_features - len(chosen), len(rest)), replace=False)
        chosen += extra.tolist()
        per_bin["fill"] = int(len(extra))
    chosen = sorted(int(c) for c in chosen)
    return chosen, per_bin, bin_of, excluded


def main() -> None:
    args = parse()
    torch.set_float32_matmul_precision("high")
    out_root = Path(args.out_root)
    corpus_dir = out_root / f"corpus_shard{args.shard}_{args.n_batches}x{args.prompts}x{args.seq_len}"
    out = out_root / args.name
    out.mkdir(parents=True, exist_ok=True)

    ids = build_corpus(args, corpus_dir)
    pkl = next(corpus_dir.glob("token_list_*.pkl"))
    target = out / pkl.name
    if not target.exists():
        target.write_bytes(pkl.read_bytes())

    sae = TopKSAE.load(args.sae, device=args.device)
    print(f"SAE {args.sae}: k={sae.k}, d_z={sae.d_z}, input {args.input}{args.layer}")
    idx, val = encode_corpus(args, ids, sae)  # [T, k]
    T = idx.shape[0]
    active = val > 0
    flat_idx = idx[active].long()
    counts = torch.bincount(flat_idx, minlength=sae.d_z).numpy()
    density = counts / T
    np.save(out / "density.npy", density)
    live = int((counts > 0).sum())
    print(f"{T} tokens; live latents {live}/{sae.d_z}; mean active per token {active.float().sum(1).mean():.2f}")

    # ---- explicit latent list, or stratified sampling on fixed density edges
    edges = [float(e) for e in args.bins.split(",")]
    bin_edges = np.digitize(density, edges) - 1
    if args.latents:
        asked = [int(x) for x in args.latents.replace(",", " ").split()]
        silent = [lat for lat in asked if counts[lat] == 0]
        chosen = sorted(lat for lat in asked if counts[lat] > 0)
        print(f"explicit latent list: {len(asked)} asked, {len(chosen)} fire on the corpus"
              + (f"; never fire: {silent}" if silent else ""))
        per_bin = {"explicit": {"asked": asked, "silent": silent}}
        bin_of = bin_edges
        excluded = []
    else:
        chosen, per_bin, bin_of, excluded = sample_by_density(args, density, counts, edges)
    print("bins:" if not args.latents else "selection:", json.dumps(per_bin)[:400])

    # ---- write feature records (batch, prompt, pos, activation)
    tok_pos = torch.arange(T)
    batch = (tok_pos // (args.prompts * args.seq_len)).to(torch.float32)
    prompt = ((tok_pos // args.seq_len) % args.prompts).to(torch.float32)
    pos = (tok_pos % args.seq_len).to(torch.float32)
    tok_of_firing = tok_pos.unsqueeze(1).expand_as(idx)[active]
    act_of_firing = val[active]
    order = torch.argsort(flat_idx, stable=True)
    sorted_lat = flat_idx[order]
    bounds = torch.searchsorted(sorted_lat, torch.tensor(chosen + [sae.d_z]))
    meta = {}
    for i, lat in enumerate(chosen):
        sel = order[bounds[i]: torch.searchsorted(sorted_lat, torch.tensor(lat + 1))]
        t = tok_of_firing[sel]
        rows = torch.stack([batch[t], prompt[t], pos[t], act_of_firing[sel]], dim=1).float()
        torch.save(rows, out / f"feature{lat}__{args.tag}.pt")
        meta[lat] = {"bin": int(bin_of[lat]), "density": float(density[lat]), "n_firings": int(counts[lat]),
                     "max_act": float(rows[:, 3].max()) if len(rows) else 0.0}
    (out / "sampling.json").write_text(json.dumps({
        "sae": args.sae, "layer": args.layer, "input": args.input, "corpus": str(corpus_dir), "tokens": T,
        "live_latents": live, "n_features": len(chosen), "seed": args.seed, "edges": edges,
        "min_firings": args.min_firings, "excluded": excluded, "bins": per_bin, "features": meta}, indent=2))
    print(f"wrote {len(chosen)} feature files + sampling.json + density.npy to {out}")


if __name__ == "__main__":
    main()
