# Example activation data

Three latents of the latest result set (layer 7, see `results/feature_test32b_260918/`), enough
to run the pipeline end to end without a GPU:

| file | latent | what it turned out to be |
|---|---|---|
| `feature25609__260918.pt` | 25609 | the verb *give* in all its forms (1,479 firings) |
| `feature15329__260918.pt` | 15329 | *drink* and beverage words (556 firings) |
| `feature971__260918.pt` | 971 | religious narrative context (3,187 firings) — detection high, fuzzing near chance |

`token_list_260911.pkl` is the corpus they index into: `tokens[batch][prompt][pos] -> str`,
825 x 16 x 128 = 1,689,600 tokens of a held-out SmolLM2-135M-10B shard. Each `.pt` holds `[N,4]`
rows `(batch, prompt, pos, activation)`, one per firing.

`run_autointerp.py` reads this directory by default. The full activation data of a result set is
regenerated with `extract_features.py` (minutes per layer) and is not tracked.
