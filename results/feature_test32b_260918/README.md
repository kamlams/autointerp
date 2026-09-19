# Second batch: the 96 latents of the 2026-09-18 `feature_to_test.txt`

The collaborator replaced `<collaborator-data>/data260913/feature_to_test.txt`
(02:49 on 2026-09-18) with a new list: 32 latents each for layers 0, 7 and 15 of
`OLMoE_moein_unitnormFalse_lrwarmup0_lr0.001_seed0_sample200M`. Same corpus, same protocol and
same tooling as the first batch of 2026-09-16 (not tracked in this repository); 16 latents appear in both lists
(L0 5, L7 4, L15 7) and their explanations were reused verbatim rather than re-generated.

- Scoring: `run_scoring.sh`, 6 processes, 2026-09-18 23:20 - 2026-09-19 00:19. **96 of 96
  explained, no give-ups** (the first batch lost 2).
- Reports: `features_L0.md`, `features_L7.md`, `features_L15.md`; per-feature JSON and logs in
  `moein_L*_test32b/`.

| layer | detection median | fuzzing median | both >= 0.7 | latents with density >= 5% |
|---|---|---|---|---|
| 0 | 0.70 | 0.70 | 13 / 32 | 14 |
| 7 | 0.70 | 0.66 | 11 / 32 | 1 |
| 15 | 0.70 | 0.64 | 6 / 32 | 2 |

This list is much denser than the first one: 17 of 96 latents fire on at least 5% of tokens,
including three extreme cases — L0 2565 at **97.8%** (fires on almost every token), L7 8846 at
78.9% and L0 18871 at 28.8%. Such latents get low fuzzing scores (median 0.60 for density >= 5%
against 0.65 for < 1%): no explanation can say which token they land on.

Two findings worth passing to the collaborator:

- **Duplicated `" to"` detectors at layer 0.** Latents 36, 6584 and 6758 all fire on ` to` at
  ~52% of its occurrences with density 1.73-1.77%; the pairwise Jaccard of their firing positions
  is 0.95, 0.95 and 0.99. They are the same feature three times over, the same redundancy seen in
  a cluster of near-identical position-0 latents found earlier at layer 11 of the same SAE.
- **Clean lexical detectors repeat across layers**: ` give`/`given`/`gives`/`giving` appears as
  latent 25609 at layer 7 (det 0.88, fuzz 0.85) and as 7932 at layer 15 (det 0.85, fuzz 1.00);
  ` kill` appears as 29038 at layer 7 and 907 at layer 15.
