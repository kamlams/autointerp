# Autointerp run — model `moonshotai/Kimi-K3`

| feature | human label | density | LLM label | det bal.acc | fuzz bal.acc |
|---|---|---|---|---|---|
| 12298 | after_a | 3.48% | Word following indefinite article "a/an" | 0.90 | 1.00 |
| 14825 | a_ubiquitous_feature | 12.32% | Sequence openings and religious vocabulary | 0.60 | 0.53 |
| 24792 | content_word_marker | 92.65% | Dense context-gated surprisal signal | 0.75 | 0.57 |
| 25337 | preceded_by_a_verb | 4.35% | Particles and suffixes after stop-final stems | 0.62 | 0.68 |
| 28617 | to | 1.89% | " to" token detector | 0.80 | 1.00 |

## feature 12298 (human: after_a)
- **LLM label**: Word following indefinite article "a/an"
- **explanation**: It fires on the token immediately after "a"/"an" — essentially every occurrence of the article (any capitalization, plus "another", and even word-initial subword "a" as in "aft"/"aurally") is followed by a firing, while "the" almost never triggers it; sequence-initial "a" also fires on itself. It fires most strongly on that first post-article token, especially sequence-initial "a" and modifiers/quantifiers ("a more/much/given/lot/bit"), and spills weakly onto the next token, which explains frequent weak firings on "of", punctuation, "to", "that" ("a lot of", "a vaccine,"). Weaker secondary firing also follows the determiner "one" (33%) and occasionally "some"/"every".
- **detection**: bal.acc 0.90 (TPR 0.80 / TNR 1.00, parse_err 0)
  - TPR by strength decile: 0:0.00, 1:0.50, 2:1.00, 3:1.00, 4:1.00, 5:1.00, 6:1.00, 7:1.00
- **fuzzing**: bal.acc 1.00 (TPR 1.00 / TNR 1.00, parse_err 0)
- density 3.48%, max_act 3.40, elapsed 3267.0s

## feature 14825 (human: a_ubiquitous_feature)
- **LLM label**: Sequence openings and religious vocabulary
- **explanation**: A dense feature with weak, patternless background firing on common tokens, but strong firing concentrated at the very start of sequences (position 0 fires at 24% with the highest mean activation, decaying over the first ~15 tokens) and on religious vocabulary — " God" (74% conditional), " Lord" (75%), " His", " He", Allah, creation — especially within Bible/Quran/sermon passages. It stays silent on most ordinary mid/late-sequence tokens, whitespace, and subword continuations.
- **detection**: bal.acc 0.60 (TPR 0.20 / TNR 1.00, parse_err 0)
  - TPR by strength decile: 0:0.00, 1:0.08, 2:0.00, 3:0.00, 4:1.00, 5:1.00, 7:1.00
- **fuzzing**: bal.acc 0.53 (TPR 0.40 / TNR 0.65, parse_err 0)
- density 12.32%, max_act 4.58, elapsed 1196.9s

## feature 24792 (human: content_word_marker)
- **LLM label**: Dense context-gated surprisal signal
- **explanation**: It fires on ~93% of tokens — essentially any token with preceding context — but stays silent at sequence position 0 (18.5% fire rate) and on highly predictable tokens, above all " the" (61% vs 93% baseline; e.g. "for the", "taking the time", deterministic code punctuation). Strength is dominated by position, peaking at tokens 1–15 and decaying toward the end of the 128-token window, and is maximal on apostrophes, opening quotes/parens, and sentence-initial "It"/"I" — consistent with a prediction-difficulty signal that requires context and fades as context makes tokens predictable.
- **detection**: bal.acc 0.75 (TPR 0.90 / TNR 0.60, parse_err 0)
  - TPR by strength decile: 1:0.00, 2:1.00, 3:1.00, 4:1.00, 5:0.67, 6:1.00, 7:1.00, 8:1.00, 10:1.00
- **fuzzing**: bal.acc 0.57 (TPR 0.85 / TNR 0.30, parse_err 0)
- density 92.65%, max_act 1.94, elapsed 3437.8s

## feature 25337 (human: preceded_by_a_verb)
- **LLM label**: Particles and suffixes after stop-final stems
- **explanation**: The feature fires on the token immediately following a short stem ending in a stop consonant (-t, -ck, -k, -p, -mp, and voiced -d/-g): after cut, pick, click, kick, break, jump, dump, put it fires on essentially 100% of occurrences. The fired-on token is usually a phrasal-verb particle (up, on, out, off, away, by — "break up", "cut out") or a subword suffix completing the stem ("cut+t+ings", "flick+ering", "dump+ster", "Pack+aging"), and these completions fire strongest, with a mild boost early in sequences. It stays silent on the same particles after other verbs ("fuel up") and after function words; weak scatter on "the/to/of/with" is noise.
- **detection**: bal.acc 0.62 (TPR 0.25 / TNR 1.00, parse_err 0)
  - TPR by strength decile: 1:0.00, 2:0.00, 3:0.00, 4:0.00, 5:0.00, 6:0.67, 8:1.00, 9:1.00
- **fuzzing**: bal.acc 0.68 (TPR 0.40 / TNR 0.95, parse_err 0)
- density 4.35%, max_act 1.58, elapsed 2078.7s

## feature 28617 (human: to)
- **LLM label**: " to" token detector
- **explanation**: It fires on virtually every occurrence of the exact token " to" (100% of its corpus occurrences, ~98% of all firings), whether it is an infinitive marker, a preposition, or even a subword chunk inside words like "tote", and it stays silent on "to" without the leading space. Strength is uniformly high with no reliable contextual modulator — perhaps slightly weaker inside the most frequent verb+to collocations — and the rare weak firings on other tokens ("\n", ".", " the") are noise-level spillover.
- **detection**: bal.acc 0.80 (TPR 0.60 / TNR 1.00, parse_err 0)
  - TPR by strength decile: 1:0.00, 2:0.00, 3:0.33, 4:1.00, 5:0.00, 6:1.00, 7:1.00, 8:1.00
- **fuzzing**: bal.acc 1.00 (TPR 1.00 / TNR 1.00, parse_err 0)
- density 1.89%, max_act 1.66, elapsed 299.8s

Note: feature14825 was scored in a separate rerun (`run_14825.log`, `calls_14825.jsonl`) after the
retry-policy fix (commit af0e66d); the first attempt lost it to gateway stream drops at 8 workers.
