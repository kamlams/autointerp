# Autointerp run — model `deepseek-ai/DeepSeek-V4-Flash-reasoning`

| feature | human label | density | LLM label | det bal.acc | fuzz bal.acc |
|---|---|---|---|---|---|
| 12298 | after_a | 3.48% | after indefinite article | 0.75 | 0.90 |
| 14825 | a_ubiquitous_feature | 12.32% | religious/divine topic | 0.62 | 0.68 |
| 24792 | content_word_marker | 92.65% | dense text continuation | 0.65 | 0.55 |
| 25337 | preceded_by_a_verb | 4.35% | post-stem continuation | 0.62 | 0.72 |
| 28617 | to | 1.89% | space-to token detector | 0.80 | 1.00 |

## feature 12298 (human: after_a)
- **LLM label**: after indefinite article
- **explanation**: This feature fires on the token immediately following the indefinite determiners "a", "an", "another" (and sometimes "one"), i.e. the first content word of an indefinite noun phrase; it also fires weakly on the article itself, especially at sequence start. It fires most strongly on the head word after the article, particularly common adjectives/quantifiers like "more", "much", "well", "given", and when the article is at position 0. It stays silent on words not preceded by an indefinite determiner, and frequent "of", "to", "that" firings are weak spillover onto the next function word after the noun phrase.
- **detection**: bal.acc 0.75 (TPR 0.50 / TNR 1.00, parse_err 0)
  - TPR by strength decile: 0:0.00, 1:0.00, 2:0.50, 3:1.00, 4:1.00, 5:1.00, 6:0.80, 7:0.00
- **fuzzing**: bal.acc 0.90 (TPR 0.80 / TNR 1.00, parse_err 0)
- density 3.48%, max_act 3.40, elapsed 58.6s

## feature 14825 (human: a_ubiquitous_feature)
- **LLM label**: religious/divine topic
- **explanation**: This feature fires on tokens in religious/scriptural passages, most strongly on divine nouns and pronouns (God, Lord, Allah, His, Him, He) and on religious keywords like creation and works, especially at sequence-initial positions. Weak firings are broad spillover onto surrounding content words in religious/spiritual/moral texts and onto common pronouns/nouns elsewhere, while ordinary non-religious words and punctuation are much less likely to fire.
- **detection**: bal.acc 0.62 (TPR 0.30 / TNR 0.95, parse_err 0)
  - TPR by strength decile: 0:0.00, 1:0.08, 2:0.00, 3:1.00, 4:0.00, 5:1.00, 7:1.00
- **fuzzing**: bal.acc 0.68 (TPR 0.65 / TNR 0.70, parse_err 0)
- density 12.32%, max_act 4.58, elapsed 65.1s

## feature 24792 (human: content_word_marker)
- **LLM label**: dense text continuation
- **explanation**: Fires on almost every token after the first, acting as a dense text-flow/continuation feature. It stays silent mainly on sequence-initial tokens, on the article “the” (and “a”) in ordinary mid-sentence contexts, and on non-sentence punctuation such as the period in “common.exceptions”. It fires most strongly on tokens that follow sentence/clause boundaries—spaces, apostrophes, parentheses, quotes, and capitalized pronouns like “It”—and more weakly at the very start and end of sequences.
- **detection**: bal.acc 0.65 (TPR 0.85 / TNR 0.45, parse_err 0)
  - TPR by strength decile: 1:0.00, 2:1.00, 3:1.00, 4:1.00, 5:1.00, 6:1.00, 7:0.00, 8:0.00, 10:1.00
- **fuzzing**: bal.acc 0.55 (TPR 0.90 / TNR 0.20, parse_err 0)
- density 92.65%, max_act 1.94, elapsed 176.9s

## feature 25337 (human: preceded_by_a_verb)
- **LLM label**: post-stem continuation
- **explanation**: The feature fires on the token immediately following a short, consonant-final stem, most reliably after verbs like cut, pick, click, dump, break, and jump (100% of those occurrences trigger the next token) and after subword stems like "ump" and "ck". It fires strongly when the continuation is a phrasal-verb particle ("up", "out", "on") or a derivational suffix ("ster", "ings", "ers"), and stays silent after common function words and in most other positions.
- **detection**: bal.acc 0.62 (TPR 0.25 / TNR 1.00, parse_err 0)
  - TPR by strength decile: 1:0.00, 2:0.00, 3:0.00, 4:0.00, 5:0.00, 6:0.67, 8:1.00, 9:1.00
- **fuzzing**: bal.acc 0.72 (TPR 0.45 / TNR 1.00, parse_err 0)
- density 4.35%, max_act 1.58, elapsed 180.1s

## feature 28617 (human: to)
- **LLM label**: space-to token detector
- **explanation**: This feature fires on every occurrence of the token `" to"` (space + “to”), regardless of whether it is an infinitive marker or preposition, and is silent on all other tokens. It fires most strongly after punctuation and certain subword endings, and somewhat more weakly after highly predictive verbs like “need” or “able”; it does not fire on “to” without the leading-space tokenization or on other words.
- **detection**: bal.acc 0.80 (TPR 0.60 / TNR 1.00, parse_err 0)
  - TPR by strength decile: 1:0.00, 2:0.00, 3:0.33, 4:1.00, 5:0.00, 6:1.00, 7:1.00, 8:1.00
- **fuzzing**: bal.acc 1.00 (TPR 1.00 / TNR 1.00, parse_err 0)
- density 1.89%, max_act 1.66, elapsed 24.6s
