# Autointerp run — model `deepseek-ai/DeepSeek-V4-Flash`

| feature | human label | density | LLM label | det bal.acc | fuzz bal.acc |
|---|---|---|---|---|---|
| 12298 | after_a | 3.48% | Reduplicated indefinite article "a a" | 0.50 | 0.55 |
| 14825 | a_ubiquitous_feature | 12.32% | Religious discourse divine-noun detector | 0.65 | 0.68 |
| 24792 | content_word_marker | 92.65% | sequence position detector | 0.55 | 0.57 |
| 25337 | preceded_by_a_verb | 4.35% | verb-particle completion trigger | 0.57 | 0.57 |
| 28617 | to | 1.89% | Infinitive "to" token detector | 0.68 | 0.88 |

## feature 12298 (human: after_a)
- **LLM label**: Reduplicated indefinite article "a a"
- **explanation**: The feature fires on the token " a" (the word "a" with a leading space) when it is immediately preceded by another indefinite article or determiner like " a", " an", " A", "a", "A", or "another", creating a reduplicated or doubled article pattern. It fires strongly when the preceding token is exactly " a" or "an" (100% conditional rate), and weakly when the pattern is less direct or the " a" is followed by a noun. It does not fire on " a" in other contexts, such as after "the" or at the start of a sentence without a preceding determiner.
- **detection**: bal.acc 0.50 (TPR 0.00 / TNR 1.00, parse_err 0)
  - TPR by strength decile: 0:0.00, 1:0.00, 2:0.00, 3:0.00, 4:0.00, 5:0.00, 6:0.00, 7:0.00
- **fuzzing**: bal.acc 0.55 (TPR 0.10 / TNR 1.00, parse_err 0)
- density 3.48%, max_act 3.40, elapsed 10.9s

## feature 14825 (human: a_ubiquitous_feature)
- **LLM label**: Religious discourse divine-noun detector
- **explanation**: This feature fires on tokens in religious or spiritual passages, most strongly on capitalized divine nouns like "God", "Lord", "His", and at sequence starts, where it marks the opening of such discourse. It fires weakly on common function words within these passages, but stays silent in secular, technical, or conversational contexts, indicating it tracks topic and register rather than specific syntax or word forms.
- **detection**: bal.acc 0.65 (TPR 0.30 / TNR 1.00, parse_err 0)
  - TPR by strength decile: 0:0.00, 1:0.08, 2:1.00, 3:1.00, 4:0.00, 5:1.00, 7:0.00
- **fuzzing**: bal.acc 0.68 (TPR 0.50 / TNR 0.85, parse_err 0)
- density 12.32%, max_act 4.58, elapsed 5.7s

## feature 24792 (human: content_word_marker)
- **LLM label**: sequence position detector
- **explanation**: This feature fires on nearly every token after the first, acting as a dense "in-sequence" signal that is strongest at the start of a sequence and at sentence/clause boundaries, and weakest at the very first token and in the final quarter of long sequences. It is not selective for any word, topic, or syntax; the only reliable predictor is position, with token identity being irrelevant. Non-firing tokens are rare random misses in the middle of text, not systematic exclusions.
- **detection**: bal.acc 0.55 (TPR 0.95 / TNR 0.15, parse_err 0)
  - TPR by strength decile: 1:1.00, 2:1.00, 3:1.00, 4:1.00, 5:1.00, 6:1.00, 7:1.00, 8:1.00, 10:0.00
- **fuzzing**: bal.acc 0.57 (TPR 1.00 / TNR 0.15, parse_err 0)
- density 92.65%, max_act 1.94, elapsed 8.8s

## feature 25337 (human: preceded_by_a_verb)
- **LLM label**: verb-particle completion trigger
- **explanation**: This feature fires on the token immediately following short, action-oriented verbs (e.g., "cut", "pick", "click", "dump", "jump", "break") that are part of phrasal verbs or gerund constructions, with the fired-on token being a particle like "up", "out", "on" or a suffix like "-t" or "-ing". It fires strongly (strength 8-10) when the verb is a reliable trigger (100% conditional rate) and the following token completes the phrasal verb; it fires weakly (strength 1-3) on common function words and punctuation as noise, and does not fire on verbs without a following particle or on non-action verbs.
- **detection**: bal.acc 0.57 (TPR 0.15 / TNR 1.00, parse_err 0)
  - TPR by strength decile: 1:0.00, 2:0.00, 3:0.25, 4:0.00, 5:0.00, 6:0.33, 8:0.00, 9:0.50
- **fuzzing**: bal.acc 0.57 (TPR 0.15 / TNR 1.00, parse_err 0)
- density 4.35%, max_act 1.58, elapsed 3.0s

## feature 28617 (human: to)
- **LLM label**: Infinitive "to" token detector
- **explanation**: This feature fires almost exclusively on the token " to" (the standalone word with a leading space), firing on 100% of its occurrences and essentially nowhere else. It fires strongly when "to" is part of a clear infinitive construction (e.g., "able to", "want to", "going to", "purpose to"), with strength increasing when the preceding verb or phrase is semantically salient, and weakly or not at all on other tokens like " the" or " be" which appear as rare noise. It does not fire on "to" embedded in other words (e.g., "today") and shows no positional, topic, or register dependence.
- **detection**: bal.acc 0.68 (TPR 0.50 / TNR 0.85, parse_err 5)
  - TPR by strength decile: 1:0.00, 2:0.00, 3:0.33, 4:1.00, 5:0.00, 6:1.00, 7:0.75, 8:1.00
- **fuzzing**: bal.acc 0.88 (TPR 1.00 / TNR 0.75, parse_err 5)
- density 1.89%, max_act 1.66, elapsed 4.7s
