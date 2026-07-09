# Building the etymological labeller: the journey

How we went from a near-useless labeller to one that reliably sorts the corpus
vocabulary into **Latinate** (Latin/French-rooted, the "fancy" stratum) and
**Germanic** (Old English/Norse-rooted, the "plain" stratum). Written in plain
terms; the technical mapping is at the bottom.

The mental picture: every English word is like a person whose family came from
somewhere. We go through the ~111,000 distinct words in the corpus and check each
one's origin, because the project swaps fancy words for plain ones.

## Coverage at a glance

| Step | What we used | Latinate words found | % of text covered |
|---|---|---|---|
| 1. Start | tiny phrasebook (EtymoLink) | 707 | 0.42% |
| 4. Big book | huge encyclopedia (EtymDB) | ~11k | 15.5% *(inflated)* |
| 5. Cleaned up | same, de-biased | ~16k | ~9% *(honest)* |
| 8. Final | all sources combined | **16,386** | **~9.5%** |

## The narrative

**1. The pocket phrasebook.** We started with a tiny reference that knew only
~1,700 words. It confidently flagged just 707 Latinate words — barely 0.42% of
the text. For 99% of words it shrugged "never heard of it." And the few it knew
were polluted with month names and city names.

**2. The lightbulb.** We realised that shrug ("don't know") was being misread as
"not Latinate." We tested it on obviously-Latinate words — *important,
different, question* — and it had never heard of any of them. The problem wasn't
our text. The reference book was just too thin.

**3. The big encyclopedia.** We found a giant reference built from Wiktionary
that knows 880,000 English words and their family trees, and plugged it in.

**4. Too good to be true.** Coverage leapt from 0.42% to 15.5%. Exciting — until
two independent reviews caught it cheating. The "fanciest" words it found were
nonsense: subtitle gibberish (*chi, mot*) and ordinary words like *are* and
*name* that merely *look* like fancy foreign words. It was fooled by look-alikes,
so much of that 15.5% was fake.

**5. The honest cleanup.** We fixed the look-alike trap, taught it that *houses*
is just *house* with an "s", and filled in labels the encyclopedia spelled
differently. Coverage settled to a real ~9% — a smaller but honest number. The
missing 6% had been fake all along.

**6. The stubborn gap.** The cleanup revealed that even the giant encyclopedia
drew a blank on some everyday Latinate words — *table, voice, liberty*. Their
family trees stopped halfway, with no link back to Latin. It had the words, but
not their full history.

**7. Call in the LLM.** For those few hundred stubborn common words, we used a
language model to label them (it is genuinely good at "where did this word come
from"). It correctly sorted ~90% of them — recovering exactly the everyday
Latinate words the encyclopedia missed. These labels are frozen to a file, so
the pipeline stays reproducible and no model runs at training time.

**8. Stitch it together.** We combined everything into one detector that checks,
in order of confidence: hand-fixes, the LLM labels, the encyclopedia, a
word-ending grammar trick, then spelling-pattern guesses. We hand-corrected a
handful of stubborn look-alikes at the very top of the list (*best, sea, ball*
were being mistaken for fancy words).

**Result:** from 707 Latinate words to 16,386, and from 0.42% to ~9.5% of the
text — with the most frequent words now sorted correctly and reliably.

## The lesson

The honest path was not a straight climb. We hit a fake high (15.5%), forced
ourselves back down to the truth (9%), then climbed back up the honest way
(9.5%). That dip *is* the integrity of the result: a defensible 9.5% beats a
flattering 15% we could not stand behind.

## Plain term → technical term

| Plain term in the story | What it actually is |
|---|---|
| pocket phrasebook | EtymoLink gold list (~1.7k words) + Macro-Etym fallback |
| huge encyclopedia | EtymDB-2.0, a Wiktionary-derived etymology graph (~882k English lexemes) |
| family tree / origin | tracing a word up its ancestry edges to the nearest Germanic/Latinate ancestor |
| look-alike trap | case-folded homograph conflation across senses and languages |
| "houses → house" trick | lemmatisation before lookup |
| word-ending / spelling guess | Latinate suffix heuristic (-tion, -ity, -ous), guarded against Germanic suffixes |
| hand-fixes | a small manual override table for known high-frequency errors |
| de-biased / honest cleanup | dropping proper nouns, excluding stop-words and token artifacts |

See the labelling cascade diagram in `paper/figures/labeler-flow.mmd`, and the
implementation in `src/babylm_2026/etymology.py` (`EtymologyLabeler`).
