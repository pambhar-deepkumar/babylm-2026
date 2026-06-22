# From "does word origin matter?" to a measurable answer

A plain-language narrative of the register study: the question, what we built,
what we found, why a direct swap experiment is not feasible, and the honest
limitations. Intended as raw material for the report and for briefing the TA.

## The question we started with

When a language model learns from only a small amount of text (roughly what a
child hears), does the *kind* of vocabulary matter? English has two layers:
plain everyday words with Germanic roots (*understand, freedom, buy*) and
fancier words borrowed from Latin and French (*comprehend, liberty, purchase*).
Our hypothesis: maybe a small model learns more efficiently from plainer text.
If so, you could improve small-model training just by simplifying the vocabulary.

## Step one: we had to label the words

To study this, we first needed to automatically sort every word in the corpus
into "plain" or "fancy." This turned out to be most of the work. Our first
attempt used a tiny reference and could confidently label only 0.42% of the
text. We rebuilt it around a large etymology database (EtymDB) plus a few
supporting tricks, and ended with a reliable labeller covering about 9.5% of the
text (16,000+ words), checked against a reference for accuracy.

## The first finding: "fancy" is mostly "rare"

As soon as we could measure it, a pattern jumped out. Fancy words are almost
always *rare* words, and plain words are the *common* ones. The two are tangled
together. This matters: if you simply ask "does the model handle plain words
better than fancy ones?", you are really asking "does it handle common words
better than rare ones?" which every model does, regardless of origin. So word
origin and word frequency are **confounded**.

## Two ways to separate them

To learn whether word origin matters *on its own*, there are two routes:

- **Statistical:** compare a fancy word and a plain word of *similar frequency*,
  so frequency is held constant. We did this, and once frequency is controlled,
  the word-origin effect is essentially **zero**.
- **Experimental:** physically swap fancy words for plain ones throughout the
  training text, retrain, and see if the scores change.

## Why the experiment is not feasible

We set up the experimental route and hit a wall, for three reasons:

1. **Meaning.** A clean swap must preserve the sentence's meaning and grammar.
   But most fancy words either shift meaning by context (*address, present,
   charge*) or have no plain one-word equivalent at all (*people, important,
   beautiful*). Clean one-to-one swaps are the minority, not the rule.
2. **Reach.** Even using every clean swap we could find, and covering all their
   word forms, we could change only **~0.27% of the corpus**, about one word in
   370. The reason is circular but real: the fancy words that *do* have plain
   synonyms are themselves rare in everyday and child-directed speech.
3. **Detectability.** A 0.27% change is far below the level that could move
   benchmark scores. Even run perfectly, the experiment could not produce a
   meaningful answer; the signal would be lost in noise.

## Limitations (stated plainly)

- **Corpus-specific.** This corpus is child-directed speech, which is already
  heavily plain/Germanic, leaving little fancy vocabulary to work with. A more
  formal corpus might allow a larger intervention.
- **The labeller is good, not perfect.** It still leaves some words unlabelled,
  and it treats Greek-origin learned words as "fancy" by an explicit choice.
- **Single-word swaps only.** We deliberately did *not* rewrite whole sentences
  (e.g. with an LLM), because that changes many things at once (length, grammar,
  style) and would reintroduce the very confounds we were trying to avoid. Our
  intervention is conservative by design.
- **A null swap would show feasibility, not irrelevance.** If we ran the swap
  and saw no change, that would only prove the intervention is too weak, *not*
  that register is irrelevant. The "register does nothing" conclusion rests on
  the statistical test, not the swap.

## The bottom line

Word origin and word frequency are deeply entangled in a small,
developmentally-plausible corpus. The vocabulary you would need to swap is too
sparse to pull them apart by intervention: a clean swap tops out at ~0.27% of
the text. When you separate them statistically instead, the word-origin effect
is essentially zero. So the honest result is not "plain text helps small
models." It is that, in this setting, **register cannot be meaningfully
separated from frequency by intervention, and once frequency is controlled,
register on its own does nothing.** Alongside that finding, we produced a
reusable etymology-labelling tool as a concrete artifact.

See also: `docs/etymology-labeler-journey.md` (how the labeller was built),
`paper/figures/labeler-flow.mmd` (labelling cascade), and the labelled word
lists under `data/derived/`.
