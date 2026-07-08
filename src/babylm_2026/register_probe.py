"""Frequency-matched register probe: does a model prefer the Germanic synonym
beyond what frequency alone explains?

The core measurement. For a synonym pair (Latinate L, Germanic G), we drop both
words into identical neutral carrier frames and measure each word's surprisal in
context. The per-pair effect is

    delta = surprisal(L) - surprisal(G)        (averaged over frames)

A positive delta means the model prefers the Germanic word *here*. Splitting the
pairs by corpus frequency ratio gives the whole point of the study:

    - unmatched bin (L much rarer than G): delta is large and positive, but that
      is just frequency.
    - matched bin (L and G comparably frequent): whatever delta survives is the
      *register* effect net of frequency. The contrast between the two bins is
      the result.

Surprisal of a word = -sum_t log P(token_t | left context), summed over the
word's subword tokens. This mirrors the eval harness's causal scoring
(``compute_results.compute_causal_results``: gather target log-probs, sum over a
phrase mask) so our numbers are comparable to the official BLiMP pipeline, but it
returns the continuous score the harness throws away (it only saves the winning
sentence). We locate a word's tokens by a tokenizer-agnostic prefix diff rather
than offset mapping, because the GPT-BERT checkpoints ship a custom tokenizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

# --- Carrier frames, keyed by coarse POS -------------------------------------
#
# Each frame has a single ``{}`` slot. Frames are deliberately bland: a register
# probe must not lean the context toward formal or informal style, or the frame
# itself would leak the answer. The same frames serve both members of a pair
# because synonyms share a POS. Verbs assume the base (lemma) form.

POS_FRAMES: dict[str, list[str]] = {
    "VERB": [
        "They wanted to {} it.",
        "We will {} the plan tomorrow.",
        "I need to {} this first.",
        "She decided to {} the request.",
        "You should {} them soon.",
        "He tried to {} the problem.",
    ],
    "NOUN": [
        "The {} was important.",
        "They talked about the {}.",
        "We need a new {}.",
        "This is the main {}.",
        "Everyone noticed the {}.",
        "I wrote down the {}.",
    ],
    "ADJ": [
        "It was very {}.",
        "The result seemed {}.",
        "That is a {} answer.",
        "We found it {} in the end.",
        "The plan was {} enough.",
        "Her reasons were {}.",
    ],
}

CONTENT_POS = set(POS_FRAMES)


@dataclass
class PairFrames:
    latinate: str
    germanic: str
    pos: str
    frames: list[str]  # frame templates (with {} slot) shared by both words


def pos_tag(word: str, nlp) -> str:
    """Coarse POS of a word in isolation, via spaCy. Returns e.g. 'VERB'.

    Tagging a bare word is noisy, so we tag the word inside a tiny neutral
    carrier ("They {}." / "The {}.") is overkill; spaCy's lemmatiser+tagger on
    the lone token is good enough for high-frequency vocabulary and keeps the
    function pure. Callers should keep only pairs where both members land on the
    *same* content POS.
    """
    doc = nlp(word)
    return doc[0].pos_ if len(doc) else "X"


def build_pair_frames(pairs: pd.DataFrame, nlp) -> list[PairFrames]:
    """Keep pairs whose two members share a content POS, attach carrier frames.

    ``pairs`` needs columns ``latinate`` and ``germanic``. Pairs where the two
    words disagree on POS, or land on a non-content POS (modals, discourse
    markers, function words), are dropped — those can't be swapped cleanly in a
    neutral frame, which is exactly the junk we want gone (shall/must,
    however/yet, effect/choose).
    """
    out: list[PairFrames] = []
    for row in pairs.itertuples():
        lat_pos = pos_tag(row.latinate, nlp)
        ger_pos = pos_tag(row.germanic, nlp)
        if lat_pos != ger_pos or lat_pos not in CONTENT_POS:
            continue
        out.append(
            PairFrames(
                latinate=row.latinate,
                germanic=row.germanic,
                pos=lat_pos,
                frames=POS_FRAMES[lat_pos],
            )
        )
    return out


# --- Surprisal scoring --------------------------------------------------------


class WordScorer:
    """Loads one causal checkpoint and scores a word's surprisal in a frame.

    Mirrors the harness's causal backend: next-token log-probs, summed over the
    target word's subword tokens. Use one instance per ``revision`` (checkpoint).
    """

    def __init__(self, model_name: str, revision: str | None = None, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=True)["input_ids"]

    def surprisal(self, frame: str, word: str) -> float:
        """-log P(word | left context) in ``frame`` (one ``{}`` slot), in nats.

        The word's token span is found by a prefix diff: tokenise the text up to
        the slot, then up to and including the word; the new tokens are the
        word. This is tokenizer-agnostic (no offset mapping needed). Surprisal is
        the summed next-token loss over those positions.
        """
        import torch

        before, after = frame.split("{}")
        # A leading space makes the word a clean BPE unit when it is mid-sentence.
        before_ids = self._ids(before.rstrip()) if before.strip() else self._ids(before)
        # Reconstruct the exact left context as it appears in the full sentence.
        left_text = before
        full_text = before + word + after
        left_ids = self._ids(left_text)
        full_ids = self._ids(full_text)

        # The word occupies the token positions that appear in full but not in the
        # left context; locate them by the longest shared prefix with left_ids.
        start = 0
        for a, b in zip(left_ids, full_ids):
            if a != b:
                break
            start += 1
        # Word tokens run from `start` up to where `after` begins. Find that by
        # matching the suffix (after-text tokens) at the tail.
        after_ids = self._ids((word + after))  # tokens of word+after in context-free form
        # Simpler and robust: take tokens of (left+word) minus (left).
        leftword_ids = self._ids(before + word)
        word_start = 0
        for a, b in zip(left_ids, leftword_ids):
            if a != b:
                break
            word_start += 1
        word_ids = leftword_ids[word_start:]
        n_word = len(word_ids)
        if n_word == 0:
            return float("nan")

        input_ids = torch.tensor([leftword_ids], device=self.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids)
            logits = out["logits"] if isinstance(out, dict) else (out[0] if isinstance(out, tuple) else out.logits)
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)  # [T, V]

        # Token at position i is predicted from logits at position i-1.
        total = 0.0
        for pos in range(word_start, word_start + n_word):
            if pos == 0:  # no left context for the very first token; skip
                continue
            tok = leftword_ids[pos]
            total += log_probs[pos - 1, tok].item()
        return -total  # surprisal in nats

    def pair_delta(self, pf: PairFrames) -> dict:
        """Mean delta = surprisal(L) - surprisal(G) over the pair's frames."""
        lat = [self.surprisal(fr, pf.latinate) for fr in pf.frames]
        ger = [self.surprisal(fr, pf.germanic) for fr in pf.frames]
        deltas = [l - g for l, g in zip(lat, ger) if not (math.isnan(l) or math.isnan(g))]
        return {
            "latinate": pf.latinate,
            "germanic": pf.germanic,
            "pos": pf.pos,
            "surprisal_lat": sum(lat) / len(lat),
            "surprisal_ger": sum(ger) / len(ger),
            "delta": sum(deltas) / len(deltas) if deltas else float("nan"),
            "n_frames": len(deltas),
        }
