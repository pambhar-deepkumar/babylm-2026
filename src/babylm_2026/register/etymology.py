"""Per-word etymology labelling: Germanic vs Latinate.

This is the shared Germanic/Latinate labeller used by the age-of-acquisition
(AoA) analysis and the corpus register labelling. It reuses the machinery
calibrated in notebooks 01 and 03:

  - EtymoLink (Gao & Sun, ACL 2024) as the gold source. Origin is buried in a
    derivation chain in the `sorted` column, e.g. ``(ask_E, ascian_OE);...``;
    we read the immediate ancestor's language code and bucket it.
  - Macro-Etym as a fallback for words EtymoLink doesn't cover. It is noisy
    (notebook 03: Latinate F1 0.59, Germanic F1 0.76, population-weighted 0.63)
    and silently returns "other" for words it doesn't recognise (e.g. "buy",
    "freedom"), so a label's `source` is always recorded for honesty.

Labels are therefore a *lower-confidence* signal for macroetym-only words; keep
the `etym_source` column and report the gold/fallback split downstream.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import csv
from collections import Counter, defaultdict, deque
from pathlib import Path

import pandas as pd
import requests

# --- Macro-Etym (CLI fallback) -------------------------------------------------

# Resolve the binary inside the active venv rather than via PATH (notebook 03).
MACROETYM_BIN = str(Path(sys.executable).parent / "macroetym")


def macroetym_single_word(word: str) -> str:
    """Macro-Etym's family verdict for one word: 'Latinate', 'Germanic', or 'other'.

    In-process re-implementation of the CLI's single-word verdict (validated to
    match the CLI exactly on test words: purchase/comprehend/liberty=Latinate,
    understand/ask=Germanic, buy/freedom=other). ~1000x faster than spawning the
    CLI per word, which matters when labelling thousands of corpus words.

    Mechanism: look up the word's immediate parent languages in macroetym's
    Etymological Wordnet (ignoring affixes and current-language ancestors), map
    each to a family via ``Text.langDict``, and return the dominant family.
    Returns 'other' when no parent maps to Germanic/Latinate (incl. words the
    resource simply doesn't cover — coverage on core English vocab is poor).
    """
    from macroetym.main import Text, Word

    stats = Word(word.lower(), lang="eng").parent_languages.stats  # {lang: pct}
    fam_tot: dict[str, float] = {}
    for lang, pct in stats.items():
        fam = _LANG2FAM.get(lang)
        if fam in ("Germanic", "Latinate"):
            fam_tot[fam] = fam_tot.get(fam, 0.0) + pct
    if not fam_tot:
        return "other"
    return max(fam_tot.items(), key=lambda kv: kv[1])[0]


# language code -> family, inverted from macroetym's Text.langDict (built lazily).
class _Lang2FamProxy(dict):
    def __missing__(self, key):
        from macroetym.main import Text

        self.update({l: fam for fam, ls in Text.langDict.items() for l in ls})
        return self.get(key)


_LANG2FAM = _Lang2FamProxy()


# --- EtymoLink (gold) ----------------------------------------------------------

ETYMOLINK_URL = (
    "https://raw.githubusercontent.com/yuan-w-gao/etymolink/main/annotated_formatted.csv"
)

# Language codes collapsed to two buckets matching Macro-Etym's families (nb 03).
GERMANIC_CODES = {
    "OE", "PGer", "WGer", "PWGer", "Ger", "ModG", "HGer", "OHGer", "LGer", "GE",
    "ONor", "Dut", "ODut", "ADut", "Du", "OFri", "Scan", "Dan", "ODan", "Swe",
    "Norw", "Nor", "Flem", "Go", "Frank", "pgER",
}
LATINATE_CODES = {
    "L", "OL", "VL", "LateL", "MediL", "ModL", "ChuL", "preL", "EL", "AL",
    "F", "OF", "MF", "AF", "ONF", "AngF", "Fran", "FCan", "LF", "I",
    "Span", "ASpan", "MexSpan", "CuSpan", "Port", "Por", "Cata", "Pro", "OPro",
    "Ro", "GaRo", "Gas",
}

_FIRST_TUPLE_RE = re.compile(r"\(\s*([^,_]+)_(\w+)\s*,\s*([^,_)]+)_(\w+)\s*\)")


def _immediate_origin(sorted_chain: str) -> str | None:
    """Immediate-ancestor language code from EtymoLink's first tuple, or None."""
    if not isinstance(sorted_chain, str):
        return None
    m = _FIRST_TUPLE_RE.search(sorted_chain)
    if not m:
        return None
    _, head_lang, _, ancestor_lang = m.groups()
    if head_lang != "E":  # first tuple should be (word_E, ancestor_LANG)
        return None
    return ancestor_lang


def _bucket(code: str | None) -> str | None:
    if code in GERMANIC_CODES:
        return "Germanic"
    if code in LATINATE_CODES:
        return "Latinate"
    return None  # Greek, Arabic, PIE, href, etc. — excluded from the binary.


def load_etymolink_gold(url: str = ETYMOLINK_URL) -> dict[str, str]:
    """Download EtymoLink and return {lowercased word: 'Germanic'|'Latinate'}.

    Words whose immediate ancestor is neither family (Greek, PIE, ...) are dropped.
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    import io

    df = pd.read_csv(io.StringIO(resp.text))[["word", "sorted"]].copy()
    df["word"] = df["word"].astype(str).str.lower().str.strip()
    df["label"] = df["sorted"].map(_immediate_origin).map(_bucket)
    df = df.dropna(subset=["label"]).drop_duplicates("word")
    return dict(zip(df["word"], df["label"]))


# --- Hybrid labeller -----------------------------------------------------------

def label_word(word: str, gold: dict[str, str]) -> tuple[str, str]:
    """Return (label, source). EtymoLink gold if available, else macroetym fallback.

    label  ∈ {'Germanic', 'Latinate', 'other'}
    source ∈ {'etymolink', 'macroetym', 'none'}
    """
    w = word.lower().strip()
    if w in gold:
        return gold[w], "etymolink"
    me = macroetym_single_word(w)
    return (me, "macroetym") if me != "other" else ("other", "none")


# --- Corpus frequency ----------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z]+")


def corpus_word_frequencies(corpus_dir: Path) -> Counter[str]:
    """Whitespace/alpha word-form frequencies over every *.train.txt in the dir.

    Fast (no spaCy): lowercased, alphabetic tokens only. This is the word-form
    frequency in the model's *training* corpus — the key confound to control for
    in the AoA regression.
    """
    counts: Counter[str] = Counter()
    for path in sorted(Path(corpus_dir).glob("*.train.txt")):
        text = path.read_text(encoding="utf-8").lower()
        counts.update(_TOKEN_RE.findall(text))
    return counts


def build_wordlist(
    corpus_dir: Path,
    top_n: int = 5000,
    min_len: int = 3,
    min_freq: int = 5,
) -> pd.DataFrame:
    """Build an etymology-tagged candidate word list from the corpus.

    Takes the top-N most frequent content-ish word forms (drops spaCy stopwords,
    short words, rare words), labels each Germanic/Latinate via the hybrid
    labeller, and annotates log-frequency and length.

    Returns all rows (including 'other'); filter to Germanic/Latinate downstream.
    Columns: word, freq, log_freq, length, etym_label, etym_source.
    """
    import math

    from spacy.lang.en.stop_words import STOP_WORDS

    freqs = corpus_word_frequencies(corpus_dir)
    candidates = [
        (w, c)
        for w, c in freqs.most_common()
        if len(w) >= min_len and c >= min_freq and w not in STOP_WORDS
    ][:top_n]

    gold = load_etymolink_gold()
    rows = []
    for w, c in candidates:
        label, source = label_word(w, gold)
        rows.append(
            {
                "word": w,
                "freq": c,
                "log_freq": math.log10(c),
                "length": len(w),
                "etym_label": label,
                "etym_source": source,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# EtymDB-backed labeller
# =============================================================================
# The EtymoLink/Macro-Etym labeller above is high-precision but very low-recall
# (~99% of corpus types come back "other" — i.e. "unknown", not "non-Latinate").
# This section adds a higher-coverage labeller layered as:
#
#   1. frozen LLM labels   — a hand-frozen one-shot pass over the high-frequency
#      content words EtymDB cannot reach (its chains dead-end at Middle English).
#   2. EtymDB ancestry      — trace each English word up the Wiktionary-derived
#      etymology graph (inh/bor/der edges, never `cog` cognates) to its nearest
#      Germanic or Latinate ancestor; homograph conflicts resolved toward the
#      native (`inh`) sense, otherwise skipped.
#   3. lemma fallback       — retry the lemma when the surface form is unlabelled.
#   4. affix backfill       — conservative Latinate suffixes, with a Germanic-
#      suffix guard (-wise/-ness/...).
#
# Calibrated and validated in scripts/etymdb_label*.py (precision 98.9% vs the
# EtymoLink gold; Latinate token coverage 0.42% -> ~10%).

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ETYMDB_DIR = _REPO_ROOT / "data" / "external" / "etymdb"
_LLM_LABELS_CSV = _REPO_ROOT / "data" / "derived" / "llm_labels.csv"

# Wiktionary language codes -> family.
_GERMANIC_LANGS = {
    "ang", "gem-pro", "gmw-pro", "goh", "gmh", "de", "nl", "dum", "odt", "non",
    "got", "is", "da", "sv", "no", "nb", "nn", "fy", "ofs", "frr", "stq", "nds",
    "sco", "gml", "osx", "gem", "gmq-pro", "nrn",
}
_LATINATE_LANGS = {
    "la", "la-vul", "la-lat", "la-med", "la-new", "la-ecc", "la-ren", "la-cla",
    "itc-pro", "itc-ola", "fro", "fro-nor", "frm", "fr", "xno", "nrf", "pro",
    "oco", "ca", "it", "es", "pt", "ro", "roa-opt", "osp", "gl", "vec", "scn",
    "nap", "lld", "frp",
}
_NEUTRAL_LANGS = {"en", "enm"}                 # English stages: keep climbing
_FOLLOW_RELS = {"inh", "bor", "der", "der(s)", "der(p)", "cmpd+bor"}  # never 'cog'

# Words we refuse to label: clitic/n't fragments + audit-flagged artifacts.
_DROP_TOKENS = (
    {"isn", "didn", "doesn", "wasn", "weren", "wouldn", "couldn", "shouldn",
     "hasn", "hadn", "haven", "mustn", "needn", "aren", "ain", "shan", "daren"}
    | {"ll", "ve", "re", "em"}
    | {"chi", "mot"}
)

# Hand-verified corrections for high-frequency EtymDB homograph errors: common
# Germanic words whose native chain dead-ends in EtymDB while a rare Latinate
# homograph completes (so the trace wrongly returns Latinate). Checked first.
_OVERRIDES = {
    "don": "other",        # 'don't' fragment; EtymDB -> Latin dominus
    "mean": "Germanic",    # 'to intend'; loses to Latinate 'average' homograph
    "means": "Germanic",
    "far": "Germanic",     # OE feor; EtymDB completes a Latin 'far' (grain)
    "found": "Germanic",   # past of find; loses to Latinate 'establish'
    "seen": "Germanic",    # past of see; lemma 'see' hits a Latin homograph
    "de": "other",         # foreign particle / name artifact
    "best": "Germanic",    # OE betst
    "lost": "Germanic",    # past of lose (OE losian); lemma hits a Latin homograph
    "ball": "Germanic",    # ON bollr (round toy); loses to French 'bal' (dance)
    "comes": "Germanic",   # 3sg of come; EtymDB matches Latin 'comes' (count)
    "sea": "Germanic",     # OE sae
    "named": "Germanic",   # of name (OE nama); lemma hits Spanish 'name' (yam)
}

_LAT_SUFFIX = [
    "tion", "sion", "ity", "ousness", "ous", "ment", "ance", "ence", "ive",
    "ization", "isation", "ize", "ise", "ism", "ist", "able", "ible", "ary",
    "ory", "ition", "actic",
]
_GERMANIC_SUFFIX = ("wise", "ness", "less", "ful", "ship", "hood", "dom",
                    "ward", "wards", "some", "like")


def load_etymdb(etymdb_dir: Path = _ETYMDB_DIR):
    """Return (idx2lang, en_index, child2parents) from EtymDB CSVs.

    en_index keys lowercased English lexemes -> [indices], indexing only
    lowercase senses so capitalised proper nouns ("Side", "March") don't
    case-fold onto common words. child2parents keeps only ancestry edges.
    """
    idx2lang: dict[str, str] = {}
    en_index: dict[str, list[str]] = defaultdict(list)
    with open(etymdb_dir / "etymdb_values.csv") as f:
        for r in csv.reader(f, delimiter="\t"):
            if len(r) < 4:
                continue
            i, lang, _, lex = r[0], r[1], r[2], r[3]
            idx2lang[i] = lang
            if lang == "en" and lex and lex == lex.lower():
                en_index[lex].append(i)
    child2par: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with open(etymdb_dir / "etymdb_links_info.csv") as f:
        for r in csv.reader(f, delimiter="\t"):
            if len(r) < 3:
                continue
            rel, child, par = r[0], r[1], r[2]
            if rel in _FOLLOW_RELS:
                child2par[child].append((rel, par))
    return idx2lang, en_index, child2par


def load_llm_labels(path: Path = _LLM_LABELS_CSV) -> dict[str, str]:
    """Frozen LLM labels for the high-frequency residue. {word: family}."""
    if not Path(path).exists():
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["word"].lower()] = r["llm_label"]
    return out


def _trace_sense(start, idx2lang, child2par, max_depth=12):
    """Nearest Germanic/Latinate ancestor of one sense + its first-hop relation."""
    seen = {start}
    frontier = deque()
    for rel, par in child2par.get(start, []):
        if par not in seen:
            seen.add(par)
            frontier.append((par, rel, 1))
    while frontier:
        node, first_rel, depth = frontier.popleft()
        fam = idx2lang.get(node)
        if fam in _GERMANIC_LANGS:
            return "Germanic", first_rel
        if fam in _LATINATE_LANGS:
            return "Latinate", first_rel
        if fam in _NEUTRAL_LANGS and depth < max_depth:
            for rel, par in child2par.get(node, []):
                if par not in seen:
                    seen.add(par)
                    frontier.append((par, first_rel, depth + 1))
    return "other", None


def trace_family(word, idx2lang, en_index, child2par) -> str:
    """EtymDB family for a word: Germanic/Latinate/conflict/other.

    Aggregates all (lowercase) senses; on a Germanic/Latinate disagreement,
    prefers the natively-inherited (`inh`) sense, else returns 'conflict'.
    """
    starts = en_index.get(word.lower())
    if not starts:
        return "other"
    results = [_trace_sense(s, idx2lang, child2par) for s in starts]
    fams = {f for f, _ in results if f != "other"}
    if not fams:
        return "other"
    if len(fams) == 1:
        return next(iter(fams))
    inh_fams = {f for f, rel in results if rel == "inh" and f != "other"}
    if len(inh_fams) == 1:
        return next(iter(inh_fams))
    return "conflict"


def affix_label(word) -> str | None:
    """Conservative Latinate-suffix verdict, guarded against Germanic suffixes."""
    w = word.lower()
    if len(w) < 6 or w.endswith(_GERMANIC_SUFFIX):
        return None
    for suf in _LAT_SUFFIX:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return "Latinate"
    return None


class EtymologyLabeler:
    """Combined Germanic/Latinate labeller (LLM -> EtymDB -> lemma -> affix).

    label(word) -> (family, source) where family in {Germanic, Latinate, other}
    and source records which layer decided (llm/etymdb/lemma/affix/excluded/
    conflict/none). Greek-origin LLM labels map to 'other' for the binary.
    """

    def __init__(self, etymdb_dir=_ETYMDB_DIR, llm_labels_path=_LLM_LABELS_CSV,
                 use_affix=True, use_llm=True, exclude=None):
        self.idx2lang, self.en_index, self.child2par = load_etymdb(etymdb_dir)
        self.llm = load_llm_labels(llm_labels_path) if use_llm else {}
        self.use_affix = use_affix
        if exclude is None:
            from spacy.lang.en.stop_words import STOP_WORDS
            exclude = set(STOP_WORDS)
        self.exclude = set(exclude) | _DROP_TOKENS
        self._nlp = None

    def _lemma(self, word):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        doc = self._nlp(word)
        return doc[0].lemma_.lower() if len(doc) else word

    def label(self, word, lemma=None) -> tuple[str, str]:
        w = word.lower()
        if w in _OVERRIDES:                                # 0. hand corrections
            fam = _OVERRIDES[w]
            return (fam if fam in ("Germanic", "Latinate") else "other", "override")
        if w in self.llm:                                  # 1. frozen LLM labels
            fam = self.llm[w]
            return (fam if fam in ("Germanic", "Latinate") else "other", "llm")
        if w in self.exclude:                              # function words / junk
            return "other", "excluded"
        fam = trace_family(w, self.idx2lang, self.en_index, self.child2par)  # 2.
        if fam in ("Germanic", "Latinate"):
            return fam, "etymdb"
        if fam == "conflict":
            return "other", "conflict"
        lem = lemma if lemma is not None else self._lemma(w)   # 3. lemma fallback
        if lem and lem != w:
            fam = trace_family(lem, self.idx2lang, self.en_index, self.child2par)
            if fam in ("Germanic", "Latinate"):
                return fam, "lemma"
        if self.use_affix:                                 # 4. affix backfill
            a = affix_label(w)
            if a:
                return a, "affix"
        return "other", "none"


def _cased_counts(corpus_dir: Path) -> Counter[str]:
    """Case-preserving alphabetic token counts, for the proper-noun heuristic."""
    cre = re.compile(r"[A-Za-z]+")
    counts: Counter[str] = Counter()
    for path in sorted(Path(corpus_dir).glob("*.train.txt")):
        counts.update(cre.findall(path.read_text(encoding="utf-8")))
    return counts


def build_etymology_wordlist(corpus_dir: Path, top_n: int | None = None,
                             labeler: EtymologyLabeler | None = None) -> pd.DataFrame:
    """Label every corpus type with the combined labeller.

    Columns: word, freq, global_rank, pct_of_corpus, cap_ratio, likely_proper,
    etym_label, etym_source. `likely_proper` flags proper nouns (capitalised
    >=50% of the time in the raw corpus).
    """
    import math

    freqs = corpus_word_frequencies(corpus_dir)
    total = sum(freqs.values())
    ranked = freqs.most_common(top_n) if top_n else freqs.most_common()
    types = [w for w, _ in ranked]

    cased = _cased_counts(corpus_dir)

    def cap_ratio(w):
        lo = cased.get(w.lower(), 0)
        ti = cased.get(w.capitalize(), 0)
        up = cased.get(w.upper(), 0)
        tot = lo + ti + up
        return (ti + up) / tot if tot else 0.0

    if labeler is None:
        labeler = EtymologyLabeler()

    # batch-lemmatise once (much faster than per-word spaCy calls)
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    lemmas = {doc.text: (doc[0].lemma_.lower() if len(doc) else doc.text)
              for doc in nlp.pipe(types, batch_size=2000)}

    rows = []
    for rank, w in enumerate(types):
        lab, src = labeler.label(w, lemma=lemmas.get(w))
        cr = cap_ratio(w)
        rows.append({
            "word": w,
            "freq": freqs[w],
            "global_rank": rank,
            "pct_of_corpus": round(100 * freqs[w] / total, 5),
            "cap_ratio": round(cr, 3),
            "likely_proper": cr >= 0.5,
            "etym_label": lab,
            "etym_source": src,
        })
    return pd.DataFrame(rows)
