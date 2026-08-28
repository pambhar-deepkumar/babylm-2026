"""Build a fixed probe set for the AoA / acquisition-trajectory analysis.

For each curated (Latinate, Germanic) synonym pair, sample up to K natural
contexts from the ORIGINAL corpus (bb26_en.train) in which the target word
occurs, storing the left context + the exact surface form. The same probe set
is reused across all arms (A/B/C/D) so surprisal curves are directly comparable
— the contexts are identical; only the trained model differs.

Output: data/aoa_probes.json  (list of {word, klass, pair_id, ctx_left, target})

Run once on the login node:
    python scripts/register/build_aoa_probes.py --corpus data/bb26_en.train --k 30
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Curated (Latinate, Germanic) synonym pairs (meaning-preserving, low-polysemy).
PAIRS = [
    ("comprehend", "understand"), ("acquire", "get"), ("obtain", "get"),
    ("require", "need"), ("desire", "want"), ("provide", "give"),
    ("receive", "get"), ("attempt", "try"), ("assist", "help"),
    ("construct", "build"), ("demonstrate", "show"), ("purchase", "buy"),
    ("depart", "leave"), ("permit", "let"), ("allow", "let"), ("request", "ask"),
    ("inquire", "ask"), ("reply", "answer"), ("respond", "answer"),
    ("conceal", "hide"), ("discover", "find"), ("select", "choose"),
    ("repair", "mend"), ("possess", "have"), ("conclude", "end"),
    ("commence", "start"), ("observe", "see"), ("remain", "stay"),
    ("retain", "keep"), ("maintain", "keep"), ("perform", "do"),
    ("infant", "baby"), ("beverage", "drink"), ("residence", "home"),
    ("error", "mistake"), ("illness", "sickness"), ("remainder", "rest"),
    ("assistance", "help"),
    ("enormous", "huge"), ("gigantic", "huge"), ("immense", "huge"),
    ("massive", "huge"), ("rapid", "fast"), ("difficult", "hard"),
    ("sufficient", "enough"), ("entire", "whole"), ("numerous", "many"),
    ("multiple", "many"), ("additional", "more"), ("ancient", "old"),
    ("previous", "earlier"), ("prior", "earlier"), ("initial", "first"),
    ("subsequent", "next"), ("fortunate", "lucky"), ("strange", "odd"),
    ("peculiar", "odd"), ("ill", "sick"), ("secure", "safe"), ("final", "last"),
    ("however", "but"), ("therefore", "so"), ("perhaps", "maybe"),
    ("frequently", "often"), ("previously", "before"),
    ("occasionally", "sometimes"), ("currently", "now"), ("shall", "will"),
]

# max characters of left context to keep (bounds forward-pass cost)
MAX_CTX_CHARS = 240


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/bb26_en.train")
    p.add_argument("--out", default="data/aoa_probes.json")
    p.add_argument("--k", type=int, default=30, help="max contexts per word")
    p.add_argument("--min_ctx_words", type=int, default=3,
                   help="skip occurrences with too little left context")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # word -> (klass, pair_id); a Germanic word can belong to several pairs —
    # keep the first (pair_id only labels the matched-pair grouping).
    meta: dict[str, tuple[str, int]] = {}
    for pid, (lat, ger) in enumerate(PAIRS):
        meta.setdefault(lat, ("Latinate", pid))
        meta.setdefault(ger, ("Germanic", pid))

    patterns = {w: re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in meta}
    counts = {w: 0 for w in meta}
    probes: list[dict] = []

    with open(args.corpus, encoding="utf-8") as fh:
        for line in fh:
            if all(c >= args.k for c in counts.values()):
                break
            for w, (klass, pid) in meta.items():
                if counts[w] >= args.k:
                    continue
                m = patterns[w].search(line)
                if not m:
                    continue
                start = m.start()
                # include the leading space in the target surface (byte-BPE
                # attaches it to the following token), split the context there.
                if start > 0 and line[start - 1] == " ":
                    ctx_left = line[:start - 1]
                    target = line[start - 1:m.end()]
                else:
                    ctx_left = line[:start]
                    target = line[start:m.end()]
                if len(ctx_left.split()) < args.min_ctx_words:
                    continue
                probes.append({
                    "word": w.lower(),
                    "klass": klass,
                    "pair_id": pid,
                    "ctx_left": ctx_left[-MAX_CTX_CHARS:],
                    "target": target,
                })
                counts[w] += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(probes, fh, ensure_ascii=False)

    covered = sum(1 for c in counts.values() if c > 0)
    low = {w: c for w, c in sorted(counts.items()) if c < args.k}
    print(f"probes: {len(probes)} | words covered: {covered}/{len(meta)} | target k={args.k}")
    print(f"under-k words ({len(low)}): {low}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
