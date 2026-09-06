"""
Pre-segment each language's training text with its own (already-trained)
SentencePiece tokenizer into piece-strings, concatenate across languages,
then train one word-level meta-tokenizer over the combined piece vocabulary.

This produces a single .model file that train_mask.py can load completely
normally -- the real per-language tokenization already happened in this
preprocessing step, so the meta-tokenizer only does whitespace-delimited
id lookup (model_type=word), no further subword splitting.

Note: SentencePiece's word mode mandatorily prepends its own "_" (U+2581)
boundary marker to every word (escape_whitespaces cannot be disabled in
word mode). Since our BPE pieces already contain that same "_" character
internally (from the first tokenization pass), joining them with plain
spaces causes a collision that silently shrinks the realized vocabulary
below what we expect. Fix: rename the embedded "_" to a different
placeholder before joining, so there's no ambiguity with SentencePiece's
own boundary marker.
"""
import argparse
import os
import re
import sentencepiece as spm

SPM_SPACE = "▁"  # the "_" SentencePiece uses internally
PLACEHOLDER = "<SP>"

parser = argparse.ArgumentParser()
parser.add_argument("--lang", action="append", required=True, help="Language code, repeatable (e.g. --lang eng --lang nld --lang zho)")
parser.add_argument("--tokenizer", action="append", required=True, help="Path to that language's SentencePiece .model, same order as --lang")
parser.add_argument("--train_file", action="append", required=True, help="Path to that language's raw training text, same order as --lang")
parser.add_argument("--valid_lang", required=True, help="Which language is used for validation (must be one of --lang)")
parser.add_argument("--valid_file", required=True, help="Path to the raw validation text for --valid_lang")
parser.add_argument("--out_dir", required=True)
parser.add_argument("--out_prefix", required=True, help="e.g. run2 or run3")
args = parser.parse_args()

assert len(args.lang) == len(args.tokenizer) == len(args.train_file)
os.makedirs(args.out_dir, exist_ok=True)

lang_to_tok = dict(zip(args.lang, args.tokenizer))
lang_to_train = dict(zip(args.lang, args.train_file))

def presegment(tokenizer_path, input_path, output_path):
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    n_lines = 0
    with open(input_path, encoding="utf-8", errors="ignore") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                continue
            pieces = sp.encode(line, out_type=str)
            pieces = [p.replace(SPM_SPACE, PLACEHOLDER) for p in pieces]
            f_out.write(" ".join(pieces) + "\n")
            n_lines += 1
    return n_lines

# 1. Pre-segment each language's training text with its own tokenizer.
train_pieces_path = os.path.join(args.out_dir, f"{args.out_prefix}_pieces_train.txt")
total_lines = 0
with open(train_pieces_path, "w", encoding="utf-8") as out_f:
    for lang in args.lang:
        tmp_path = os.path.join(args.out_dir, f"{args.out_prefix}_{lang}_pieces_tmp.txt")
        n_lines = presegment(lang_to_tok[lang], lang_to_train[lang], tmp_path)
        print(f"  {lang}: {n_lines} lines pre-segmented with {lang_to_tok[lang]}")
        with open(tmp_path, encoding="utf-8") as f:
            for line in f:
                out_f.write(line)
        os.remove(tmp_path)
        total_lines += n_lines
print(f"Combined training pieces: {total_lines} lines -> {train_pieces_path}")

# 2. Pre-segment the validation text with whichever tokenizer covers its language in this run.
valid_pieces_path = os.path.join(args.out_dir, f"{args.out_prefix}_pieces_valid.txt")
n_valid_lines = presegment(lang_to_tok[args.valid_lang], args.valid_file, valid_pieces_path)
print(f"Validation pieces ({args.valid_lang}): {n_valid_lines} lines -> {valid_pieces_path}")

# 3. Compute the exact distinct-piece count actually realized in the combined training corpus,
#    so the word-level meta-tokenizer's vocab_size covers every piece with no truncation/<unk> loss.
distinct_pieces = set()
with open(train_pieces_path, encoding="utf-8") as f:
    for line in f:
        distinct_pieces.update(line.split())
n_special = 9  # 4 builtin (pad/unk/bos/eos) + 5 user-defined ([PAD]/[UNK]/[CLS]/[SEP]/[MASK])
meta_vocab_size = len(distinct_pieces) + n_special
print(f"Distinct pieces realized in corpus: {len(distinct_pieces)} -> meta vocab_size={meta_vocab_size}")

# 4. Train the word-level meta-tokenizer (whitespace-delimited lookup only, no subword splitting).
#    Auto-retry with the corrected vocab_size if SentencePiece reports a different exact count
#    than we computed (e.g. due to special-token overlap with corpus content).
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
meta_prefix = os.path.join(args.out_dir, f"{args.out_prefix}_meta")
train_kwargs = dict(
    input=train_pieces_path,
    model_prefix=meta_prefix,
    model_type="word",
    num_threads=8,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    user_defined_symbols=special_tokens,
    character_coverage=1.0,
    max_sentence_length=200000,  # default 4192 silently drops long lines, causing vocab_size mismatches
    normalization_rule_name="identity",  # disable NFKC normalization, which can alter piece strings
)
try:
    spm.SentencePieceTrainer.train(vocab_size=meta_vocab_size, **train_kwargs)
except RuntimeError as e:
    m = re.search(r"set it to a value <= (\d+)", str(e))
    if not m:
        raise
    corrected = int(m.group(1))
    print(f"vocab_size mismatch ({meta_vocab_size} requested) -- retrying with corrected vocab_size={corrected}")
    spm.SentencePieceTrainer.train(vocab_size=corrected, **train_kwargs)
print(f"Meta-tokenizer saved: {meta_prefix}.model")

# 5. Sanity check: every distinct piece from the corpus must be present in the meta-vocab (no <unk> fallback).
#    SentencePiece's word mode prepends its own boundary marker to every stored piece, so account for that.
sp_meta = spm.SentencePieceProcessor(model_file=meta_prefix + ".model")
meta_vocab = set(sp_meta.id_to_piece(i) for i in range(sp_meta.get_piece_size()))
expected = set(SPM_SPACE + p for p in distinct_pieces)
missing = expected - meta_vocab
print(f"Sanity check: {len(missing)} pieces missing from meta-vocab (should be 0)")
if missing:
    print(f"  Sample missing: {list(missing)[:10]}")
print("Done.")
