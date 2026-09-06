#!/usr/bin/env python3
"""
train_tokenizer.py - Train a SentencePiece tokenizer for BabyLM 2026
compatible with DebertaV2Tokenizer in train_mask.py.

Output filename pattern: <output>/<lang>_<name>_<vocab_size//1000>k.model
(e.g. nld_tokenizer_32k.model, eng_tokenizer_4k.model).

Usage:
    python scripts/train_tokenizer.py --input src/babylm_2026/data/nld/texts1.txt \
                                       --output src/babylm_2026/tokenizers \
                                       --lang nld --vocab_size 32000

    python scripts/train_tokenizer.py --input src/babylm_2026/data/zho/texts1.txt \
                                       --output src/babylm_2026/tokenizers \
                                       --lang zho --vocab_size 16000

    python scripts/train_tokenizer.py --input src/babylm_2026/data/eng/eng_train.txt \
                                       --output src/babylm_2026/tokenizers \
                                       --lang eng --vocab_size 40000 --name tokenizer
"""

import sentencepiece as spm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input",      required=True,  help="Path to training text file")
parser.add_argument("--output",     required=True,  help="Output directory (model saved as <dir>/<lang>_<name>_<vocab_size//1000>k.model)")
parser.add_argument("--lang",       required=True,  help="eng, nld, or zho")
parser.add_argument("--name",       default="tokenizer",
                    help="Base name used in the output filename, e.g. '<lang>_<name>_<vocab_size//1000>k' (default: 'tokenizer')")
parser.add_argument("--vocab_size", type=int, default=40000,
                    help="Vocabulary size. Keep at 40000 to match bb24.model (default).")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
model_prefix = os.path.join(args.output, f"{args.lang}_{args.name}_{args.vocab_size // 1000}k")

# These special tokens must be INSIDE the vocabulary at known positions
# so that DebertaV2Tokenizer can find them without exceeding vocab_size.
# The ordering: <pad>=0, <unk>=1, <s>=2, </s>=3, then user_defined_symbols start at 4.
# [PAD] etc. are added as extra user-defined symbols inside the vocabulary.
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

common_kwargs = dict(
    input=args.input,
    model_prefix=model_prefix,
    vocab_size=args.vocab_size,
    model_type="bpe",
    num_threads=8,
    pad_id=0,           # <pad> at position 0
    unk_id=1,           # <unk> at position 1
    bos_id=2,           # <s>   at position 2
    eos_id=3,           # </s>  at position 3
    user_defined_symbols=special_tokens,  # [PAD],[UNK],[CLS],[SEP],[MASK] inside vocab
)

if args.lang == "zho":
    spm.SentencePieceTrainer.train(
        **common_kwargs,
        character_coverage=0.9995,   # high coverage needed for Chinese characters
    )
else:
    spm.SentencePieceTrainer.train(
        **common_kwargs,
        character_coverage=1.0,      # full coverage for Latin-script languages (eng, nld)
    )

print(f"")
print(f"Tokenizer saved:")
print(f"  {model_prefix}.model  ← pass this to --tokenizer in run.sh")
print(f"  {model_prefix}.vocab")
print(f"")

# Quick sanity check
import sentencepiece as spm_check
sp = spm_check.SentencePieceProcessor()
sp.Load(model_prefix + ".model")
print(f"Sanity check:")
print(f"  vocab_size : {sp.get_piece_size()}")
print(f"  pad_id     : {sp.pad_id()}  (must be < vocab_size)")
print(f"  [PAD] id   : {sp.piece_to_id('[PAD]')}")
print(f"  [CLS] id   : {sp.piece_to_id('[CLS]')}")
print(f"  [SEP] id   : {sp.piece_to_id('[SEP]')}")
print(f"  [MASK] id  : {sp.piece_to_id('[MASK]')}")
assert sp.pad_id() < sp.get_piece_size(), "ERROR: pad_id >= vocab_size — training failed!"
print(f"  ✓ pad_id is within vocab_size — safe to use with train_mask.py")
