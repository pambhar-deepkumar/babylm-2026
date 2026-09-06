import sentencepiece as spm
import os

DATA_DIR = os.path.expanduser("~/git/nour/babylm_2026/data/budget100m")
TOK_DIR = os.path.expanduser("~/git/nour/babylm_2026/tokenizers")

special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

def train(input_files, model_prefix, vocab_size, character_coverage):
    combined_input = ",".join(input_files)
    print(f"Training {model_prefix} (vocab={vocab_size}, coverage={character_coverage}) from {input_files} ...")
    spm.SentencePieceTrainer.train(
        input=combined_input,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        num_threads=8,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=special_tokens,
        character_coverage=character_coverage,
    )
    print(f"Saved {model_prefix}.model")

# Joint tokenizer: all 3 languages combined, 24k vocab (3 x 8k), needs Chinese coverage
train(
    [f"{DATA_DIR}/eng_train_33m.txt", f"{DATA_DIR}/nld_train_33m.txt", f"{DATA_DIR}/zho_train_33m.txt"],
    os.path.join(TOK_DIR, "multi_tokenizer_24k"),
    24000,
    0.9995,
)

# English+Dutch combined tokenizer: 16k vocab (2 x 8k), Latin-only so full coverage
train(
    [f"{DATA_DIR}/eng_train_33m.txt", f"{DATA_DIR}/nld_train_33m.txt"],
    os.path.join(TOK_DIR, "engnld_tokenizer_16k"),
    16000,
    1.0,
)

print("Done.")
