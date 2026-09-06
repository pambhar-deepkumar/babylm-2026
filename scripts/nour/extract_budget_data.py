import sentencepiece as spm
import os

TOK_DIR = os.path.expanduser("~/git/nour/babylm_2026/tokenizers")
DATA_DIR = os.path.expanduser("~/git/nour/babylm_2026/data")
OUT_DIR = os.path.expanduser("~/git/nour/babylm_2026/data/budget100m")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_TRAIN_TOKENS = 33_000_000
TARGET_VALID_TOKENS = 10_000_000

configs = {
    "eng": {"tokenizer": f"{TOK_DIR}/eng_tokenizer_8k.model", "source": f"{DATA_DIR}/eng/eng_train.txt"},
    "nld": {"tokenizer": f"{TOK_DIR}/nld_spm_8k.model", "source": f"{DATA_DIR}/nld/texts2.txt"},
    "zho": {"tokenizer": f"{TOK_DIR}/zho_spm_8k.model", "source": f"{DATA_DIR}/zho/texts1.txt"},
}

def extract(tokenizer_path, source_path, target_tokens, out_path, skip_first_n_tokens=0):
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    total = 0
    skipped = 0
    n_lines = 0
    with open(source_path, encoding="utf-8", errors="ignore") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line_stripped = line.rstrip("\n")
            if not line_stripped:
                continue
            n_tok = len(sp.encode(line_stripped))
            if skipped < skip_first_n_tokens:
                skipped += n_tok
                continue
            f_out.write(line_stripped + "\n")
            total += n_tok
            n_lines += 1
            if total >= target_tokens:
                break
    return n_lines, total

print("=== Training slices (33M tokens each) ===")
for lang, cfg in configs.items():
    out_path = f"{OUT_DIR}/{lang}_train_33m.txt"
    n_lines, total = extract(cfg["tokenizer"], cfg["source"], TARGET_TRAIN_TOKENS, out_path)
    print(f"{lang}: {n_lines} lines, {total} tokens -> {out_path}")

print("=== English validation slice (10M tokens, disjoint from training) ===")
val_out = f"{OUT_DIR}/eng_valid_10m.txt"
n_lines, total = extract(configs["eng"]["tokenizer"], configs["eng"]["source"], TARGET_VALID_TOKENS, val_out, skip_first_n_tokens=TARGET_TRAIN_TOKENS)
print(f"eng_valid: {n_lines} lines, {total} tokens -> {val_out}")

print("Done.")
