# Evaluation

Every checkpoint is scored with the official BabyLM 2026 pipeline,
vendored as a git submodule at `third_party/babylm-eval/`. Our track is
**strict-small**, so we only use `third_party/babylm-eval/strict/`.

## One-time setup

```bash
make setup
```

This initialises the submodule and installs its requirements alongside
ours. If you cloned without `--recursive`, run it now.

## Evaluating a checkpoint

```bash
make eval MODEL=<hf_id_or_local_path> [BACKEND=mlm]
```

`MODEL` accepts either a HuggingFace Hub model ID or a local checkpoint
directory. `BACKEND` defaults to `mlm` (BERT-style); other valid values
are `causal` (GPT-style), `mntp`, `enc_dec_mask`, `enc_dec_prefix`.

Under the hood this calls the upstream `collate_preds.sh` with
`TRACK=strict-small` and `--fast`, which runs both zero-shot and
fine-tuning evaluation. Scores land under
`third_party/babylm-eval/strict/results/` — see the upstream README in
that directory for the exact layout.

## Bumping the eval version

The submodule is pinned to a specific commit so everyone runs the same
eval. To upgrade:

```bash
cd third_party/babylm-eval
git fetch
git checkout <new-sha>
cd ../..
git add third_party/babylm-eval
git commit -m "Bump babylm-eval to <new-sha>"
```

Always pin a SHA, not a branch — that's what makes runs comparable
across the team.
