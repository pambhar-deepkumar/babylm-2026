"""
Wrapper entry point for evaluating DeBERTa masked-LM checkpoints with lm_eval.

Importing deberta_mlm_model triggers the @register_model("deberta-mlm") decorator,
making the model type visible to lm_eval before it tries to resolve --model deberta-mlm.

Usage (same flags as python -m lm_eval):
    python run_deberta_eval.py \
        --model deberta-mlm \
        --model_args "pretrained=/path/to/checkpoint,batch_size=8,max_length=256" \
        --tasks zeroshot_eng \
        --device cuda \
        --output_path results/nour/my_run \
        --batch_size 1 \
        --num_fewshot 0 \
        --log_samples \
        --include_path tasks/
"""
import os
import sys

# Ensure deberta_mlm_model.py (in the same directory as this script) is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deberta_mlm_model  # noqa: F401 — side-effect: registers "deberta-mlm" model type

from lm_eval.__main__ import cli_evaluate

cli_evaluate()
