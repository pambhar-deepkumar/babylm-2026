# BabyLM25

This repo contains our code for reproducing our [BabyLM2025 submission](https://aclanthology.org/2025.babylm-main.31/).
You can find our trained models here: https://huggingface.co/collections/leukas/babylm-2025-68a74f07d8570c914c69e4be

## Training
To train our hard_decay model, for example, run:
```bash
python train_mask.py --train_data data/bb24.train --valid_data data/bb25_small.dev --tokenizer tokenizers/bb24.model --output_path models/test_model/ --mask_update_steps 200 --logging_steps 200 --intermediate_size 1280 --hidden_size 384 --max_seq_len 0:64,5:256 --lamb --all_checkpoints --mlm_prob 0.4 --mask_decay 0.25 --seed 0
```

All flags are explained here:
```
--train_data            Path to the training data file.
--valid_data            Path to the validation data file.
--max_seq_len           Maximum sequence length. Can be a single number (e.g. 64) or depending on epoch (e.g. 0:64,5:128).
--model_path            Model path. Defaults to microsoft/deberta-v3-base.
--output_path           Output directory for model checkpoints and logs.
--tokenizer             Tokenizer path or name. If not specified, uses the model’s default tokenizer.
--batch_size            Batch size. Default: 256
--grad_acc              Gradient accumulation steps. Default: 1
--lr                    Learning rate. Default: 0.007
--epochs                Number of training epochs. Default: 10
--cpus                  Number of CPU workers for data loading. Default: 64
--logging_steps         Log training metrics every N steps. Default: 100
--eval_steps            Run evaluation every N steps. Default: 1000
--save_steps            Save a checkpoint every N steps. Default: 1000
--all_checkpoints       Save and evaluate model at multiple checkpoints (1/10/100M words) according to BabyLM requirements. Overrides eval_steps and save_steps.
--log_mlm_probs         Log masked language model probabilities for analysis.
--mask_update_steps     Steps between dynamic mask updates. Default: 100
--first_mask_update     Do not perform a mask update before this global step. Default: 0
--hidden_size           Hidden size of the model. Default: 768
--intermediate_size     Intermediate size of the feedforward layers. Default: 3072
--dropout               Dropout probability. Default: 0.1
--weight_decay          Weight decay for optimizer. Default: 0.01
--mlm_prob              Probability of masking a token for MLM. Default: 0.15
--mask_replace_prob     Probability of replacing a masked token with [MASK]. Default: 0.8
--random_replace_prob   Probability of replacing a masked token with a random token. Default: 0.1
--seed                  Random seed. Default: 0
--pretrained            Load pretrained model weights.
--eval_only             Run evaluation only, without training.
--debug                 Activate debug mode.
--wandb                 Report metrics to Weights & Biases.
--regular_mlm           Use standard masked language modeling objective.
--custom                Path or identifier for a custom model.
--lamb                  Use LAMB optimizer instead of AdamW.
--lower                 Lowercase all input text.
--soft                  Use soft masking strategy.
--flops                 Compute FLOPs during training.
--mask_decay            Mask decay rate. For example, 0.1 decays masking probability linearly by 0.1 over training.
```
