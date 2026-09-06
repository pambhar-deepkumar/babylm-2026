"""
lm_eval model backend for DeBERTa-v2 (encoder-only masked LM).

Scoring method: pseudo-log-likelihood (PLL, Salazar et al. 2020).
For each token position i in the continuation, replace token_i with [MASK],
run one MLM forward pass, take log P(token_i | all other tokens).
Sum across all continuation positions = PLL score for that continuation.

For efficiency, all masked copies of a single sentence are stacked into one
batched forward pass (one pass per request, not one pass per token).

Usage with lm_eval:
    python -m lm_eval \
        --model deberta-mlm \
        --model_args pretrained=/path/to/checkpoint,batch_size=8 \
        --tasks zeroshot_eng \
        --device cuda \
        --include_path tasks/

Two-stage tokenization (for meta-tokenizer checkpoints, e.g. multi_run2/3):
    --model_args pretrained=/path/to/checkpoint,preseg_model=/path/to/bpe.model
  The preseg_model is a standard SentencePiece BPE model that first segments
  raw text into pieces (with ▁ word-boundary markers); those pieces then go
  through the checkpoint's word-type meta-tokenizer to produce token IDs.
  This replicates the two-stage tokenization used during training.
"""
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("deberta-mlm")
class DebertaMLM(LM):
    def __init__(
        self,
        pretrained: str,
        batch_size: int = 1,
        device: str = "cuda",
        max_length: int = 512,
        preseg_model: str = None,
        **kwargs,
    ):
        super().__init__()
        pretrained = os.path.expanduser(pretrained)
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._max_length = int(max_length)

        # use_fast=False: word-type SentencePiece meta-tokenizers (used by
        # multi_run2/3) cannot be converted by the Rust fast tokenizer backend,
        # which requires BPE or Unigram. The slow Python tokenizer handles all
        # SentencePiece model types uniformly.
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            pretrained, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        self.model.to(self._device)
        self.model.eval()

        self._batch_size = int(batch_size) if batch_size is not None else 1

        self.mask_token_id = self.tokenizer.mask_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        assert self.mask_token_id is not None, "Tokenizer must have a [MASK] token"

        # Two-stage tokenization for meta-tokenizer checkpoints (run2/run3).
        # preseg_model is a per-language BPE tokenizer that produces piece strings;
        # those pieces (with ▁ replaced by <SP> to avoid SentencePiece collision)
        # are then encoded by the checkpoint's meta-tokenizer.
        if preseg_model is not None:
            import sentencepiece as spm_lib
            self._preseg_sp = spm_lib.SentencePieceProcessor()
            self._preseg_sp.load(os.path.expanduser(preseg_model))
        else:
            self._preseg_sp = None

    # ------------------------------------------------------------------ #
    # Properties required by the LM interface                             #
    # ------------------------------------------------------------------ #

    @property
    def eot_token_id(self):
        return self.sep_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self._encode_text(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _encode_text(self, text: str) -> list:
        """
        Encode raw text to token IDs. If a preseg_model is loaded, first
        segment with the BPE tokenizer and apply the ▁→<SP> substitution
        to match training preprocessing, then encode with the meta-tokenizer.
        """
        if self._preseg_sp is not None:
            pieces = self._preseg_sp.encode(text, out_type=str)
            # ▁ (U+2581) inside BPE pieces was replaced with <SP> at training
            # time to avoid collision with SentencePiece's own boundary marker.
            pieces = [p.replace('▁', '<SP>') for p in pieces]
            text = ' '.join(pieces)
        return self.tokenizer.encode(text, add_special_tokens=False)

    # ------------------------------------------------------------------ #
    # PLL scoring                                                          #
    # ------------------------------------------------------------------ #

    def _pll_score(self, input_ids_1d: torch.Tensor, cont_start: int):
        """
        Score the continuation tokens of a single sequence using PLL.

        Creates one masked copy per continuation token (all stacked into a
        single batched forward pass), reads the log-prob of the original token
        at each masked position.

        Returns (total_logprob: float, is_greedy: bool).
        """
        seq_len = input_ids_1d.size(0)

        # Continuation ends just before the final [SEP] (if present)
        if self.sep_token_id is not None and input_ids_1d[-1].item() == self.sep_token_id:
            cont_end = seq_len - 1
        else:
            cont_end = seq_len

        n_cont = cont_end - cont_start
        if n_cont <= 0:
            return 0.0, True

        # Stack n_cont copies, each with one position masked
        base = input_ids_1d.unsqueeze(0).expand(n_cont, -1).clone()  # [n_cont, seq_len]
        for i, pos in enumerate(range(cont_start, cont_end)):
            base[i, pos] = self.mask_token_id
        base = base.to(self._device)

        # Process in sub-batches to avoid OOM on very long continuations
        all_logits = []
        for start in range(0, n_cont, self._batch_size):
            chunk = base[start : start + self._batch_size]
            with torch.no_grad():
                out = self.model(input_ids=chunk)
            all_logits.append(out.logits.cpu())  # keep on CPU between chunks
        logits = torch.cat(all_logits, dim=0)  # [n_cont, seq_len, vocab_size]

        total_logprob = 0.0
        all_correct = True
        for i, pos in enumerate(range(cont_start, cont_end)):
            orig = input_ids_1d[pos].item()
            lp = F.log_softmax(logits[i, pos].float(), dim=-1)
            total_logprob += lp[orig].item()
            if logits[i, pos].argmax().item() != orig:
                all_correct = False

        return total_logprob, all_correct

    def _build_input(self, context: str, continuation: str):
        """
        Tokenize context + continuation into a single [CLS] ctx cont [SEP]
        sequence. Returns (input_ids_1d, cont_start).
        Truncates context from the left if the sequence is too long.
        """
        ctx_ids = self._encode_text(context)
        cont_ids = self._encode_text(continuation)

        prefix = [self.cls_token_id] if self.cls_token_id is not None else []
        suffix = [self.sep_token_id] if self.sep_token_id is not None else []

        # How much space do we have for context?
        overhead = len(prefix) + len(cont_ids) + len(suffix)
        max_ctx = self._max_length - overhead
        if max_ctx < 0:
            # Continuation itself is too long — truncate it
            cont_ids = cont_ids[:self._max_length - len(prefix) - len(suffix)]
            ctx_ids = []
        elif len(ctx_ids) > max_ctx:
            ctx_ids = ctx_ids[-max_ctx:]  # keep the most recent context

        full_ids = prefix + ctx_ids + cont_ids + suffix
        cont_start = len(prefix) + len(ctx_ids)

        return torch.tensor(full_ids, dtype=torch.long), cont_start

    # ------------------------------------------------------------------ #
    # lm_eval interface                                                    #
    # ------------------------------------------------------------------ #

    def loglikelihood(self, requests):
        results = []
        for req in tqdm(requests, desc="PLL loglikelihood"):
            context, continuation = req.args
            input_ids, cont_start = self._build_input(context, continuation)
            score, is_greedy = self._pll_score(input_ids, cont_start)
            results.append((score, is_greedy))
        return results

    def loglikelihood_rolling(self, requests):
        """Score full texts with no context split — used by perplexity tasks."""
        results = []
        for req in tqdm(requests, desc="PLL rolling"):
            (text,) = req.args
            ids = self._encode_text(text)
            # Add special tokens manually
            if self.cls_token_id is not None:
                ids = [self.cls_token_id] + ids
            if self.sep_token_id is not None:
                ids = ids + [self.sep_token_id]
            ids = ids[:self._max_length]
            input_ids = torch.tensor(ids, dtype=torch.long)
            # Score everything after [CLS]
            start = 1 if (self.cls_token_id is not None and ids[0] == self.cls_token_id) else 0
            score, _ = self._pll_score(input_ids, start)
            results.append(score)
        return results

    def generate_until(self, requests):
        raise NotImplementedError(
            "DebertaMLM is an encoder-only masked LM and cannot generate text autoregressively. "
            "Only zero-shot scoring tasks (loglikelihood) are supported."
        )

