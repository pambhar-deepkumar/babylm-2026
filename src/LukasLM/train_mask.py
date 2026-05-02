# train_mask.py
import argparse
import os
import math
import torch
import numpy as np
from tqdm import tqdm
from transformers import set_seed
from transformers import AutoConfig, AutoModelForMaskedLM, DebertaV2Tokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
from datasets import load_dataset, load_from_disk

from preprocessing import tokenize, padding_collate_fn, group_texts

from bitsandbytes.optim import LAMB

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type=str, default="")
parser.add_argument("--valid_data", type=str, default="data/even.dev")
parser.add_argument("--max_seq_len", type=str, default="64", help="Either num or e.g. 0:64,5:128")
parser.add_argument("--model_path", type=str, default="microsoft/deberta-v3-base")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--grad_acc", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.007)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--cpus", type=int, default=64)
parser.add_argument("--logging_steps", type=int, default=100)
parser.add_argument("--eval_steps", type=int, default=1000)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--all_checkpoints", action="store_true", help="Save and evaluate model every 1/10/100M tokens, \
                    per challenge stipulations. Overrides eval_steps and save_steps.")
parser.add_argument("--log_mlm_probs", action="store_true", help="Log MLM probabilities for analysis")
parser.add_argument("--mask_update_steps", type=int, default=100)
parser.add_argument("--first_mask_update", type=int, default=0, help="Do not update mask before this global step")
parser.add_argument("--hidden_size", type=int, default=768)
parser.add_argument("--intermediate_size", type=int, default=3072)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--mlm_prob", type=float, default=0.15)
parser.add_argument("--mask_replace_prob", type=float, default=0.8)
parser.add_argument("--random_replace_prob", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pretrained", action="store_true", help="Load pretrained model")
parser.add_argument("--eval_only", action="store_true", help="Evaluate only")
parser.add_argument("--debug", action="store_true", help="Activates debug mode")
parser.add_argument("--wandb", action="store_true", help="Report to wandb")
parser.add_argument("--regular_mlm", action="store_true", help="Regular MLM")
parser.add_argument("--custom", type=str, default="", help="Use custom model")
parser.add_argument("--lamb", action="store_true", help="LAMB optimization")
parser.add_argument("--lower", action="store_true", help="Lowercase")
parser.add_argument("--soft", action="store_true", help="Soft mask")
parser.add_argument("--flops", action="store_true", help="Compute FLOPs")
parser.add_argument("--mask_decay", type=float, default=0.0, help="Mask decay. e.g. 0.1 means decay by 0.1 \
                    over the course of training")

def evaluate(model, tokenizer, dataloader, args):
    model.eval()
    correct = 0
    total = 0
    avg_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if len(batch["input_ids"]) == 0:
                continue # NOTE: not sure why this happens in 100M case...

            masked_batch = mask_batch(batch, tokenizer, mask_weights=None, mlm_prob=0.15, mask_replace_prob=0.8, random_replace_prob=0.1)
            # split batch into grad_acc chunks
            batches = split_batch(masked_batch, args)
            for minibatch in batches:
                with torch.autocast(dtype=torch.bfloat16, device_type="cuda:0"):
                    outputs = model(**move_dict_to_cuda(minibatch))
            
                avg_loss += outputs.loss.item()
                logits = outputs.logits
                preds = logits.argmax(dim=-1)

                labels = minibatch["labels"].to(device=logits.device, dtype=logits.dtype)
                label_mask = labels != -100
                correct += (preds[label_mask] == labels[label_mask]).sum().item()
                total += preds[label_mask].numel()

    model.train()
    return {'acc': 100 * correct / total, 'loss': avg_loss / (len(dataloader) * args.grad_acc)}


def regroup_texts(args, max_seq_len):
    cur_max_seq_len = args.cur_max_seq_len
    print("CUR MAX SEQ LEN:", cur_max_seq_len)

    grouped_dataset = args.dataset.map(group_texts,
        batched = True,
        fn_kwargs = {'max_len': max_seq_len},
        # remove_columns = dataset["train"].column_names,
        num_proc=args.cpus,
        desc = "Grouping",
        # load_from_cache_file=False,
        )
    change_ratio = max_seq_len / cur_max_seq_len
    args.batch_size = max(1, int(args.batch_size / change_ratio))
    
    train_dataloader = torch.utils.data.DataLoader(
        grouped_dataset['train'], 
        batch_size=args.batch_size, 
        num_workers=args.cpus,
        shuffle=True, 
        collate_fn=padding_collate_fn
        )
    
    eval_dataloader = torch.utils.data.DataLoader(
        grouped_dataset['validation'], 
        batch_size=args.batch_size, 
        num_workers=args.cpus,
        shuffle=False, 
        collate_fn=padding_collate_fn
        )

    args.cur_max_seq_len = max_seq_len

    return train_dataloader, eval_dataloader

def mask_batch(batch, tokenizer, mask_weights=None, mlm_prob=0.15, mask_replace_prob=0.8, random_replace_prob=0.1):
    if mask_weights is None:
        mask_weights = torch.full((tokenizer.vocab_size,), mlm_prob, device=batch["input_ids"].device)
    masked_batch = batch.copy()
    for i in range(len(batch["input_ids"])):
        enc = batch["input_ids"][i]

        enc_mask_weights = mask_weights[enc] # weights for each token
        enc_mask_weights[enc == tokenizer.pad_token_id] = 0 # dont mask pad tokens

        enc_mask_weights = mlm_prob * len(enc) * enc_mask_weights / enc_mask_weights.sum() # normalize to avg .15

        enc_mask_mask = torch.rand(len(enc)) < enc_mask_weights # tokens to mask/replace
        rand = torch.rand(len(enc))
        to_mask = rand < mask_replace_prob # tokens to replace with mask
        to_replace = (rand > mask_replace_prob) & (rand < mask_replace_prob + random_replace_prob) # tokens to replace randomly

        to_mask = to_mask & enc_mask_mask
        to_replace = to_replace & enc_mask_mask

        randoms = torch.randint(0, tokenizer.vocab_size, (len(enc),))
        enc[to_mask] = tokenizer.mask_token_id
        enc[to_replace] = randoms[to_replace]

        masked_batch["input_ids"][i] = enc

        labels = batch["labels"][i]
        labels[~enc_mask_mask] = -100
        masked_batch["labels"][i] = labels

    return masked_batch


def get_batch_accuracy(logits, labels, prev_stats):
    vocab_size = logits.shape[-1]

    # Mask out ignored labels
    mask = labels != -100
    labels = labels[mask]
    preds = logits.argmax(dim=-1)[mask]

    # Get match mask
    correct_mask = preds == labels

    # Use bincount on correct and incorrect
    correct_counts = torch.bincount(labels[correct_mask], minlength=vocab_size)
    incorrect_counts = torch.bincount(labels[~correct_mask], minlength=vocab_size)

    prev_stats['correct'] += correct_counts
    prev_stats['incorrect'] += incorrect_counts

    return prev_stats

def update_mask_weights(mask_weights, mask_stats, mlm_prob=0.15):
    correct_pct = (mask_stats['correct'] + 0.5) / (mask_stats['incorrect'] + mask_stats['correct'] + 1)
    # if there are no instances, treated as 50% accuracy
    # things that never show up will go to 7.5% mlm_prob (seems fine?)

    new_weights = mlm_prob - (correct_pct * mlm_prob)
    mask_weights = 0.2 * mask_weights + 0.8 * new_weights

    mask_weights = mask_weights.clamp(0.005)
    mask_weights = mlm_prob * mask_weights.shape[0] * mask_weights / mask_weights.sum() # normalize back to mlm_prob

    return mask_weights

def get_batch_accuracy_soft(masked_inputs, logits, labels, prev_stats):
    with torch.no_grad():
        vocab_size = logits.shape[-1]

        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
        loss = loss.view(masked_inputs.shape)

        # Mask out ignored labels and replaced tokens
        mask = masked_inputs == 4 # mask_token_id

        loss = loss[mask]
        labels = labels[mask]

        prev_stats['loss'].scatter_add_(0, labels, loss)
        prev_stats['total'].scatter_add_(0, labels, torch.ones_like(loss))

        return prev_stats
    
def update_mask_weights_soft(mask_weights, mask_stats, mlm_prob=0.15):
    max_loss = torch.log(torch.tensor(mask_weights.shape[0])).item() # e.g. log(40000) = 10.5966, uniform guess
    avgs = mask_stats['loss'] / mask_stats['total']
    norm_loss = avgs / max_loss

    # min-max normalize and invert
    norm_loss = 1 - (norm_loss - norm_loss.min()) / (norm_loss.max() - norm_loss.min())

    new_weights = mlm_prob - (norm_loss * mlm_prob)
    mask_weights = 0.2 * mask_weights + 0.8 * new_weights

    mask_weights = mask_weights.clamp(0.005)
    mask_weights = mlm_prob * mask_weights.shape[0] * mask_weights / mask_weights.sum() # normalize back to mlm_prob

    return mask_weights

def split_batch(batch, args):
    minibatch_size = args.batch_size // args.grad_acc
    if len(batch["input_ids"]) == minibatch_size:
        return [batch]

    batches = []
    for i in range(0, len(batch["input_ids"]), minibatch_size):
        minibatch = {}
        for key in batch.keys():
            minibatch[key] = batch[key][i:i+minibatch_size]
        batches.append(minibatch)
    return batches

def move_dict_to_cuda(d):
    return {key: value.to(device="cuda:0") for key, value in d.items()}

def move_dict_to_cuda_bf16(d):
    return {key: value.to(dtype=torch.bfloat16, device="cuda:0") for key, value in d.items()}


def reset_stats(mask_stats):
    mask_stats = {
        'correct': torch.zeros_like(mask_stats['correct']), 
        'incorrect': torch.zeros_like(mask_stats['incorrect']),
        'loss': torch.zeros_like(mask_stats['loss']) + torch.log(torch.tensor(mask_stats['loss'].shape[0])).item(),
        'total': torch.ones_like(mask_stats['total'])
        }
    return mask_stats

def calculate_num_words(examples):
    return {'num_words': [sum(len(examples["text"][i]) for i in range(len(examples["text"])))]}

def calculate_num_tokens(examples):
    return {'num_tokens': [sum(len(examples["input_ids"][i]) for i in range(len(examples["input_ids"])))]}

def calculate_total_steps(args):
    def calc_exs_per_epoch(tokens_per_1000, max_seq_len):
        return sum([t // max_seq_len for t in tokens_per_1000])

    
    exs_per_epoch = calc_exs_per_epoch(args.tokens_per_1000, args.init_max_seq_len)
    batches_per_epoch = math.ceil(exs_per_epoch / args.batch_size)
    cur_epoch = 0
    if len(args.max_seq_len) == 0:
        return batches_per_epoch * args.epochs
    else:
        total_steps = 0
        batch_size = args.batch_size
        prev_seq_len = args.init_max_seq_len
        for epoch_num, seq_len in args.max_seq_len:
            total_steps = total_steps + batches_per_epoch * (epoch_num - cur_epoch)

            len_ratio = prev_seq_len / seq_len
            batch_size = int(batch_size * len_ratio)

            exs_per_epoch = calc_exs_per_epoch(args.tokens_per_1000, seq_len)
            batches_per_epoch = math.ceil(exs_per_epoch / batch_size)
            cur_epoch = epoch_num
            prev_seq_len = seq_len
    
        total_steps = total_steps + batches_per_epoch * (args.epochs - cur_epoch)
        return total_steps


def is_step(step_type: str, global_step: int, args):
    # step_arg = args.logging_steps, args.save_steps, or args.eval_steps
    step_arg = getattr(args, f'{step_type}_steps')

    if args.all_checkpoints:
        if global_step in args.checkpoints:
            return True
    else:
        if global_step % step_arg == 0 and global_step != 0:
            return True
        
    return False



def train(args, model, tokenizer, train_dataloader, eval_dataloader):
    if args.flops:
        from fvcore.nn import FlopCountAnalysis


    if args.lamb:
        optimizer = LAMB(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.1)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-08, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.total_steps//100, num_training_steps=args.total_steps)

    model.train()
    model = model.to(dtype=torch.bfloat16, device="cuda:0")


    mask_weights = torch.full((tokenizer.vocab_size,), args.mlm_prob).to(device="cuda:0")
    mask_stats = {
        'correct': torch.zeros(tokenizer.vocab_size, dtype=torch.float32), 
        'incorrect': torch.zeros(tokenizer.vocab_size, dtype=torch.float32),
        'loss': torch.zeros(tokenizer.vocab_size, dtype=torch.float32) + torch.log(torch.tensor(mask_weights.shape[0])).item(),
        'total': torch.ones(tokenizer.vocab_size, dtype=torch.float32)
        }
    mask_stats = move_dict_to_cuda_bf16(mask_stats)
    if args.log_mlm_probs:
        # create parent folder
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        log_file = open(os.path.join(args.output_path, "mlm_probs.csv"), "w")
        # get all vocab as list
        vocab_list = list(tokenizer.get_vocab().keys())
        header = ",".join(vocab_list)
        log_file.write(f"step,{header}\n")

    global_step = 0
    print(f"Total steps: {args.total_steps}")
    print(f"Max seq len: {args.init_max_seq_len}")
    print(f"Next seq len: {args.max_seq_len}")
    with tqdm(total=args.total_steps) as pbar:
        for epoch in range(args.epochs):
            if len(args.max_seq_len) > 0:
                if epoch >= args.max_seq_len[0][0]:
                    train_dataloader, eval_dataloader = regroup_texts(args, args.max_seq_len[0][1])
                    args.max_seq_len = args.max_seq_len[1:]

            for step, batch in enumerate(train_dataloader):
                masked_batch = mask_batch(batch, 
                                          tokenizer, mask_weights.to(device="cpu"), 
                                          mlm_prob=args.mlm_prob, 
                                          mask_replace_prob=args.mask_replace_prob, 
                                          random_replace_prob=args.random_replace_prob
                                          )            

                # split batch into grad_acc chunks
                batches = split_batch(masked_batch, args)
                
                for minibatch in batches:

                    with torch.autocast(dtype=torch.bfloat16, device_type="cuda:0"):
                        if args.flops:
                            model.eval()
                            flops = FlopCountAnalysis(model, tuple(move_dict_to_cuda(minibatch).values()))
                            flops = flops.by_operator()
                            total = 0
                            for key in flops:
                                total += flops[key]
                            print(f"Estimated Total FLOPs: {total*3*args.total_steps}")
                            exit()


                        outputs = model(**move_dict_to_cuda(minibatch))
                        loss = outputs.loss

                    # Update stats per-minibatch and discard logits
                    if not args.regular_mlm:
                        with torch.no_grad():
                            if args.soft:
                                mask_stats = get_batch_accuracy_soft(
                                    minibatch["input_ids"].to(device="cuda:0"),
                                    outputs.logits.detach(),
                                    minibatch["labels"].to(device="cuda:0"),
                                    mask_stats,
                                )
                            else:
                                mask_stats = get_batch_accuracy(
                                    outputs.logits.detach(),
                                    minibatch["labels"].to(device="cuda:0"),
                                    mask_stats,
                                )

                    loss = loss / args.grad_acc # To ensure consistent gradient magnitude
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # stats already updated per-minibatch; nothing to do here

                if (
                    global_step % args.mask_update_steps == 0
                    and global_step != 0
                    and global_step >= args.first_mask_update
                    and not args.regular_mlm
                ):
                    if args.soft:
                        mask_weights = update_mask_weights_soft(mask_weights, mask_stats, args.mlm_prob)
                    else:
                        mask_weights = update_mask_weights(mask_weights, mask_stats, args.mlm_prob)

                    if args.log_mlm_probs:
                        # convert probs to .4f string
                        probs_str = [f"{prob.item():.4f}" for prob in mask_weights]
                        probs_str = ",".join(probs_str)
                        log_file.write(f"{global_step},{probs_str}\n")
                        log_file.flush()
                        

                    # print top 5 tokens and their probs
                    top_tokens = torch.topk(mask_weights, 5)
                    print("Top tokens: ", flush=True, end="")
                    print_str = []
                    for token, prob in zip(tokenizer.convert_ids_to_tokens(top_tokens.indices), top_tokens.values):
                        print_str.append(f"{token} ({prob.item():.2f})")

                    print("     ".join(print_str), flush=True)

                    # print bottom 5 tokens and their probs
                    bottom_tokens = torch.topk(-mask_weights, 5)
                    print("Bottom tokens: ", flush=True, end="")
                    print_str = []
                    for token, prob in zip(tokenizer.convert_ids_to_tokens(bottom_tokens.indices), -bottom_tokens.values):
                        print_str.append(f"{token} ({prob.item():.2f})")

                    print("     ".join(print_str), flush=True)

                    mask_stats = reset_stats(mask_stats)

                # ----- LOGGING -----
                if is_step("logging", global_step, args):
                    epoch_float = global_step * args.epochs / args.total_steps
                    print(f"Epoch {epoch_float:.2f}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)

                    if args.wandb:
                        wandb.log({
                            "epoch": epoch_float,
                            "loss": loss.item(),
                            "lr": scheduler.get_last_lr()[0]
                        })

                # ----- EVALUATION -----
                if is_step("eval", global_step, args):
                    metrics = evaluate(model, tokenizer, eval_dataloader, args)
                    print(f"----- Eval accuracy: {metrics['acc']:.2f}, Loss: {metrics['loss']:.4f} -----", flush=True)

                    if args.wandb:
                        wandb.log({
                            "eval_acc": metrics["acc"],
                            "eval_loss": metrics["loss"]
                        })

                # ----- SAVING -----
                if is_step("save", global_step, args):
                    save_path = os.path.join(args.output_path, f"checkpoint-{global_step}")
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"----- Saved checkpoint to: {save_path} -----", flush=True)

                pbar.update(1)
                global_step += 1
                if args.mask_decay > 0:
                    args.mlm_prob = args.mlm_prob - (args.mask_decay/args.total_steps)

    metrics = evaluate(model, tokenizer, eval_dataloader, args)
    print(f"Final eval accuracy: {metrics['acc']:.2f}, Loss: {metrics['loss']:.4f}", flush=True)

    save_path = os.path.join(args.output_path, f"checkpoint-{args.total_steps}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    if args.wandb:
        wandb.finish()

def parse_max_seq_len(max_seq_len):
    if "," in max_seq_len:
        max_seq_len = max_seq_len.split(",")
        assert ":" in max_seq_len[0]
        return [(int(val.split(":")[0]), int(val.split(":")[1])) for val in max_seq_len]
    elif ":" in max_seq_len:
        max_seq_len = max_seq_len.split(":")[1]
    return [(0, int(max_seq_len))]

def main():
    args = parser.parse_args()
    args.max_seq_len = parse_max_seq_len(args.max_seq_len)
    set_seed(args.seed)

    if args.wandb:
        assert wandb_available
        output_dir = os.path.basename(os.path.normpath(args.output_path))
        wandb.init(
            project='babylm25',
            name=output_dir,
            config=vars(args),   
        )

    tokenizer = DebertaV2Tokenizer(args.tokenizer, do_lower_case=args.lower)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    config.vocab_size = tokenizer.vocab_size
    config.max_position_embeddings = 1024
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.cls_token_id
    config.cls_token_id = tokenizer.cls_token_id
    config.eos_token_id = tokenizer.sep_token_id
    config.sep_token_id = tokenizer.sep_token_id


    config.hidden_size = args.hidden_size
    config.intermediate_size = args.intermediate_size
    config.dropout = args.dropout
    config.hidden_dropout_prob = args.dropout

    if args.custom == "nhot":
        from ngram_model import NGramDebertaV2ForMaskedLM
        model = NGramDebertaV2ForMaskedLM(config, tokenizer)
    elif args.custom == "lora":
        from ngram_model import LoraModelForMaskedLM
        model = LoraModelForMaskedLM(config)
    else:
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)

    if args.pretrained:
        model = model.from_pretrained(args.model_path)


    # print model num parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {num_params}")

    # for p in model.named_parameters():
    #     print(p[0], p[1].shape)

    if args.custom == "pos":
        dataset = load_from_disk(args.train_data)
        dataset['validation'] = load_dataset('text', data_files = args.valid_data)['train']
    else:
        dataset = load_dataset('text', data_files = {'train': args.train_data, 'validation': args.valid_data})

    if args.debug:
        dataset['train'] = dataset['train'].select(range(100))
        dataset['validation'] = dataset['validation'].select(range(100))

    dataset = dataset.map(tokenize, 
        batched = True, 
        fn_kwargs = {'tokenizer': tokenizer, 'input_field': 'text'}, 
        remove_columns = dataset["train"].column_names, 
        num_proc=args.cpus,
        desc = "Tokenizing",
        # load_from_cache_file=False,
        )
        
    args.dataset = dataset

    print(dataset)

    max_seq_len = args.max_seq_len.pop(0)[1]
    args.init_max_seq_len = max_seq_len
    args.cur_max_seq_len = max_seq_len
    args.dataset_len_lines = len(dataset['train'])
    args.tokens_per_1000 = dataset['train'].map(calculate_num_tokens, 
                                                       batched = True,
                                                       num_proc=args.cpus, 
                                                       remove_columns=dataset["train"].column_names)['num_tokens']

    args.dataset_len_tokens = sum(args.tokens_per_1000)
    args.total_steps = calculate_total_steps(args)

    args.is_strict_small = (args.dataset_len_tokens // 10e6) < 10 # assuming there are fewer than 10 tokens per word

    if args.is_strict_small:
        steps_1m = np.round(np.linspace(args.total_steps//100, args.total_steps//10, 10)).astype(int)
        steps_10m = np.round(np.linspace(args.total_steps//10, args.total_steps, 10)).astype(int)
        args.checkpoints = list(steps_1m) + list(steps_10m)[1:]
    else: # strict
        steps_1m = np.linspace(args.total_steps//1000, args.total_steps//100, 10).astype(int)
        steps_10m = np.linspace(args.total_steps//100, args.total_steps//10, 10).astype(int)
        steps_100m = np.linspace(args.total_steps//10, args.total_steps, 10).astype(int)
        args.checkpoints = list(steps_1m) + list(steps_10m)[1:] + list(steps_100m)[1:]


    print(f"Dataset length: {args.dataset_len_lines} lines, {args.dataset_len_tokens} tokens", flush=True)
    print(f"Total steps: {args.total_steps}", flush=True)
    print(f"Save checkpoints: {args.checkpoints}", flush=True)


    grouped_dataset = dataset.map(group_texts,
        batched = True,
        fn_kwargs = {'max_len': max_seq_len},
        # remove_columns = dataset["train"].column_names,
        num_proc=args.cpus,
        desc = "Grouping",
        # load_from_cache_file=False,
        )

    print("Dataset length after grouping:", len(grouped_dataset['train']))

    train_dataloader = torch.utils.data.DataLoader(
        grouped_dataset['train'], 
        batch_size=args.batch_size, 
        num_workers=args.cpus,
        shuffle=True, 
        collate_fn=padding_collate_fn
        )
    
    eval_dataloader = torch.utils.data.DataLoader(
        grouped_dataset['validation'], 
        batch_size=args.batch_size, 
        num_workers=args.cpus,
        shuffle=False, 
        collate_fn=padding_collate_fn
        )

    train(args, model, tokenizer, train_dataloader, eval_dataloader)

if __name__ == "__main__":
    main()
