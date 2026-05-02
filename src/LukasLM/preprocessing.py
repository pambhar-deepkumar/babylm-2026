# preprocessing.py
import torch
from transformers.data.data_collator import DataCollatorForLanguageModeling

# def group_texts(examples, max_len, pad_token_id):
#     grouped = {
#         'input_ids': [],
#         'attention_mask': [],
#         'labels': []
#     }

#     for i in range(len(examples['input_ids'])):
#         for j in range(0, len(examples['input_ids'][i]), max_len):
#             if j + max_len > len(examples['input_ids'][i]):
#                 grouped['input_ids'].append(examples['input_ids'][i][j:] + [pad_token_id] * (max_len - (len(examples['input_ids'][i]) - j)))
#                 grouped['attention_mask'].append(examples['attention_mask'][i][j:] + [0] * (max_len - (len(examples['input_ids'][i]) - j)))
#                 grouped['labels'].append(examples['labels'][i][j:] + [pad_token_id] * (max_len - (len(examples['input_ids'][i]) - j)))
#             else:
#                 grouped['input_ids'].append(examples['input_ids'][i][j : j + max_len])
#                 grouped['attention_mask'].append(examples['attention_mask'][i][j : j + max_len])
#                 grouped['labels'].append(examples['labels'][i][j : j + max_len])

#     return grouped

def group_texts(examples, max_len):
    # Concatenate all texts.
    try:
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    except TypeError:
        print(examples)
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_len:
        total_length = (total_length // max_len) * max_len
    # Split by chunks of max_len.

    result = {
        k: [t[i : i + max_len] for i in range(0, total_length, max_len)]
        for k, t in concatenated_examples.items()
    }

    return result



def group_pack(examples, max_length, pad_token_id):
    grouped = {key: [] for key in examples}

    key0 = list(examples.keys())[0]
    example = {key: [] for key in examples}
    for i in range(len(examples[key0])):
        if len(examples[key0][i]) + len(example[key0]) > max_length:
            for key in example:
                grouped[key].append(example[key] + [pad_token_id] * (max_length - len(example[key])))
            example = {key: [] for key in examples}
            
        for key in example:
            example[key].append(examples[key][i])

    for key in example:
        grouped[key].append(example[key])

    return grouped

def tokenize(examples, tokenizer, input_field):
    encoded = {
        'input_ids': [], 
        'attention_mask': [], 
        'labels': []}
    for i in range(len(examples[input_field])):
        src = examples[input_field][i]
        src_tok = tokenizer.encode(src)
        # encoded['input_ids'] += [src_tok[:-1]]
        # encoded['labels'] += [src_tok[1:]]
        encoded['input_ids'] += [src_tok]
        encoded['labels'] += [src_tok]
        attention_mask = [1] * len(src_tok)#torch.ones(len(src_tok), dtype=torch.long).tolist()
        encoded['attention_mask'] += [attention_mask]
    return encoded

def padding_collate_fn(batch, max_len=2048, skip_fields=[], add_labels=False):
    """ 
        Pads each list with zeros and concatenates by key.
        Input: List[{key: List[], ...}]
        Output: {key: LongTensor(), ...}
    """
    padded_batch = {}

    if add_labels:
        for b in range(len(batch)):
            batch[b]['labels'] = batch[b]['input_ids']

    for key in batch[0]:
        if key in skip_fields:
            padded_batch[key] = []
            continue
        largest = min(max_len, max([len(b[key]) for b in batch]))
        padded_batch[key] = torch.zeros((len(batch), largest), dtype=torch.long)
        if "labels" in key:
            padded_batch[key] -= 100
    
    for i, sample in enumerate(batch):
        for key in padded_batch:
            if key in skip_fields: 
                padded_batch[key].append(batch[i][key]) 
                continue
            key_len = min(max_len, len(sample[key]))
            padded_batch[key][i, -key_len:] = torch.LongTensor(sample[key][:key_len])

    return padded_batch