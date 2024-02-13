import torch
import datasets
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from transformers import DebertaV2Tokenizer
from datasets import load_dataset, Dataset, DatasetDict

from transformers import (
    AutoConfig,
    BertForMaskedLM,
    AutoTokenizer,
    DataCollatorForWholeWordMask,
    Trainer,
    TrainingArguments,
    set_seed,
)

set_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

parser = ArgumentParser()
parser.add_argument('--train_norm_file', required=True, type=str)
parser.add_argument('--val_norm_file', required=True, type=str)
parser.add_argument('--cache_dir', default=None, type=str)
parser.add_argument('--save_dir', required=True, type=str)
args = parser.parse_args()

# add saved dir
for p in ['tokenizer_saved', 'tokens_saved', 'model_saved']:
    locals()[p+'_dir'] = Path(args.save_dir) / p

# Assembling a corpus
raw_datasets = load_dataset("text",
    data_files={"train": args.train_norm_file, "val": args.val_norm_file})
print(f"raw_datasets >>>>>>>>> \n {raw_datasets}")

def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 10000]["text"]
        for i in range(0, len(raw_datasets["train"]), 10000)
    )
training_corpus = get_training_corpus()

#  traing a new tokenizer
old_tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/deberta-v3-base",
    cache_dir = args.cache_dir)


tokenizer = old_tokenizer.train_new_from_iterator(
                                text_iterator = training_corpus,
                                vocab_size = 128_100)
print(f"The length of old tokenizer: {len(old_tokenizer)}")
print(f"The length of new tokenizer: {len(tokenizer)}")

# saving tokenizer
tokenizer.save_pretrained(tokenizer_saved_dir)
print("The pretrained new tokenizer has been saved!")


# text tokenize
tokenizer = AutoTokenizer.from_pretrained(tokenizer_saved_dir)

max_seq_length = 512
start_tok = tokenizer.convert_tokens_to_ids('[CLS]')
sep_tok = tokenizer.convert_tokens_to_ids('[SEP]')
pad_tok = tokenizer.convert_tokens_to_ids('[PAD]')

def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        padding=True,
        max_length=512,
        return_overflowing_tokens=True,
        return_length=True,
        return_tensors='pt'
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == 512:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

def full_sent_tokenize(file_name):

    with open(file_name, 'r', encoding='utf-8') as f:
        sents = f.read().strip().split('\n')

    tok_sents = [tokenizer(s, padding=False, truncation=False)['input_ids'] for s in tqdm(sents)]
    for s in tok_sents:
        s.pop(0)
    
    res = [[]]
    l_curr = 0
    
    for s in tok_sents:
        l_s = len(s)
        idx = 0
        while idx < l_s - 1:
            if l_curr == 0:
                res[-1].append(start_tok)
                l_curr = 1
            s_end = min(l_s, idx + max_seq_length - l_curr) - 1
            res[-1].extend(s[idx:s_end] + [sep_tok])
            idx = s_end
            if len(res[-1]) == max_seq_length:
                res.append([])
            l_curr = len(res[-1])
    
    for s in res[:-1]:
        assert s[0] == start_tok and s[-1] == sep_tok
        assert len(s) == max_seq_length
        
    attention_mask = []
    for s in res:
        attention_mask.append([1] * len(s) + [0] * (max_seq_length - len(s)))
    
    return {'input_ids': res, 'attention_mask': attention_mask}


df_train = pd.DataFrame(full_sent_tokenize(args.train_norm_file))
df_val = pd.DataFrame(full_sent_tokenize(args.val_norm_file))
tokenized_datasets = DatasetDict({
    'train': Dataset.from_pandas(df_train),
    'val': Dataset.from_pandas(df_val)
})

print(f"tokenized_datasets >>>>>>>>> \n {tokenized_datasets}")

# save dataset tokens
tokenized_datasets.save_to_disk(tokens_saved_dir)
print(tokens_saved_dir)