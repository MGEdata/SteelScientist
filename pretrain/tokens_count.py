from datasets import load_dataset, Dataset
from transformers import DebertaV2Config
from transformers import DebertaV2ForMaskedLM
from pathlib import Path
import logging
from transformers import DebertaV2Tokenizer, AutoTokenizer, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset

from argparse import ArgumentParser
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

SEED = 666

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

parser = ArgumentParser()
parser.add_argument('--train_norm_file', required=True, type=str)
parser.add_argument('--val_norm_file', required=True, type=str)
parser.add_argument('--save_dir', required=True, type=str)
args = parser.parse_args()

# add saved dir
for p in ['tokens_len', 'tokenizer_saved']:
    locals()[p+'_dir'] = Path(args.save_dir) / p

# Assembling a corpus
raw_datasets = load_dataset("text", 
    data_files={"train": args.train_norm_file, "val": args.val_norm_file})
print(f"raw_datasets >>>>>>>>> \n {raw_datasets}")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_saved_dir)


def count_len(element):
    outputs = tokenizer(
        element["text"],
        truncation=False,
        padding=False,
        # max_length=512,
        # return_overflowing_tokens=True,
        return_length=True,
        # return_tensors='pt'
    )
    tokens_len = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        tokens_len.append(length)
    return {"tokens_len": tokens_len}

tokens_lens = raw_datasets.map(
    count_len, batched=True, remove_columns=raw_datasets["train"].column_names
)

# 
tokens_lens.save_to_disk(tokens_len_dir)
print(tokens_lens)