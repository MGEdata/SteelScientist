import logging
import torch
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from torch.nn import functional as F

from datasets import load_dataset, Dataset, DatasetDict

from transformers import (
    AutoConfig,
    AutoTokenizer,
    DebertaV2ForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# # wandb seeting
os.environ["WANDB_API_KEY"] = '02b38da752f7eb4bae5d07de169f4f9f0edaae70'
os.environ["WANDB_PROJECT"] = 'steelberta'
os.environ["WANDB_MODE"] = 'offline'


# logging.basicConfig(level=logging.INFO)
SEED = 666
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

parser = ArgumentParser()
parser.add_argument('--cache_dir', default=None, type=str)
parser.add_argument('--save_dir', required=True, type=str)
args = parser.parse_args()

# add saved dir
for p in ['tokenizer_saved', 'tokens_saved', 'model_saved', 'model_final_saved']:
    locals()[p+'_dir'] = Path(args.save_dir) / p

tokenizer = AutoTokenizer.from_pretrained(tokenizer_saved_dir)

config  = AutoConfig.from_pretrained('microsoft/deberta-v3-base', cache_dir=args.cache_dir)
print(f'model config: \n {config}')

model = DebertaV2ForMaskedLM(config=config)
model_size = sum(t.numel() for t in model.parameters())
print(f"Pretrained model: {model_size/1000**2:.1f}M parameters")

# Resize input token embeddings matrix of the model 
# if new_num_tokens != config.vocab_size.
model.resize_token_embeddings(len(tokenizer))

# load tokens
dataset_train = Dataset.load_from_disk(Path(tokens_saved_dir) / 'train')
dataset_val = Dataset.load_from_disk(Path(tokens_saved_dir) / 'val')
print(dataset_train, dataset_val)
# print('*'*20, dataset_train[:5], '\n')
# print('*'*20, dataset_val[:5], '\n')
# print(dataset_train[:5], dataset_val[:5])


dataset_train.set_format(type='torch', columns=['input_ids'])
dataset_val.set_format(type='torch', columns=['input_ids'])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# def compute_metrics(eval_pred):
    
#     preds, labels = eval_pred
#     print("*"*100)
#     print(type(preds), preds.shape, preds)
#     print(type(labels), labels.shape, labels)

#     # logits = torch.from_numpy(pred.predictions)
#     # labels = torch.from_numpy(pred.label_ids)
#     # loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
#     # return {'perplexity': math.exp(loss), 'calculated_loss': loss}
#     return {'perplexity': 1, 'calculated_loss': 1}

# from pynvml import *


# def print_gpu_utilization():
#     nvmlInit()
#     handle = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(handle)
#     print(f"GPU memory occupied: {info.used//1024**2} MB.")


# def print_summary(result):
#     print(f"Time: {result.metrics['train_runtime']:.2f}")
#     print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
#     print_gpu_utilization()

# print_gpu_utilization()

training_args = TrainingArguments(
    output_dir=model_saved_dir,
    overwrite_output_dir=True,
    evaluation_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=80,
    gradient_accumulation_steps=72,
    # load_best_model_at_end=True,

    learning_rate=1e-4,
    weight_decay=1e-2,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    max_grad_norm=1.0,
    num_train_epochs=1,
    lr_scheduler_type='linear',
    warmup_ratio=0.048,
    warmup_steps=10_000,

    # eval_steps=10,
    # save_strategy='steps',
    # logging_strategy='steps',
    # logging_steps=2,
    # save_steps=10,
    # save_total_limit=10,
    # seed=SEED,
    # data_seed=SEED,
    # fp16=True,
    # optim='adamw_torch',
    # max_steps=100

    eval_steps=1,
    save_strategy='steps',
    logging_strategy='steps',
    logging_steps=10,
    save_steps=1,
    save_total_limit=10,
    seed=SEED,
    data_seed=SEED,
    fp16=True,
    optim='adamw_torch',
    # max_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    tokenizer=tokenizer,
)


result = trainer.train()
print_summary(result)
trainer.save_model(model_final_saved_dir)

# # resume = None if len(os.listdir(model_saved_dir)) == 0 else True
# # train_res = trainer.train(resume_from_checkpoint=model_final_saved_dir)
# # print(train_res)

# train_output = trainer.evaluate(dataset_train)
# eval_output = trainer.evaluate()

# print(train_output)
# print(eval_output)