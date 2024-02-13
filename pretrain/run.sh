#!/bin/sh

cache_dir=./cache
save_dir=E:/steelberta_data/output
train_corpus_file=E:/steelberta_data/train_corpus.json
val_corpus_file=E:/steelberta_data/val_corpus.json
train_norm_file=E:/steelberta_data/train_corpus_norm.txt
val_norm_file=E:/steelberta_data/val_corpus_norm.txt
corpus_range=material

# python -u json_combination.py

# python -u corpus_normalize.py \
#     --train_corpus_file $train_corpus_file \
#     --val_corpus_file $val_corpus_file \
#     --train_norm_file $train_norm_file \
#     --val_norm_file $val_norm_file \
#     --corpus_range $corpus_range

# python -u tokenizer_train.py \
#     --train_norm_file $train_norm_file \
#     --val_norm_file $val_norm_file \
#     --save_dir $save_dir \
#     --cache_dir $cache_dir

# python -u tokens_count.py \
#     --train_norm_file $train_norm_file \
#     --val_norm_file $val_norm_file \
#     --save_dir $save_dir \

python -u model_train.py \
    --save_dir $save_dir \
    --cache_dir $cache_dir