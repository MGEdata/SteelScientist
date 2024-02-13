import re
import json
import random
from tqdm import tqdm
from argparse import ArgumentParser
from tokenizers.normalizers import BertNormalizer

random.seed(123)

parser = ArgumentParser()
parser.add_argument('--train_corpus_file', required=True, type=str)
parser.add_argument('--val_corpus_file', required=True, type=str)
parser.add_argument('--train_norm_file', required=True, type=str)
parser.add_argument('--val_norm_file', required=True, type=str)
parser.add_argument('--corpus_range', required=True, type=str, default='steel',
                    choices=['material', 'steel'])
args = parser.parse_args()


def load_corpus(path, corpus_range):
    """
    Select whether to add corpus for other materials.
    """
    with open(path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    texts = []
    # single steel corpus
    if corpus_range == 'steel':
        for content in corpus['steel']:
            texts.append(content['abstract'])
            if content['content'] == '':
                continue
            else:
                for para in content['content']:
                    texts.append(para)
    # steel corpus and abstracts of other materials
    else:
        for content in corpus['steel']:
            if content['content'] == '':
                continue
            else:
                for para in content['content']:
                    texts.append(para)
        for s in ['article', 'patent', 'meeting']:
            for content in corpus[s]:
                texts.append(content['abstract'])
    return texts

normalize = BertNormalizer(
    lowercase=False,
    strip_accents=True,
    clean_text=True,
    handle_chinese_chars=True
)

def normalize_export_file(corpus_path, corpus_range, save_path):
    """
    Normalize text and export as TXT file.
    """
    texts = load_corpus(corpus_path, corpus_range)
    corpus_norm = [normalize.normalize_str(sent) for sent in tqdm(texts)]
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(corpus_norm))

normalize_export_file(args.train_corpus_file, args.corpus_range, args.train_norm_file)
normalize_export_file(args.val_corpus_file, args.corpus_range, args.val_norm_file)

