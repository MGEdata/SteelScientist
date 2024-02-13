import json
from tqdm import tqdm
import random
from collections import Counter, OrderedDict

random.seed(123)

with open(r"E:\\steelberta_data\\steel_abstract.json", 'r', encoding='utf-8') as f1:
    abs_text = json.load(f1)

with open(r"E:\\steelberta_data\\steel_full_text_v3.json", 'r', encoding='utf-8') as f2:
    full_text = json.load(f2)
    
abs_text['steel'] = full_text['steel']\


# abstract corpus statistics
for k,v in abs_text.items():
    print(f"{k}: {len(abs_text[k])}")


# full-text corpus statistics
publisher = []
for i in tqdm(range(len(full_text['steel']))):
    if len(full_text['steel'][i]['content'])>0:
        publisher.append(full_text['steel'][i]['publisher'])
print(Counter(publisher))

# export as train file and eval file
train_dict = {}
eval_dict = {}
for k in ['steel', 'article', 'meeting', 'patent']:
    train_dict[k], eval_dict[k] = [], []

    temp_li = [i for i in range(len(abs_text[k]))]
    eval_idx = random.sample(temp_li, round(len(temp_li)/10)+1)
    print(f"{k}_sum:{len(temp_li)}, eval:{len(eval_idx)}, train:{len(temp_li)-len(eval_idx)}")
    for i, text in enumerate(tqdm(abs_text[k])):
        if i in eval_idx:
            eval_dict[k].append(text)
        else:
            train_dict[k].append(text)
            
with open('E:\\steelberta_data\\train_corpus.json', 'w', encoding='utf-8') as p1:
    json.dump(train_dict, p1, indent=4, allow_nan=False, ensure_ascii=False)
    
with open('E:\\steelberta_data\\val_corpus.json', 'w', encoding='utf-8') as p2:
    json.dump(eval_dict, p2, indent=4, allow_nan=False, ensure_ascii=False)


# export corpus as one file
new_dic = {}
for k,v in abs_text.items():
    new_dic[k] = abs_text[k]
new_dic['steel'] = full_text['steel']

with open(r"E:\\steelberta_data\\steelberta_corpus.json", 'w', encoding='utf-8') as ff:
    json.dump(new_dic, ff, indent=4, allow_nan=False, ensure_ascii=False)