#!/usr/bin/env python
# coding: utf-8
#!/usr/bin/python
# -*- coding: UTF-8 -*-
############### built-in modules ###############
import gc
import random
import shutil
import re
import os
import csv
import warnings
from typing import Optional
from datetime import datetime
from filelock import FileLock

import pandas as pd
import numpy as np
############### pytorch ###############
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# from datasets import Dataset
############### ray tune ###############
import ray
from ray import air, tune, train
from ray.air import session
from ray.air.integrations.wandb import setup_wandb, WandbLoggerCallback
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import PopulationBasedTraining, ASHAScheduler, HyperBandScheduler
############### utils ###############
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
# from torchviz import make_dot

warnings.filterwarnings('ignore')
os.environ["WANDB_MODE"] = 'offline'
# os.environ["RAY_AIR_NEW_OUTPUT"] = 0
# os.environ["RAY_DEDUP_LOGS"] = "0"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
print(os.getcwd())

ALL_ACT_LAYERS = {
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}

ALL_LOSS_FUNC = {
    "mseloss": nn.MSELoss,
    "l1loss": nn.L1Loss,
    "smoothl1loss": nn.SmoothL1Loss
}

ALL_OPTIM_FUNC = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD
}

def gpu_cache_clc():
    """
    Clear cuda cache.
    """
    for i in range(5):
        torch.cuda.empty_cache()
        gc.collect()

def set_global_seed(seed=123):
    # for Tensorflow
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.random.set_seed(seed)
        tf.experimental.numpy.random.seed(seed)
        tf.set_random_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def mk_sync_dir(tune_dir_path, dst="wandb"):
    """
    Creat wnadb folder by extracting log files.
    """
    for d in os.listdir(tune_dir_path):
        dir_name = os.path.join(tune_dir_path, d)
        if os.path.isdir(dir_name):
            temp_path = os.path.join(dir_name, "wandb")
            if os.path.isdir(temp_path):
                for t in os.listdir(temp_path):
                    if not re.findall(r"debug|latest", t):
                        temp_dst = os.path.join(dst, t)
                        src = os.path.join(temp_path, t)
                        shutil.copytree(src, temp_dst, dirs_exist_ok=True)

def range_data(temp_df, ts_range=[10, 3000], ys_range=[0, 3000], el_range=[5, 150]):
    """
    Return property value in ranges.
    """
    temp_df = temp_df[(temp_df['Tensile_value']>=ts_range[0]) & (temp_df['Tensile_value']<=ts_range[1])]
    temp_df = temp_df[(temp_df['Yield_value']>=ys_range[0]) & (temp_df['Yield_value']<=ys_range[1])]
    temp_df = temp_df[(temp_df['Elongation_value']>=el_range[0]) & (temp_df['Elongation_value']<=el_range[1])]
    temp_df.reset_index(drop=True, inplace=True)
    return temp_df

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddinngs(text_list):
    """
    Get text [cls] embeddings.
    """
    encoded_input = tokenizer(
        text_list, padding='max_length', max_length = 512, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output).detach().cpu().numpy()[0].tolist()

def gen_text_embed(bert_df, col_embed='Text'):
    """
    Get text embeddings and return DataFrame with encodings columns.
    """
    tqdm.pandas(desc='Progress bar!')
    bert_df['process_emb'] = bert_df[col_embed].progress_apply(get_embeddinngs)

    temp_bert_df = pd.DataFrame(pd.Series(bert_df['process_emb'][0])).T
    for i in range(1, bert_df.shape[0]):
        new_row = pd.DataFrame(pd.Series(bert_df['process_emb'][i])).T
        temp_bert_df = pd.concat([temp_bert_df, new_row], ignore_index=True, axis=0)
    temp_bert_df.reset_index(drop=True, inplace=True)
    temp_bert_df.columns = [col_embed+str(i) for i in range(768)]
    df = pd.concat([bert_df, temp_bert_df], axis=1)
    df.drop(columns=['process_emb'], inplace=True)

    return df

def get_ele_embeddinngs(text_list, ):
    """
    Get element text embeddings and return DataFrame with ele encodings.
    """
    encoded_input = tokenizer(
        text_list, padding='max_length', max_length = 512, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)

    return cls_pooling(model_output).detach().cpu().numpy()

def add_ele_embed(df):
    """
    Return DataFrame with composition columns.
    """
    coms = list(df.columns[3:-770])
    x = np.matrix(get_ele_embeddinngs(coms))
    w1 = list(df.iloc[0, 3:-770])
    y = np.average(x, axis=0, weights=w1)

    for i in np.arange(1, df.shape[0]):
        w = list(df.iloc[i, 3:-770])
        temp_y = np.average(x, axis=0, weights=w)
        y = np.vstack((y, temp_y))

    ele_df = pd.DataFrame(y, columns=['ele'+str(i) for i in range(768)])

    return ele_df


def load_data(data_origin, label='train_data', pred_prop='Yield_value', split_ratio=0.75,
              fes=['com', 'com_embed', 'text_embed', 'action_embed'], seed=555, perplexity=0.1):
    """
    Parameters
    ----------
    label
        ['train_data', 'text_test', 'exp_test']
    pred_prop
        ['Tensile_value', 'Yield_value', 'Elongation_value', 'all']
    fes
        ['com', 'text_embed', 'com_embed', text', 'action_embed']
    split_ratio
        train data : test data
    perplexity
        perplexity of composition TSNE method
    """
    # data_origin = pd.read_excel(f'./datasets/{label}.xlsx')
    com_cols = set(data_origin.columns[17:-4])

    data = data_origin.copy()

    # fill composition
    for e_col in com_cols:
        data[e_col].fillna(0.0, inplace=True)
    data.fillna('', inplace=True)

    # drop some description cols
    drop_cols = ['DOIs', 'Files', 'problem', 'status', 'Table_topic', 'title', 'abstract', 'Other_ele', \
                 'Text_addition', 'Tensile_name', 'Tensile_unit', 'Yield_name', 'Yield_unit', \
                 'Elongation_name', 'Elongation_unit', 'Material']
    data.drop(columns=drop_cols, inplace=True)

    # processing text embeddings
    data = gen_text_embed(data)

    com_drop = ['Fe']
    data.drop(columns=com_drop, inplace=True)

    # compostion embeddings
    data = pd.concat([data, add_ele_embed(data)], axis=1)

    # property selection
    prop_all = ['Tensile_value', 'Yield_value', 'Elongation_value']
    if pred_prop == 'all':
        drop_col = []
    else:
        drop_col = list(set(prop_all) - set([pred_prop]))
    # actions embeddings
    data = gen_text_embed(data, col_embed='actions')

    # features selection
    if 'com' not in fes:
        drop_col += [col for col in list(data.columns) if col in com_cols]
    if 'text_embed' not in fes:
        drop_col += ['Text'+str(i) for i in range(768)]
    if 'com_embed' not in fes:
        drop_col += ['ele'+str(i) for i in range(768)]
    if 'action_embed' not in fes:
        drop_col += ['actions'+str(i) for i in range(768)]
    if 'text' not in fes:
        drop_col.append('Text')
    data.drop(columns=drop_col+['actions'], inplace=True)

    if perplexity:
        if 'com_embed' in fes:
            # data_com_embed = pd.DataFrame(TSNE(n_components=3, learning_rate='auto',
            #                   init='random', perplexity=perplexity).fit_transform(data.iloc[:, -768:]),
            #                          columns=['tsne'+str(i) for i in range(3)])

            data_com_embed = pd.DataFrame(np.random.randint(2, size=(1,3)), columns=['tsne'+str(i) for i in range(3)])
            data = pd.concat([data, data_com_embed], axis=1)

    if label=='train_data':
        data.reset_index(drop=True, inplace=True)
    return data


def gen_data_class(df):

    eles_cols = list(df.columns[1:-2307])
    text_embeds = ['Text'+str(i) for i in range(768)]
    com_embeds = ['ele'+str(i) for i in range(768)]
    com_tsne = ['tsne'+str(i) for i in range(3)]
    action_embeds = ['actions'+str(i) for i in range(768)]

    dic = {
        "targets": df.iloc[:, 0],
        "eles": df.loc[:, eles_cols],
        "text_embeds": df.loc[:, text_embeds],
        "com_embeds": df.loc[:, com_embeds],
        "action_embeds": df.loc[:, action_embeds],
        "com_tsne_embeds": df.loc[:, com_tsne]
    }
    return dic


def eval_model(y_true, y_pred):
    """
    Model evaluation for predicted results with "r2" and "rootmean squared error"
    """
    y_true = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred

    r2 = round(r2_score(y_true, y_pred), 3)
    rmse = round(mean_squared_error(y_true, y_pred, squared=False), 3)
    mae = round(mean_absolute_error(y_true, y_pred), 3)
    result = {"r2":r2, "rmse":rmse, "mae":mae}
    return result

def plot_test_data(best_new_text_y, best_new_text_preds, text_result,
                   best_exp_y, best_exp_preds, exp_result, labels, fig_name="./outputs/test.png",point_size=10):

    text_val = [text_result['r2'], text_result['rmse'], text_result['mae']]
    exp_val = [exp_result['r2'], exp_result['rmse'], exp_result['mae']]
    ############## plot #########################
    fig = plt.figure(figsize=(10,5), dpi=120)

    if isinstance(best_new_text_y, torch.Tensor):
        # on new literature data
        y_new_text = best_new_text_y.detach().cpu().numpy()
        y_new_text_pred = best_new_text_preds.detach().cpu().numpy()

        # on exp data
        y_exp = best_exp_y.detach().cpu().numpy()
        y_exp_pred = best_exp_preds.detach().cpu().numpy()
    else:
        y_new_text, y_new_text_pred = best_new_text_y, best_new_text_preds
        y_exp, y_exp_pred = best_exp_y, best_exp_preds
    print(type(y_exp), y_exp.shape)


    plt.subplot(121)
    plt.scatter(y_new_text, y_new_text_pred, s=point_size, color='#1F4B73')
    # add annotatd text for experiment data
    if y_new_text.shape[0] < 100:
        point_labels = list(pd.read_excel(f"{root_dir}/datasets/text_test.xlsx")['abstract'])
        assert len(point_labels) == y_new_text.shape[0]
        for i, txt in enumerate(point_labels):
            plt.annotate(txt, (y_new_text[i, 0], y_new_text_pred[i, 0]))
    plt.plot(np.arange(int(np.max(y_new_text)*0.1), int(np.max(y_new_text)*1.1)), np.arange(int(np.max(y_new_text)*0.1),int(np.max(y_new_text)*1.1)), '-', color='#A2555A')
    plt.title(f"$R^2$={text_val[0]}, RMSE={text_val[1]}, MAE={text_val[2]}",
              fontdict={'family':'Times New Roman', 'size': 10})
    plt.ylabel('prediction', fontdict={'family':'Times New Roman', 'size': 14})
    plt.xlabel('True', fontdict={'family':'Times New Roman', 'size': 14})
    plt.xlim(0, int(np.max(y_new_text)*1.2))
    plt.ylim(0, int(np.max(y_new_text)*1.2))
    plt.legend(labels=[labels[0],'Y=X'])
    plt.grid()

    # on new exp data
    plt.subplot(122)
    plt.scatter(y_exp, y_exp_pred, s=point_size, color='#1F4B73')
    # add annotatd text for experiment data
    if y_exp.shape[0] < 100:
        point_labels = list(pd.read_excel(f"{root_dir}/datasets/exp_test.xlsx")['DOIs'])
        assert len(point_labels) == y_exp.shape[0]
        for i, txt in enumerate(point_labels):
            plt.annotate(txt, (y_exp[i, 0], y_exp_pred[i, 0]))
    plt.plot(np.arange(int(np.max(y_exp)*0.1), int(np.max(y_exp)*1.1)), np.arange(int(np.max(y_exp)*0.1), int(np.max(y_exp)*1.1)), '-', color='#A2555A')
    plt.title(f"$R^2$={exp_val[0]}, RMSE={exp_val[1]}, MAE={exp_val[2]}",
              fontdict={'family':'Times New Roman', 'size': 10})
    plt.ylabel('prediction', fontdict={'family':'Times New Roman', 'size': 14})
    plt.xlabel('True', fontdict={'family':'Times New Roman', 'size': 14})
    plt.xlim(0,int(np.max(y_exp)*1.2))
    plt.ylim(0,int(np.max(y_exp)*1.2))
    plt.legend(labels=[labels[1],'Y=X'])
    plt.grid()

    plt.savefig(fig_name)

# containging text and ele features for pytorch
class CustomSimpleDataset(Dataset):
    def __init__(self, eles, text_embeds, com_embeds, com_tsne_embeds, action_embeds, targets):

        self.eles = eles.values if isinstance(eles, pd.DataFrame) else eles
        self.text_embeds = text_embeds.values if isinstance(text_embeds, pd.DataFrame) else text_embeds
        self.com_embeds = com_embeds.values if isinstance(com_embeds, pd.DataFrame) else com_embeds
        self.com_tsne_embeds = com_tsne_embeds.values if isinstance(com_tsne_embeds, pd.DataFrame) else com_tsne_embeds
        self.action_embeds = action_embeds.values if isinstance(action_embeds, pd.DataFrame) else action_embeds
        self.targets = targets.values if isinstance(targets, pd.DataFrame) else targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ele = self.eles[idx]
        text_embed = self.text_embeds[idx]
        com_embed = self.com_embeds[idx]
        com_tsne_embed = self.com_tsne_embeds[idx]
        action_embed = self.action_embeds[idx]
        target = self.targets[idx]

        return {
            "labels": torch.tensor(target, dtype=torch.float32).to(device),
            "eles": torch.tensor(ele, dtype=torch.float32).squeeze(0).to(device),
            "text_embeds": torch.tensor(text_embed, dtype=torch.float32).squeeze(0).to(device),
            "com_embeds": torch.tensor(com_embed, dtype=torch.float32).squeeze(0).to(device),
            "action_embeds": torch.tensor(action_embed, dtype=torch.float32).squeeze(0).to(device),
            "com_tsne_embeds": torch.tensor(com_tsne_embed, dtype=torch.float32).squeeze(0).to(device),
        }

class Unit(nn.Module):
    """
    One MLP layer. It orders the operations as: norm -> fc -> act_fn -> dropout
    """
    def __init__(
        self,
        normalization: str,
        in_features: int,
        out_features: int,
        activation: str,
        dropout_prob: float,
    ):
        """
        Parameters
        ----------
        normalization
            Name of activation function.
        in_features
            Dimension of input features.
        out_features
            Dimension of output features.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        """
        super().__init__()
        if normalization == "layer_norm":
            self.norm = nn.LayerNorm(in_features)
        elif normalization == "batch_norm":
            self.norm = nn.BatchNorm1d(in_features)
        elif normalization == "null_norm":
            self.norm = None
        else:
            raise ValueError(f"unknown normalization: {normalization}")
        self.fc = nn.Linear(in_features, out_features)
        if activation == "leaky_relu":
            self.act_fn = nn.LeakyReLU(negative_slope=-1)
        else:
            self.act_fn = ALL_ACT_LAYERS[activation]()

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # pre normalization
        if self.norm is not None:
            x = self.norm(x)
        else:
            pass
        x = self.fc(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x

class FcUnit(nn.Module):
    """
    One MLP layer. It orders the operations as: norm -> fc -> act_fn -> dropout
    """
    def __init__(
        self,
        num_features: int,
        normalization: Optional [str] = "null_norm",
        activation: Optional [str] = 'relu',
        dropout_prob: Optional [float] = 0.1,
    ):
        """
        Parameters
        ----------
        num_features
            Nodes of next network layer.
        normalization
            Type of activation function.
        activation
            Type of activation function.
        dropout_prob
            Dropout probability.
        """
        super().__init__()
        if normalization == "layer_norm":
            self.norm = nn.LayerNorm(num_features)
        elif normalization == "batch_norm":
            self.norm = nn.BatchNorm1d(num_features)
        elif normalization == "null_norm":
            self.norm = None
        else:
            raise ValueError(f"unknown normalization: {normalization}")

        self.fc = nn.LazyLinear(num_features)
        if activation == "leaky_relu":
            self.act_fn = nn.LeakyReLU(negative_slope=-1)
        else:
            self.act_fn = ALL_ACT_LAYERS[activation]()

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # pre normalization
        if self.norm is not None:
            x = self.norm(x)
        else:
            pass
        x = self.fc(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x

class Cnn1dUnit(nn.Module):
    """
    One 1d CNN layer. It orders the operations as: CNN -> Pooling
    """
    def __init__(
        self,
        out_channels: int,
        kernel_size: Optional [int] = 3,
        activation: Optional [str] = 'relu',
        stride: Optional [int] = 1,
        padding: Optional [int] = 1,
        pooling: Optional [str] = 'max',
    ):
        """
        Parameters
        ----------
        out_channels
            Channels of output.
        pooling
            ['max', 'avg']
        """
        super().__init__()
        if pooling == "max":
            self.pool = nn.MaxPool1d(kernel_size=kernel_size)
        elif pooling == "avg":
            self.pool = nn.AvgPool1d(kernel_size=kernel_size)
        else:
            raise ValueError(f"unknown pooling type: {pooling}")

        if activation == "leaky_relu":
            self.act_fn = nn.LeakyReLU(negative_slope=-1)
        else:
            self.act_fn = ALL_ACT_LAYERS[activation]()

        self.conv1d = nn.LazyConv1d(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        x = self.conv1d(x)
        x = self.act_fn(x)
        x = self.pool(x)
        return x


class Cnn2dUnit(nn.Module):
    """
    One 2d CNN layer. It orders the operations as: CNN -> Pooling
    """
    def __init__(
        self,
        out_channels: int,
        kernel_size: Optional [int] = 2,
        activation: Optional [str] = 'relu',
        stride: Optional [int] = 1,
        padding: Optional [int] = 1,
        pooling: Optional [str] = 'max',
    ):
        """
        Parameters
        ----------
        out_channels
            Channels of output.
        pooling
            ['max', 'avg']
        """
        super().__init__()
        if pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=kernel_size)
        elif pooling == "avg":
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)
        else:
            raise ValueError(f"unknown pooling type: {pooling}")

        if activation == "leaky_relu":
            self.act_fn = nn.LeakyReLU(negative_slope=-1)
        else:
            self.act_fn = ALL_ACT_LAYERS[activation]()


        self.conv2d = nn.LazyConv2d(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        x = self.conv2d(x)
        x = self.act_fn(x)
        x = self.pool(x)
        return x

class CustomSimpleModel(nn.Module):
    def __init__(
            self,
            simple_layer_list,
            concat_layer_list,
            seq_embed_con1d_list,
            seq_embed_fc_list,
            seq_embed_con2d_list,
            seq_embed_2d_fc_list,
            simple_layer_drop_prob=0.0,
            concat_layer_drop_prob=0.0,
            layer_pooling='max',
            mix_layer=3
    ):
        self.mix_layer = mix_layer
        super(CustomSimpleModel, self).__init__()
        # self.com_layer = nn.Sequential(nn.LazyLinear(4), nn.ReLU())
        # self.com_tsne_embed_layer = nn.Sequential(nn.LazyLinear(2), nn.ReLU())

        # initial simple layer
        simple_layer = []
        for num_fes in simple_layer_list:
            per_unit = FcUnit(
                num_features=num_fes,
                normalization='null_norm',
                activation='relu',
                dropout_prob=simple_layer_drop_prob
            )
            simple_layer.append(per_unit)
        self.simple_layer = nn.Sequential(*simple_layer)

        # con1d seq_embed_layer for [text_embed, action_embed and compostion embed]
        seq_embed_layer = []
        for num_channels in seq_embed_con1d_list:
            per_unit = Cnn1dUnit(
                out_channels=num_channels,
                kernel_size=2,
                stride=1,
                padding=1,
                pooling=layer_pooling
            )
            seq_embed_layer.append(per_unit)
        seq_embed_layer.append(nn.Flatten())
        for num_fes in seq_embed_fc_list:
            per_unit = FcUnit(
                num_features=num_fes,
                normalization='null_norm',
                activation='relu',
                dropout_prob=simple_layer_drop_prob
            )
            seq_embed_layer.append(per_unit)
        self.seq_embed_layer = nn.Sequential(*seq_embed_layer)


        # common con2d CNN layers
        common_embed_layer = []
        for num_channels in seq_embed_con2d_list:
            per_unit = Cnn2dUnit(
                out_channels=num_channels,
                kernel_size=2,
                stride=1,
                padding=2,
                pooling=layer_pooling
            )
            common_embed_layer.append(per_unit)
        common_embed_layer.append(nn.Flatten())
        for num_fes in seq_embed_2d_fc_list:
            per_unit = FcUnit(
                num_features=num_fes,
                normalization='null_norm',
                activation='relu',
                dropout_prob=simple_layer_drop_prob
            )
            common_embed_layer.append(per_unit)
        self.common_embed_layer = nn.Sequential(*common_embed_layer)

        # simple layer
        concat_layer = []
        for num_fes in concat_layer_list:
            per_unit = FcUnit(
                num_features=num_fes,
                normalization='null_norm',
                activation='relu',
                dropout_prob=concat_layer_drop_prob
            )
            concat_layer.append(per_unit)
        self.concat_layer = nn.Sequential(*concat_layer)


    def forward(self, eles, text_embeds, com_embeds, com_tsne_embeds, action_embeds, labels):

        inputs_addition = torch.concat([com_embeds.unsqueeze(1), text_embeds.unsqueeze(1)], dim=1)
        for i in range(self.mix_layer):
            inputs_addition = torch.concat([inputs_addition, com_embeds.unsqueeze(1), text_embeds.unsqueeze(1)], dim=1)
        # print(f'inputs_addition shape {inputs_addition.shape}')

        # inputs_addition = torch.concat([com_embeds.unsqueeze(1), text_embeds.unsqueeze(1), action_embeds.unsqueeze(1)], dim=1)
        # inputs_addition = torch.concat([com_embeds.unsqueeze(1), text_embeds.unsqueeze(1)], dim=1)
        com_dense_output = self.simple_layer(com_embeds)
        text_dense_output = self.simple_layer(text_embeds)
        # act_dense_output = self.simple_layer(action_embeds)

        com_embed_output = self.seq_embed_layer(com_dense_output.unsqueeze(1))
        text_embed_output = self.seq_embed_layer(text_dense_output.unsqueeze(1))

        # action_embed_output = self.seq_embed_layer(act_dense_output.unsqueeze(1))

        output_addition = self.common_embed_layer(inputs_addition.unsqueeze(1))

        # output = torch.concat([output_addition, com_embed_output, text_embed_output, action_embed_output], dim=1)
        output = torch.concat([output_addition, com_embed_output, text_embed_output], dim=1)
        output = self.concat_layer(output)

        return output

def util_data(y_data, max_len=10_000):
    y_data = y_data.T.tolist()[0]
    if len(y_data) < max_len:
        y_data += [6666.0 for i in range(max_len-len(y_data))]
    return y_data


# def train_function(config, data):
def train_function(config):
    """
    training on ensemble model
    """
    sol_temp = config["sol_temp"]
    sol_time = config["sol_time"]
    first_cr_ratio = config["first_cr_ratio"]
    first_tep_temp = config["first_tep_temp"]
    first_tep_time = config["first_tep_time"]
    second_cr_ratio = config["second_cr_ratio"]
    second_tep_temp = config["second_tep_temp"]
    second_tep_time = config["second_tep_time"]
    com_base = config['com_base']

    print(config)
    ################# build processing routes
    init_text = f"""
    The experimental steel was melted in vacuum induction melting furnace at first, and then hot-forged.
    After Some small plates were cut from the as-received hot forged plate, and then solution-treated at {sol_temp} °C for {sol_time} h. 
    """
    cr_1 = f"""
    The solution-treated specimen was cold rolled to a thickness reduction of {first_cr_ratio}%. 
    """
    tp_1 = f"""
    The the cold rolled specimen was performed by tempering at temperature of {first_cr_ratio} °C for {first_tep_time} min, followed by air cooling to room temperature. 
    """
    cr_2 = f"""
    The tempered specimen was cold rolled to a thickness reduction of {second_cr_ratio}%. 
    """
    tp_2 = f"""
    Finally, the specimen was performed by secondary tempering at temperature of {second_tep_temp} °C for {second_tep_time} min. 
    """

    seq = []
    if first_cr_ratio=='null':
        seq.append('F')
    else:
        seq.append('T')

    if 'null' in [first_tep_temp, first_tep_time]:
        seq.append('F')
    else:
        seq.append('T')

    if second_cr_ratio=='null':
        seq.append('F')
    else:
        seq.append('T')

    if 'null' in [second_tep_temp, second_tep_time]:
        seq.append('F')
    else:
        seq.append('T')

    route_dict = {
        'FFFF': init_text,
        'FTFF': init_text + tp_1,
        'TTFF': init_text + cr_1 + tp_1,
        'FTFT': init_text + tp_1 + tp_2,
        'TTFT': init_text + cr_1 + tp_1 + tp_2,
        'FTTT': init_text + tp_1 + cr_2 + tp_2,
        'TTTT': init_text + cr_1 + tp_1 + cr_2 + tp_2,
    }
    # print('*'*100, seq)
    route_text = route_dict.get(seq[0], '')
    print('*'*100, route_text)

    # build dataloader
    # module_df = pd.read_excel(r'./module_opt.xlsx')
    # module_df.iat[com_base, -3] = route_text
    # module_df.drop(index=set([0, 1, 2])-set([com_base]), inplace=True)
    # possible_data = load_data(module_df, pred_prop='all')

    # possible_data = gen_data(route_text, config)

    # for prop in ['Tensile_value', 'Yield_value', 'Elongation_value']:
    #     temp_drop_col = list(set(['Tensile_value', 'Yield_value', 'Elongation_value']) - set([prop]))
    #     temp_data = possible_data.drop(columns=temp_drop_col, inplace=False).copy()
    #     data_dataloader = DataLoader(CustomSimpleDataset(**gen_data_class(possible_data)),
    #                 batch_size=len(temp_data), shuffle=False, drop_last=False)

    #     if prop=='Tensile_value':
    #         ts_model.eval()
    #         with torch.no_grad():
    #             for batch, inputs in enumerate(data_dataloader):
    #                 ts_preds = ts_model(**inputs).detach().cpu().numpy()[0]
    #     elif prop=='Yield_value':
    #         ys_model.eval()
    #         with torch.no_grad():
    #             for batch, inputs in enumerate(data_dataloader):
    #                 ys_preds = ys_model(**inputs).detach().cpu().numpy()[0]
    #     else:
    #         el_model.eval()
    #         with torch.no_grad():
    #             for batch, inputs in enumerate(data_dataloader):
    #                 el_preds = el_model(**inputs).detach().cpu().numpy()[0]

    #     print(ts_preds, ys_preds, el_preds)



    # report_metrics = {
    #     'train_epoch': epoch_num+1,
    #     'train_loss':train_loss.item(),
    #     'val_loss':val_loss,
    #     'train_r2':train_r2,
    #     'val_r2':val_r2,
    #     'new_text_r2':new_text_r2,
    #     'exp_r2':exp_r2
    # }
    # train.report(report_metrics)

    # ######################## save result  ###################
    # with open(f"{base_path}/reg_model.csv", 'a+') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(
    #         [train_result['r2'], val_result['r2'], best_new_text_result['r2'],
    #         best_exp_result['r2']] + paras_li + [plot_best_text_result['r2']]
    #     )

def stop_fn(trial_id: str, result: dict) -> bool:
    # return (result["val_r2"] <= -2.0 and result["train_epoch"] > 20) or (result["val_r2"] <= 0 and result["train_epoch"] > 40)
    return result["val_r2"] <= -0.8 and result["train_epoch"] > 50


def tune_with_callback():
    """
    Tune hyper-parameters on multimodal model.
    """
    tuner = tune.Tuner(
        # train_function,
        # tune.with_parameters(train_function, data=data),
        tune.with_resources(
            # tune.with_parameters(train_function, data=gen_data(config)),
            train_function,
            # resources={"cpu": 1, "gpu": 1}
            resources={"gpu": 0.25, "cpu":2}
        ),
        tune_config=tune.TuneConfig(
            # metric="val_r2",
            # mode="max",
            num_samples=2,
            max_concurrent_trials=1,
            search_alg = OptunaSearch(
                # metric=["val_r2", "new_text_r2"],
                # mode=["max", "max"]
                metric=["Yield_value", "Elongation_value"],
                mode=['max', 'max']
            )
        #     search_alg = HyperOptSearch(
        #         metric="exp_r2",
        #         mode="max",
        #         points_to_evaluate=current_best_params,
        #         n_initial_points=20,
        #         random_state_seed=666,
        #         gamma=0.25
        #     )
        # #     scheduler = ASHAScheduler(
        #         time_attr='training_iteration',
        #         metric='exp_r2',
        #         mode='max',
        #         max_t=100,
        #         grace_period=10,
        #         reduction_factor=3,
        #         brackets=1)
        ),
        # search_alg=OptunaSearch(metric="agg_second_r2", mode="max"),
        # https://docs.ray.io/en/latest/tune/api/suggestion.html

        # scheduler=ASHAScheduler(),
        # https://docs.ray.io/en/latest/tune/api/schedulers.html#population-based-training-replay-tune-schedulers-populationbasedtrainingreplay

        run_config=train.RunConfig(
            name=exp_name,
            storage_path = base_path,
            # stop=stop_fn
            # callbacks=[
            #     WandbLoggerCallback(
            #         project="reg_model",
            #         mode="offline",
            #         tags=["1017"])
            # ]
        ),
        param_space=search_space
    )
    result_summary = tuner.fit()
    return result_summary



if __name__=='__main__':
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    set_global_seed(seed=123)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sol_temp_space = list(range(1000, 1300, 50))
    sol_time_space = list(range(60, 300, 60))
    cr_ratio_space = [i for i in range(550, 750, 10)] + ['null']
    tep_temp_space = [i for i in range(5, 250, 10)] + ['null']
    tep_time_space = [i for i in range(5, 80, 5)] + ['null']

    search_space = {
            'sol_temp': tune.choice(sol_temp_space),
            'sol_time': tune.choice(sol_time_space),
            'first_cr_ratio': tune.choice(cr_ratio_space),
            'first_tep_temp': tune.choice(tep_temp_space),
            'first_tep_time': tune.choice(tep_time_space),
            'second_cr_ratio': tune.choice(cr_ratio_space),
            'second_tep_temp': tune.choice(tep_temp_space),
            'second_tep_time': tune.choice(tep_time_space),
            'com_base': tune.choice([0, 1, 2]),
    }

    # load saved model
    model_name = './../../model_saved/checkpoint-140000'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    ######## predictive model
    for prop_name in ['ys', 'ts', 'el']:
        ######### yield strength
        if prop_name=='ys':
            prop = 'Yield_value'
            model_save_path = r'C:\Users\shaohan.tian\OneDrive\code\code_github\SteelBERTa\regression\exp_transfer_learning\outputs_all_prop\reg_model_saved\ys_0.8_16_0.005_5257_2000_1_9_model.pt'
            mix_layer = 16
            layer_pooling = 'avg'
            # simple layer
            simple_layer_num_512 = 4
            simple_layer_num_256 = 2
            simple_layer_drop_prob = 0.0

            # concat_layer_list
            concat_layer_num_64 = 1
            concat_layer_num_32 = 3
            concat_layer_drop_prob = 0.0

            conv_channels_layer_1 = 1
            conv_channels_layer_2 = 1

            embed_layer_num_32 = 2
            embed_layer_num_16 = 4

        ######### elongation
        elif prop_name=='el':
            prop = 'Elongation_value'
            model_save_path = r'C:\Users\shaohan.tian\OneDrive\code\code_github\SteelBERTa\regression\exp_transfer_learning\outputs_all_prop\reg_model_saved\el_0.7_8_0.001_7587_2000_1_9_model.pt'
            mix_layer = 14
            layer_pooling = 'max'
            # simple layer
            simple_layer_num_512 = 4
            simple_layer_num_256 = 4
            simple_layer_drop_prob = 0.05

            # concat_layer_list
            concat_layer_num_64 = 1
            concat_layer_num_32 = 7
            concat_layer_drop_prob = 0.0

            conv_channels_layer_1 = 3
            conv_channels_layer_2 = 1

            embed_layer_num_32 = 1
            embed_layer_num_16 = 2

        else:
            prop = 'Tensile_value'
            model_save_path = r'C:\Users\shaohan.tian\OneDrive\code\code_github\SteelBERTa\regression\exp_transfer_learning\outputs_all_prop\reg_model_saved\ts_0.75_16_0.005_8634_2000_1_9_model.pt'
            mix_layer = 8
            layer_pooling = 'max'
            # simple layer
            simple_layer_num_512 = 6
            simple_layer_num_256 = 1
            simple_layer_drop_prob = 0.0

            # concat_layer_list
            concat_layer_num_64 = 3
            concat_layer_num_32 = 5
            concat_layer_drop_prob = 0.0

            conv_channels_layer_1 = 3
            conv_channels_layer_2 = 3

            embed_layer_num_32 = 4
            embed_layer_num_16 = 4

        simple_layer_list = [512 for i in range(simple_layer_num_512)] + [256 for j in range(simple_layer_num_256)]
        concat_layer_list = [64 for i in range(concat_layer_num_64)] + [32 for j in range(concat_layer_num_32)] + [8, 4, 1]
        seq_embed_con1d_list = [conv_channels_layer_1, conv_channels_layer_2]
        seq_embed_con2d_list = [conv_channels_layer_1, conv_channels_layer_2]
        seq_embed_fc_list = [32 for i in range(embed_layer_num_32)] + [16 for j in range(embed_layer_num_16)]
        seq_embed_2d_fc_list = [32 for i in range(embed_layer_num_32)] + [16 for j in range(embed_layer_num_16)]

        if prop_name == 'ys':
            ys_model = CustomSimpleModel(
                simple_layer_list = simple_layer_list,
                concat_layer_list = concat_layer_list,
                seq_embed_con1d_list = seq_embed_con1d_list,
                seq_embed_fc_list = seq_embed_fc_list,
                seq_embed_con2d_list = seq_embed_con2d_list,
                seq_embed_2d_fc_list = seq_embed_2d_fc_list,
                simple_layer_drop_prob = simple_layer_drop_prob,
                concat_layer_drop_prob = concat_layer_drop_prob,
                layer_pooling=layer_pooling,
                mix_layer=mix_layer
            ).to(device)
            ys_model.load_state_dict(torch.load(model_save_path), strict=True)

        elif prop_name == 'ts':
            ts_model = CustomSimpleModel(
                simple_layer_list = simple_layer_list,
                concat_layer_list = concat_layer_list,
                seq_embed_con1d_list = seq_embed_con1d_list,
                seq_embed_fc_list = seq_embed_fc_list,
                seq_embed_con2d_list = seq_embed_con2d_list,
                seq_embed_2d_fc_list = seq_embed_2d_fc_list,
                simple_layer_drop_prob = simple_layer_drop_prob,
                concat_layer_drop_prob = concat_layer_drop_prob,
                layer_pooling=layer_pooling,
                mix_layer=mix_layer
            ).to(device)
            ts_model.load_state_dict(torch.load(model_save_path), strict=True)

        if prop_name == 'el':
            el_model = CustomSimpleModel(
                simple_layer_list = simple_layer_list,
                concat_layer_list = concat_layer_list,
                seq_embed_con1d_list = seq_embed_con1d_list,
                seq_embed_fc_list = seq_embed_fc_list,
                seq_embed_con2d_list = seq_embed_con2d_list,
                seq_embed_2d_fc_list = seq_embed_2d_fc_list,
                simple_layer_drop_prob = simple_layer_drop_prob,
                concat_layer_drop_prob = concat_layer_drop_prob,
                layer_pooling=layer_pooling,
                mix_layer=mix_layer
            ).to(device)
            el_model.load_state_dict(torch.load(model_save_path), strict=True)


    # creat outputs dir
    root_dir = "C:/Users/shaohan.tian/OneDrive/code/code_github/SteelBERTa/regression"
    base_path = f"{root_dir}/exp_optimize/outputs"
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    child_dir = ['preds', 'figs', 'runs']
    for d in child_dir:
        new_dir = f"{base_path}/{d}"
        if not os.path.exists(new_dir):
            os.makedirs(new_dir, exist_ok=True)

    # output column names
    with open(f"{base_path}/output_tl_result.csv", 'a+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            'sol_temp', 'sol_time', 'first_cr_ratio', 'first_tep_temp', 'first_tep_time', 'second_cr_ratio', 'second_tep_temp', 'second_tep_time',
            'ts_pred', 'ys_pred', 'el_pred'
        ])

    storage_path = os.path.expanduser("~/ray_results")
    exp_name = "exp_opt_1220"
    temp_path = os.path.join(storage_path, exp_name)
    results = tune_with_callback()