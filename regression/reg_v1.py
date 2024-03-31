#!/usr/bin/env python
# coding: utf-8
#!/usr/bin/python
# -*- coding: UTF-8 -*-
############### built-in modules ###############
import pandas as pd
import numpy as np
import random
import json
import os
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt

# pytorch
import torch
from torch import nn
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
# from autogluon.tabular import TabularDataset, TabularPredictor

# import seaborn as sns
import pandas as pd
import numpy as np
import random
import json
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

############### sklearn ###############
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error

from tqdm import tqdm
import csv
from datetime import datetime

# pytorch
import torch
from torch import nn
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
############### pytorch ###############
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
############### sklearn ###############

from datasets import Dataset
import numpy as np
import pandas as pd
import warnings
import random
import copy
import time
import pickle
import os
import pickle
import uuid
import warnings
from typing import Optional
from datetime import datetime
from collections import OrderedDict
from tqdm import tqdm
############### pytorch ###############
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
############### sklearn ###############

from sklearn.metrics import r2_score, mean_squared_error
############### ray tune ###############
import ray
from ray import air, tune
from ray.air import session
# from ray.air.integrations.wandb import setup_wandb, WandbLoggerCallback
############### autogluon ###############
# from autogluon.multimodal import MultiModalPredictor
# from autogluon.tabular import TabularDataset, TabularPredictor
from ray.air.integrations.wandb import setup_wandb, WandbLoggerCallback
from tqdm import tqdm


warnings.filterwarnings('ignore')


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def range_data(temp_df, ts_range=[10, 3000], ys_range=[0, 3000], el_range=[5, 150]):
    """
    return value in ranges
    """
    temp_df = temp_df[(temp_df['Tensile_value']>=ts_range[0]) & (temp_df['Tensile_value']<=ts_range[1])]
    temp_df = temp_df[(temp_df['Yield_value']>=ys_range[0]) & (temp_df['Yield_value']<=ys_range[1])]
    temp_df = temp_df[(temp_df['Elongation_value']>=el_range[0]) & (temp_df['Elongation_value']<=el_range[1])]
    temp_df.reset_index(drop=True, inplace=True)
    return temp_df

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddinngs(text_list):

    encoded_input = tokenizer(
        text_list, padding='max_length', max_length = 512, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output).detach().cpu().numpy()[0].tolist()

def gen_text_embed(bert_df, col_embed='Text'):
    """
    return text embeddings
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
    
    encoded_input = tokenizer(
        text_list, padding='max_length', max_length = 512, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)

    return cls_pooling(model_output).detach().cpu().numpy()

def add_ele_embed(df):
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


def load_data(label='train_data', pred_prop='Yield_value', fes=['com', 'text_embed'],
              split_ratio=0.75, seed=42, perplexity=3):
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
    data_origin = pd.read_excel(f'./datasets/{label}.xlsx')    
    com_cols = set(data_origin.columns[17:-4])

    if label=='train_data':
        data_origin = range_data(data_origin, ts_range=[0, 3000], ys_range=[0, 3000], el_range=[5, 95])
        
        # filter available data
        filter = data_origin['status']==1
        data = data_origin[filter].copy()
        data.reset_index(drop=True, inplace=True)
    else:
        data = data_origin.copy()
    
    # fill composition
    for e_col in com_cols:
        data[e_col].fillna(0.0, inplace=True)  
    # data.iloc[:, 15:-2].fillna(0.0, inplace=True)
    data.fillna('', inplace=True)

    # drop some description cols    
    drop_cols = ['DOIs', 'Files', 'problem', 'status', 'Table_topic', 'title', 'abstract', 'Other_ele', \
                 'Text_addition', 'Tensile_name', 'Tensile_unit', 'Yield_name', 'Yield_unit', \
                 'Elongation_name', 'Elongation_unit', 'Material']
    data.drop(columns=drop_cols, inplace=True)   
    
    # processing text embeddings
    data = gen_text_embed(data)
    
    # composition selection
    # com_drop = ['H', 'F', 'Na', 'Mg', 'Cl', 'Ca', 'Zn', 'As', 'Zr', 'Y', 'Bi', 'Pb', 'Ta',
    #              'Ce', 'La', 'Sb', 'Zr', 'Fe', 'Sn']
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
            data_com_embed = pd.DataFrame(TSNE(n_components=3, learning_rate='auto',
                              init='random', perplexity=perplexity).fit_transform(data.iloc[:, -768:]), 
                                     columns=['tsne'+str(i) for i in range(3)])
            # data.drop(columns=['ele'+str(i) for i in range(768)], inplace=True)
            data = pd.concat([data, data_com_embed], axis=1)

    if label=='train_data':
        # data shuffle
        for _ in range(50):
            data = data.sample(frac=1.0, random_state=seed)
        # data split
        train_data, test_data = np.split(data.sample(frac=1, random_state=seed, ignore_index=True), [int(split_ratio*len(data))])
        train_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)

        return train_data, test_data    
    else:
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

    plt.subplot(121)
    plt.scatter(y_new_text, y_new_text_pred, s=point_size, color='#1F4B73')
    plt.plot(np.arange(int(max(y_new_text)*0.1), int(max(y_new_text)*1.1)), np.arange(int(max(y_new_text)*0.1),int(max(y_new_text)*1.1)), '-', color='#A2555A')
    plt.title(f"$R^2$={text_val[0]}, RMSE={text_val[1]}, MAE={text_val[2]}",
              fontdict={'family':'Times New Roman', 'size': 10})
    plt.ylabel('prediction', fontdict={'family':'Times New Roman', 'size': 14})
    plt.xlabel('True', fontdict={'family':'Times New Roman', 'size': 14})
    plt.xlim(0, int(max(y_new_text)*1.2))
    plt.ylim(0, int(max(y_new_text)*1.2))
    plt.legend(labels=[labels[0],'Y=X'])
    plt.grid()

    # on new exp data
    plt.subplot(122)
    # y_exp = best_exp_y
    # y_exp_pred = best_exp_preds.detach().cpu().numpy()
    # train_r2 = best_exp_r2

    plt.scatter(y_exp, y_exp_pred, s=point_size, color='#1F4B73')
    plt.plot(np.arange(int(max(y_exp)*0.1), int(max(y_exp)*1.1)), np.arange(int(max(y_exp)*0.1),int(max(y_exp)*1.1)), '-', color='#A2555A')
    plt.title(f"$R^2$={exp_val[0]}, RMSE={exp_val[1]}, MAE={exp_val[2]}",
              fontdict={'family':'Times New Roman', 'size': 10})
    plt.ylabel('prediction', fontdict={'family':'Times New Roman', 'size': 14})
    plt.xlabel('True', fontdict={'family':'Times New Roman', 'size': 14})
    plt.xlim(0,int(max(y_exp)*1.2))
    plt.ylim(0,int(max(y_exp)*1.2))
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

# class CustomFlatten(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fn = nn.Flatten()
    
#     def forward(self, x):
#         return self.fn(x)

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

# class CustomModel(nn.Module):
#     """
#     Multi-layer perceptron (MLP) model. Model class that warps PyTorch model.
#     Note: this class just for regression task.
#     """
#     def __init__(
#         self,
#         in_features: [int],
#         out_features: [int],
#         hidden_features: Optional [int],
#         num_layers: Optional [int] = 6,
#         activation: Optional [str] = 'relu',
#         dropout_prob: Optional[float] = 0.5,
#         normalization: Optional [str] = "batch_norm",
#         hidden_layer_list: Optional [int] = []
#     ):
#         super(CustomModel, self).__init__()
#         self.flatten = nn.Flatten()

#         layers = []
#         if len(hidden_layer_list) == 0:
#             for _ in range(num_layers):
#                 per_unit = Unit(
#                     normalization=normalization,
#                     in_features=in_features,
#                     out_features=hidden_features,
#                     activation=activation,
#                     dropout_prob=dropout_prob,
#                 )
#                 in_features = hidden_features
#                 layers.append(per_unit)
#             output_layer = Unit(
#                     normalization=normalization,
#                     in_features=in_features,
#                     out_features=out_features,
#                     activation=activation,
#                     dropout_prob=dropout_prob,
#             )
#             layers.append(output_layer)
#             self.layers = nn.Sequential(*layers)

#         else:
#             layer_list = [in_features] + hidden_layer_list + [out_features]
#             for i in range(len(layer_list)-1):
#                 per_unit = Unit(
#                     normalization="layer_norm",
#                     in_features=layer_list[i],
#                     out_features=layer_list[i+1],
#                     activation="relu",
#                     dropout_prob=0,
#                 )
#                 layers.append(per_unit)
#             self.layers = nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.layers(x)
#         return x
    
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
    ):
        
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
                padding=1
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
                padding=2
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

        # self.output_addition_layer = nn.Sequential(
        #     nn.LazyConv2d(out_channels=1, kernel_size=2, stride=1, padding=1), nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     # nn.LazyConv2d(out_channels=1, kernel_size=2, stride=1, padding=1), nn.ReLU(),
        #     # nn.MaxPool2d(kernel_size=2),
        #     nn.Flatten(),
        #     # nn.LazyLinear(4), nn.ReLU(),
        #     # nn.LazyLinear(32), nn.ReLU(),
        #     nn.LazyLinear(16), nn.ReLU(),
        #     # nn.LazyLinear(16), nn.ReLU(),
        #     # nn.LazyLinear(16), nn.ReLU(),

        # )

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
        # ele_output = self.com_layer(eles)
        # com_tsne_embed_output = self.com_tsne_embed_layer(com_tsne_embeds)
        # output = torch.concat([com_embed_output, text_embed_output, com_tsne_embed_output, ele_output], dim=1)

        inputs_addition = torch.concat([com_embeds.unsqueeze(1), text_embeds.unsqueeze(1), action_embeds.unsqueeze(1)], dim=1)
        com_dense_output = self.simple_layer(com_embeds)
        text_dense_output = self.simple_layer(text_embeds)
        act_dense_output = self.simple_layer(action_embeds)

        com_embed_output = self.seq_embed_layer(com_dense_output.unsqueeze(1))
        text_embed_output = self.seq_embed_layer(text_dense_output.unsqueeze(1))
        action_embed_output = self.seq_embed_layer(act_dense_output.unsqueeze(1))

        output_addition = self.common_embed_layer(inputs_addition.unsqueeze(1))

        output = torch.concat([output_addition, com_embed_output, text_embed_output, action_embed_output], dim=1)
        output = self.concat_layer(output)
        
        return output

def train_function(config):
    """
    training on ensemble model
    """

    prop = config["prop"]
    seed = config["seed"]
    split_ratio = config["split_ratio"]
    train_batch = config["train_batch"]
    epoch = config["epoch"]
    step = config["step"]
    gamma_ratio = config["gamma_ratio"]
    lr = config["lr"]

    # simple layer
    simple_layer_num_512 = config["simple_layer_num_512"]
    simple_layer_num_256 = config["simple_layer_num_256"]
    simple_layer_drop_prob = config["simple_layer_drop_prob"]

    # concat_layer_list
    concat_layer_num_64 = config["concat_layer_num_64"]
    concat_layer_num_32 = config["concat_layer_num_32"]
    concat_layer_drop_prob = config["concat_layer_drop_prob"]

    conv_channels_layer_1 = config["conv_channels_layer_1"]
    conv_channels_layer_2 = config["conv_channels_layer_2"]
    
    embed_layer_num_32 = config["embed_layer_num_32"]
    embed_layer_num_16 = config["embed_layer_num_16"]
    
    simple_layer_list = [512 for i in range(simple_layer_num_512)] + [256 for j in range(simple_layer_num_256)]
    concat_layer_list = [64 for i in range(concat_layer_num_64)] + [32 for j in range(concat_layer_num_32)] + [8, 4, 1]
    seq_embed_con1d_list = [conv_channels_layer_1, conv_channels_layer_2]
    seq_embed_con2d_list = [conv_channels_layer_1, conv_channels_layer_2]
    seq_embed_fc_list = [32 for i in range(embed_layer_num_32)] + [16 for j in range(embed_layer_num_16)]
    seq_embed_2d_fc_list = [32 for i in range(embed_layer_num_32)] + [16 for j in range(embed_layer_num_16)]


    #  setting runs ID
    paras_li = ['prop', 'seed', 'split_ratio', 'train_batch',
            'epoch', 'lr', 'step', 'gamma_ratio', 'simple_layer_num_512',
            'simple_layer_num_256', 'simple_layer_drop_prob', 'concat_layer_num_64', 
            'concat_layer_num_32', 'concat_layer_drop_prob', 'conv_channels_layer_1',
            'conv_channels_layer_2', 'embed_layer_num_32', 'embed_layer_num_16']
    paras_string = [str(eval(s)) + '_' for s in paras_li]
    fes=['com', 'com_embed', 'text_embed', 'action_embed']
    set_global_seed(seed=seed)
    # paras_string = f"{prop}_{split_ratio}_{train_batch}_{epoch}_{lr}_{step}_{gamma_ratio}__{seed}"

    # ###################  setting parameters  ###################
    # # add parameters names in output files
    # with open('./outputs/reg_model.csv', 'a+') as csvfile:  
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(['prop', 'train_r2', 'best_val_r2', 'best_new_text_r2','best_exp_r2'] + paras_li[1:])
    
    ###################  build dataset class  ###################
    train_data, test_data = load_data(label='train_data', pred_prop=prop,
                            fes=fes, split_ratio=split_ratio, seed=seed, perplexity=3)
    new_text_data = load_data(label='text_test', pred_prop=prop, fes=fes, perplexity=3)
    exp_data = load_data(label='exp_test', pred_prop=prop, fes=fes, perplexity=3)
    print(f"train_data.shape:{train_data.shape}, test_data.shape:{test_data.shape}")

    # if prop != 'Elongation_value':
    #     train_data[prop] = train_data[prop] / 100
    #     test_data[prop] = test_data[prop] / 100
    #     new_text_data[prop] = new_text_data[prop] / 100
    #     exp_data[prop] = exp_data[prop] / 100
        
    # train_data[predict_label] = train_data[predict_label]  / train_data[predict_label].abs().max() 
    # test_data[predict_label] = test_data[predict_label]  / test_data[predict_label].abs().max()

    # train_data[prop]= np.log(train_data[prop] + 1)
    # test_data[prop]= np.log(test_data[prop] + 1)
    # new_text_data[prop] = np.log(new_text_data[prop] + 1)
    # exp_data[prop]= np.log(exp_data[prop] + 1)

    new_text_dataloader = DataLoader(CustomSimpleDataset(**gen_data_class(new_text_data)),
                    batch_size=len(new_text_data))
    exp_dataloader = DataLoader(CustomSimpleDataset(**gen_data_class(exp_data)),
                    batch_size=len(exp_data))
    all_train_dataloader = DataLoader(CustomSimpleDataset(**gen_data_class(train_data)),
                    batch_size=len(train_data), shuffle=False,drop_last=False)
    train_dataloader = DataLoader(CustomSimpleDataset(**gen_data_class(train_data)),
                    batch_size=train_batch, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(CustomSimpleDataset(**gen_data_class(test_data)),
                    batch_size=len(test_data))

    ###################  start training  ###################
    ## TODO: limit total saved models
    reg_model = CustomSimpleModel(
        simple_layer_list = simple_layer_list,
        concat_layer_list = concat_layer_list,
        seq_embed_con1d_list = seq_embed_con1d_list,
        seq_embed_fc_list = seq_embed_fc_list,
        seq_embed_con2d_list = seq_embed_con2d_list,
        seq_embed_2d_fc_list = seq_embed_2d_fc_list,
        simple_layer_drop_prob = simple_layer_drop_prob,
        concat_layer_drop_prob = concat_layer_drop_prob,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(reg_model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma_ratio)
    # optimizer = torch.optim.LBFGS(params=reg_model.parameters(), lr=lr)
    # optimizer = torch.optim.AdamW(reg_model.parameters(), lr=1e-54)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=50)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=lr,                                              
    #                                                end_factor=1e-4, total_iters=0.8*epoch)

    dt_string = datetime.now().strftime("%d_%m_%H.%M.%S")
    writer = SummaryWriter('./outputs/runs', flush_secs=20)

    best_val_r2 = -1e5
    best_new_text_r2 = -1e5
    best_exp_r2 = -1e5
    for epoch_num in tqdm(range(epoch)):
        # print(f"\n Epoch {epoch_num+1}\n----------------------------------")
        reg_model.train()
        for batch, inputs in enumerate(train_dataloader):
            
            y = inputs['labels'].unsqueeze(1)
            preds = reg_model(**inputs)
            train_loss = loss_fn(preds, y)
            train_r2 = eval_model(y, preds)['r2']

            # Backpropagation
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if batch % 4 == 0:
            #     train_loss, current = train_loss.item(), (batch + 1) * len(y)
            #     print(f"Train batch loss: {train_loss:>7f}  [{current:>5d}/{size:>5d}]")
        scheduler.step()
        # print(f"Train avg r2: {train_r2}, train_loss {train_loss}")

        reg_model.eval()
        with torch.no_grad():
            for batch, inputs in enumerate(val_dataloader):
                y = inputs['labels'].unsqueeze(1)
                preds = reg_model(**inputs)
                val_loss = loss_fn(preds, y).item()
                val_r2 = eval_model(y, preds)['r2']

            # text and exp performance on each epoch
            for batch, inputs in enumerate(new_text_dataloader):
                y = inputs['labels'].unsqueeze(1)
                preds = reg_model(**inputs)
                new_text_r2 = eval_model(y, preds)['r2']

                ########## best_new_text_r2
                if new_text_r2 > best_new_text_r2:
                    best_new_text_r2 = new_text_r2
                    plot_best_text_y = y.detach().cpu().numpy()
                    plot_best_text_preds = preds.detach().cpu().numpy()
                    plot_best_text_result = eval_model(plot_best_text_y, plot_best_text_preds)

            for batch, inputs in enumerate(exp_dataloader):
                y = inputs['labels'].unsqueeze(1)
                preds = reg_model(**inputs)
                exp_r2 = eval_model(y, preds)['r2']
                ########## best_new_text_r2
                if exp_r2 > best_exp_r2:
                    best_exp_r2 = exp_r2
                    plot_best_exp_y = y.detach().cpu().numpy()
                    plot_best_exp_preds = preds.detach().cpu().numpy()
                    plot_best_exp_result = eval_model(plot_best_exp_y, plot_best_exp_preds)

        # plot on tensorboard
        writer.add_scalars('Reg_'+dt_string+'/Loss',
                        tag_scalar_dict = {'train_loss':train_loss,
                                            'val_loss':val_loss},
                        global_step = epoch_num+1)
        writer.add_scalars('Reg_'+dt_string+'/R2',
                            tag_scalar_dict = {'train_r2':train_r2,
                                            'val_r2':val_r2},  # 'test_r2':test_r2},
                            global_step = epoch_num+1)

        writer.add_scalars('Reg_'+dt_string+'/Test_R2',
                        tag_scalar_dict = {'new_text_r2':new_text_r2,
                                            'exp_r2':exp_r2},
                        global_step = epoch_num+1)
        
       

    writer.close()

    ######################## test data validation  ###################
    # best_model = CustomSimpleModel().to(device)
    # best_model.load_state_dict(torch.load(f"./outputs/reg_model_saved/{paras_string}.pth"))
    best_model = torch.load(f"./outputs/reg_model_saved/{paras_string}.pt")

    best_model.eval()
    with torch.no_grad():
        # validate model performance
        for batch, inputs in enumerate(all_train_dataloader):
            y_train = inputs['labels'].unsqueeze(1).detach().cpu().numpy()
            y_train_preds = best_model(**inputs).detach().cpu().numpy()
            train_result = eval_model(y_train, y_train_preds)

        for batch, inputs in enumerate(val_dataloader):
            y_val = inputs['labels'].unsqueeze(1).detach().cpu().numpy()
            y_val_preds = best_model(**inputs).detach().cpu().numpy()
            val_result = eval_model(y_val, y_val_preds)

        for batch, inputs in enumerate(new_text_dataloader):
            best_new_text_y = inputs['labels'].unsqueeze(1)
            best_new_text_preds = best_model(**inputs)
            best_new_text_result = eval_model(best_new_text_y, best_new_text_preds)
            
        for batch, inputs in enumerate(exp_dataloader):
            best_exp_y = inputs['labels'].unsqueeze(1)
            best_exp_preds = best_model(**inputs)
            best_exp_result = eval_model(best_exp_y, best_exp_preds)

    # # save recent literature and experiment True-Predict figs
    plot_test_data(y_train, y_train_preds, train_result,
                y_val, y_val_preds, val_result,
                fig_name=f"./outputs/figs/train_{paras_string}.png",
                labels=["Model train data", "Model test data"], point_size=5)
    
    plot_test_data(plot_best_text_y, plot_best_text_preds, plot_best_text_result,
                plot_best_exp_y, plot_best_exp_preds, plot_best_exp_result,
                fig_name=f"./outputs/figs/test_{paras_string}.png",
                labels=["New literature data", "Experiment data"], point_size=15)

    # plot_test_data(best_new_text_y, best_new_text_preds, best_new_text_result,
    #             best_exp_y, best_exp_preds, best_exp_result,
    #             fig_name=f"./outputs/figs/test_{paras_string}.png",
    #             labels=["New literature data", "Experiment data"], point_size=15)

    print(y_train.shape)
    y_train = y_train.T.tolist()[0]
    y_train_preds = y_train_preds.T.tolist()[0]

    y_val = y_val.T.tolist()[0]
    y_val_preds = y_val_preds.T.tolist()[0]

    plot_best_text_y = plot_best_text_y.T.tolist()[0]
    plot_best_text_preds = plot_best_text_preds.T.tolist()[0]

    plot_best_exp_y = plot_best_exp_y.T.tolist()[0]
    plot_best_exp_preds = plot_best_exp_preds.T.tolist()[0]

    max_len = len(y_train)

    y_val += [6666.0 for i in range(max_len-len(y_val))]
    y_val_preds += [6666.0 for i in range(max_len-len(y_val_preds))]
    plot_best_text_y += [6666.0 for i in range(max_len-len(plot_best_text_y))]
    plot_best_text_preds += [6666.0 for i in range(max_len-len(plot_best_text_preds))]
    plot_best_exp_y += [6666.0 for i in range(max_len-len(plot_best_exp_y))]
    plot_best_exp_preds += [6666.0 for i in range(max_len-len(plot_best_exp_preds))]

    plot_result = pd.DataFrame({
        'y_train': y_train,
        'y_train_preds': y_train_preds,
        'y_val': y_val,
        'y_val_preds': y_val_preds,
        'text_y': plot_best_text_y,
        'text_preds': plot_best_text_preds,
        'exp_y': plot_best_exp_y,
        'exp_preds': plot_best_exp_preds
    })
    plot_result.to_excel(f"./outputs/preds/preds_{paras_string}.xlsx", index=None)

    ######################## save result  ###################
    with open('./outputs/reg_model.csv', 'a+') as csvfile:  
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            prop, train_result['r2'], val_result['r2'], best_new_text_result['r2'],
            best_exp_result['r2'], seed, split_ratio, train_batch, epoch, lr, step,
            gamma_ratio, simple_layer_num_512, simple_layer_num_256, simple_layer_drop_prob,
            concat_layer_num_64, concat_layer_num_32, concat_layer_drop_prob,
            conv_channels_layer_1, conv_channels_layer_2, embed_layer_num_32,
            embed_layer_num_16
        ])


    session.report({
        "train_r2": train_result['r2'],
        "train_rmse": train_result['rmse'],
        "train_mae": train_result['mae'],

        "val_r2": val_result['r2'],
        "val_rmse": val_result['rmse'],
        "val_mae": val_result['mae'],

        "new_text_r2": best_new_text_result['r2'],
        "new_text_rmse": best_new_text_result['rmse'],
        "new_text_mae": best_new_text_result['mae'],

        "exp_r2": best_exp_result['r2'],
        "exp_rmse": best_exp_result['rmse'],
        "exp_mae": best_exp_result['mae']
    })

def tune_with_callback():
    """
    Tune hyper-parameters on multimodal model.
    """
    tuner = tune.Tuner(
        train_function,
        tune_config=tune.TuneConfig(
            metric="val_r2",
            mode="max",
            num_samples=1,
            max_concurrent_trials=1,
            # search_alg=OptunaSearch(metric="agg_second_r2", mode="max"),
        ),
        run_config=air.RunConfig(
            name="tune_reg",
            local_dir="./outputs/logs",
            callbacks=[
                WandbLoggerCallback(
                    project="reg_model",
                    mode="offline",
                    tags=["1010"])
            ]
        ),
        param_space={
            'prop': tune.choice(['Tensile_value', 'Yield_value', 'Elongation_value']),
            'seed': tune.choice([42, 789, 666, 888]),
            'split_ratio': tune.choice([0.75, 0.7]), 
            'train_batch': tune.choice([32]),
            'epoch': tune.choice([300]),
            'lr': tune.choice([0.01]),
            'step': tune.choice([80]),
            'gamma_ratio': tune.choice([0.5, 0.1, 0.8]),
            'simple_layer_num_512': tune.choice([2, 3, 4, 5]),
            'simple_layer_num_256': tune.choice([2, 3]),
            'simple_layer_drop_prob': tune.choice([0.05, 0.0, 0.1]),
            'concat_layer_num_64': tune.choice([2, 3, 4, 5]),
            'concat_layer_num_32': tune.choice([2, 3]),
            'concat_layer_drop_prob': tune.choice([0.05, 0.0, 0.1]), 
            'conv_channels_layer_1': tune.choice([1, 2]),
            'conv_channels_layer_2': tune.choice([1, 2]),
            'embed_layer_num_32': tune.choice([2, 3, 4, 5]),
            'embed_layer_num_16': tune.choice([2, 3, 4]),
        }
    )
    tuner.fit()

    
if __name__=='__main__':
    # ###################  load pretrained model  ###################
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # device = 'cpu'
    # print(f"You are using '{device}' device!")

    # # model_name = 'bert-base-uncased'
    # # model_name = 'microsoft/deberta-v3-base'
    # # model_name = 'm3rg-iitd/matscibert'
    # model_name = './../model_saved/checkpoint-140000'
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name).to(device)

    # #  setting runs ID
    # outputs_cols = ['prop', 'train_r2', 'best_val_r2', 'best_new_text_r2','best_exp_r2', 
    #             'seed', 'split_ratio', 'train_batch', 'epoch', 'lr', 'step', 
    #             'gamma_ratio', 'simple_layer_num_512', 'simple_layer_num_256', 
    #             'simple_layer_drop_prob', 'concat_layer_num_64', 'concat_layer_num_32', 
    #             'concat_layer_drop_prob', 'conv_channels_layer_1', 'conv_channels_layer_2',
    #             'embed_layer_num_32', 'embed_layer_num_16']
    # with open('./outputs/reg_model.csv', 'a+') as csvfile:  
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(outputs_cols)
    # tune_with_callback()
    

    ###################  load pretrained model  ###################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"You are using '{device}' device!")
    # model_name = 'bert-base-uncased'
    # model_name = 'microsoft/deberta-v3-base'
    # model_name = 'm3rg-iitd/matscibert'
    model_name = './../model_saved/checkpoint-140000'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    ###################  setting parameters  ###################
    # for prop in ['Yield_value', 'Tensile_value', 'Elongation_value']:
    for prop in ['Elongation_value']:
        # add parameters names in output files
        with open('./outputs/reg_model.csv', 'a+') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['prop', 'train_r2', 'best_val_r2', 'best_new_text_r2','best_exp_r2',
                'seed', 'split_ratio', 'perplexity', 'train_batch', 'epoch', 'lr', 'step', 'gamma_ratio'])
            

            
        # for simple_layer_list in [[1024, 512, 512, 512, 512], [768, 768, 768, 768], [256, 256, 256, 256, 128], [512, 512, 512, 256]]:
        #     for concat_layer_list in  [[128, 64, 64, 16, 32, 8, 4, 1], [128, 64, 32, 8, 4, 1], [128, 128, 64, 16, 32, 8, 4, 1]]:
        #         for seq_embed_con1d_list in [[1, 1], [1, 2, 1], [1, 1, 1], [2, 2, 1, 1]]:
        #             for layer_dorp in [0, 0.1, 0.2]:

        for simple_layer_list in [[512, 512, 512, 256]]:
            for concat_layer_list in  [[64, 64, 32, 8, 4, 1]]:
                for seq_embed_con1d_list in [[1, 1]]:

                    for split_ratio in [0.75]:
                        for step in [80]:
                            for layer_dorp in [0]:
                                for seed in [42]:
                                    
                                    # prop = 'Yield_value'
                                    # seed = 42
                                    # split_ratio = 0.75
                                    # perplexity = 3

                                    train_batch = 32
                                    perplexity = 3
                                    epoch = 250
                                    lr = 0.01
                                    # step = 100
                                    gamma_ratio = 0.5
                                    
                                    set_global_seed(seed=seed)
                                    fes=['com', 'com_embed', 'text_embed', 'action_embed']
                                    paras_string = f"{prop}_{split_ratio}_{train_batch}_{epoch}_{lr}_{step}_{gamma_ratio}__{seed}"
                            
                                    ###################  build dataset class  ###################
                                    train_data, test_data = load_data(label='train_data', pred_prop=prop,
                                                            fes=fes, split_ratio=split_ratio, seed=seed, perplexity=perplexity)
                                    new_text_data = load_data(label='text_test', pred_prop=prop, fes=fes, perplexity=perplexity)
                                    exp_data = load_data(label='exp_test', pred_prop=prop, fes=fes, perplexity=perplexity)
                                    print(f"train_data.shape:{train_data.shape}, test_data.shape:{test_data.shape}")

                                    # if prop != 'Elongation_value':
                                    #     train_data[prop] = train_data[prop] / 100
                                    #     test_data[prop] = test_data[prop] / 100
                                    #     new_text_data[prop] = new_text_data[prop] / 100
                                    #     exp_data[prop] = exp_data[prop] / 100
                                        
                                    # train_data[predict_label] = train_data[predict_label]  / train_data[predict_label].abs().max() 
                                    # test_data[predict_label] = test_data[predict_label]  / test_data[predict_label].abs().max()

                                    # train_data[prop]= np.log(train_data[prop] + 1)
                                    # test_data[prop]= np.log(test_data[prop] + 1)
                                    # new_text_data[prop] = np.log(new_text_data[prop] + 1)
                                    # exp_data[prop]= np.log(exp_data[prop] + 1)

                                    new_text_dataloader = DataLoader(CustomSimpleDataset(**gen_data_class(new_text_data)),
                                                    batch_size=len(new_text_data))
                                    exp_dataloader = DataLoader(CustomSimpleDataset(**gen_data_class(exp_data)),
                                                    batch_size=len(exp_data))
                                    all_train_dataloader = DataLoader(CustomSimpleDataset(**gen_data_class(train_data)),
                                                    batch_size=len(train_data), shuffle=False,drop_last=False)
                                    train_dataloader = DataLoader(CustomSimpleDataset(**gen_data_class(train_data)),
                                                    batch_size=train_batch, shuffle=True, drop_last=False)
                                    val_dataloader = DataLoader(CustomSimpleDataset(**gen_data_class(test_data)),
                                                    batch_size=len(test_data))

                                    ###################  start training  ###################
                                    ## TODO: limit total saved models
                                    reg_model = CustomSimpleModel(
                                        # simple_layer_list = [512, 512, 512, 512, 512],
                                        # concat_layer_list = [64, 64, 64, 16, 32, 8, 4, 1],
                                        # seq_embed_con1d_list = [1, 1],

                                        simple_layer_list = simple_layer_list,
                                        concat_layer_list = concat_layer_list,
                                        seq_embed_con1d_list = seq_embed_con1d_list,

                                        seq_embed_fc_list = [32, 16],
                                        seq_embed_con2d_list = [1, 1],
                                        seq_embed_2d_fc_list = [32, 16, 16],
                                        simple_layer_drop_prob=layer_dorp,
                                        concat_layer_drop_prob=layer_dorp,
                                    ).to(device)

                                    loss_fn = nn.MSELoss()
                                    optimizer = torch.optim.AdamW(reg_model.parameters(), lr=lr, weight_decay=0.01)
                                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma_ratio)
                                    # optimizer = torch.optim.LBFGS(params=reg_model.parameters(), lr=lr)
                                    # optimizer = torch.optim.AdamW(reg_model.parameters(), lr=1e-54)
                                    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=50)
                                    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=lr,                                              
                                    #                                                end_factor=1e-4, total_iters=0.8*epoch)

                                    dt_string = datetime.now().strftime("%d_%m_%H.%M.%S")
                                    writer = SummaryWriter('./outputs/runs', flush_secs=20)

                                    best_val_r2 = -1e5
                                    best_new_text_r2 = -1e5
                                    best_exp_r2 = -1e5
                                    for epoch_num in tqdm(range(epoch)):
                                        # print(f"\n Epoch {epoch_num+1}\n----------------------------------")
                                        reg_model.train()
                                        for batch, inputs in enumerate(train_dataloader):
                                            
                                            y = inputs['labels'].unsqueeze(1)
                                            preds = reg_model(**inputs)
                                            train_loss = loss_fn(preds, y)
                                            train_r2 = eval_model(y, preds)['r2']

                                            # Backpropagation
                                            train_loss.backward()
                                            optimizer.step()
                                            optimizer.zero_grad()
                                            # if batch % 4 == 0:
                                            #     train_loss, current = train_loss.item(), (batch + 1) * len(y)
                                            #     print(f"Train batch loss: {train_loss:>7f}  [{current:>5d}/{size:>5d}]")
                                        scheduler.step()
                                        # print(f"Train avg r2: {train_r2}, train_loss {train_loss}")

                                        reg_model.eval()
                                        with torch.no_grad():
                                            for batch, inputs in enumerate(val_dataloader):
                                                y = inputs['labels'].unsqueeze(1)
                                                preds = reg_model(**inputs)
                                                val_loss = loss_fn(preds, y).item()
                                                val_r2 = eval_model(y, preds)['r2']

                                            # text and exp performance on each epoch
                                            for batch, inputs in enumerate(new_text_dataloader):
                                                y = inputs['labels'].unsqueeze(1)
                                                preds = reg_model(**inputs)
                                                new_text_r2 = eval_model(y, preds)['r2']

                                                ########## best_new_text_r2
                                                if new_text_r2 > best_new_text_r2:
                                                    best_new_text_r2 = new_text_r2
                                                    plot_best_text_y = y.detach().cpu().numpy()
                                                    plot_best_text_preds = preds.detach().cpu().numpy()
                                                    plot_best_text_result = eval_model(plot_best_text_y, plot_best_text_preds)

                                            for batch, inputs in enumerate(exp_dataloader):
                                                y = inputs['labels'].unsqueeze(1)
                                                preds = reg_model(**inputs)
                                                exp_r2 = eval_model(y, preds)['r2']
                                                ########## best_new_text_r2
                                                if exp_r2 > best_exp_r2:
                                                    best_exp_r2 = exp_r2
                                                    plot_best_exp_y = y.detach().cpu().numpy()
                                                    plot_best_exp_preds = preds.detach().cpu().numpy()
                                                    plot_best_exp_result = eval_model(plot_best_exp_y, plot_best_exp_preds)

                                        # plot on tensorboard
                                        writer.add_scalars('Reg_'+dt_string+'/Loss',
                                                        tag_scalar_dict = {'train_loss':train_loss,
                                                                            'val_loss':val_loss},
                                                        global_step = epoch_num+1)
                                        writer.add_scalars('Reg_'+dt_string+'/R2',
                                                            tag_scalar_dict = {'train_r2':train_r2,
                                                                            'val_r2':val_r2},  # 'test_r2':test_r2},
                                                            global_step = epoch_num+1)

                                        writer.add_scalars('Reg_'+dt_string+'/Test_R2',
                                                        tag_scalar_dict = {'new_text_r2':new_text_r2,
                                                                            'exp_r2':exp_r2},
                                                        global_step = epoch_num+1)
                                        
                                        # save best model for one group of parameters
                                        # sum_r2 = 0.2*new_text_r2 + 0.8*val_r2
                                        if val_r2 > best_val_r2:
                                            best_val_r2 = val_r2
                                            torch.save(reg_model, f"./outputs/reg_model_saved/{paras_string}.pt")

                                    writer.close()

                                    ######################## test data validation  ###################
                                    # best_model = CustomSimpleModel().to(device)
                                    # best_model.load_state_dict(torch.load(f"./outputs/reg_model_saved/{paras_string}.pth"))
                                    best_model = torch.load(f"./outputs/reg_model_saved/{paras_string}.pt")

                                    best_model.eval()
                                    with torch.no_grad():
                                        # validate model performance
                                        for batch, inputs in enumerate(all_train_dataloader):
                                            y_train = inputs['labels'].unsqueeze(1).detach().cpu().numpy()
                                            y_train_preds = best_model(**inputs).detach().cpu().numpy()
                                            train_result = eval_model(y_train, y_train_preds)

                                        for batch, inputs in enumerate(val_dataloader):
                                            y_val = inputs['labels'].unsqueeze(1).detach().cpu().numpy()
                                            y_val_preds = best_model(**inputs).detach().cpu().numpy()
                                            val_result = eval_model(y_val, y_val_preds)

                                        for batch, inputs in enumerate(new_text_dataloader):
                                            best_new_text_y = inputs['labels'].unsqueeze(1)
                                            best_new_text_preds = best_model(**inputs)
                                            best_new_text_result = eval_model(best_new_text_y, best_new_text_preds)
                                            
                                        for batch, inputs in enumerate(exp_dataloader):
                                            best_exp_y = inputs['labels'].unsqueeze(1)
                                            best_exp_preds = best_model(**inputs)
                                            best_exp_result = eval_model(best_exp_y, best_exp_preds)

                                    # # save recent literature and experiment True-Predict figs
                                    plot_test_data(y_train, y_train_preds, train_result,
                                                y_val, y_val_preds, val_result,
                                                fig_name=f"./outputs/figs/train_{paras_string}.png",
                                                labels=["Model train data", "Model test data"], point_size=5)
                                    
                                    plot_test_data(plot_best_text_y, plot_best_text_preds, plot_best_text_result,
                                                plot_best_exp_y, plot_best_exp_preds, plot_best_exp_result,
                                                fig_name=f"./outputs/figs/test_{paras_string}.png",
                                                labels=["New literature data", "Experiment data"], point_size=15)
                                    
                                    print('**********', train_result, val_result, plot_best_text_result, plot_best_exp_result)

                                    # plot_test_data(best_new_text_y, best_new_text_preds, best_new_text_result,
                                    #             best_exp_y, best_exp_preds, best_exp_result,
                                    #             fig_name=f"./outputs/figs/test_{paras_string}.png",
                                    #             labels=["New literature data", "Experiment data"], point_size=15)

                                    print(y_train.shape)
                                    y_train = y_train.T.tolist()[0]
                                    y_train_preds = y_train_preds.T.tolist()[0]

                                    y_val = y_val.T.tolist()[0]
                                    y_val_preds = y_val_preds.T.tolist()[0]

                                    plot_best_text_y = plot_best_text_y.T.tolist()[0]
                                    plot_best_text_preds = plot_best_text_preds.T.tolist()[0]

                                    plot_best_exp_y = plot_best_exp_y.T.tolist()[0]
                                    plot_best_exp_preds = plot_best_exp_preds.T.tolist()[0]

                                    max_len = len(y_train)

                                    y_val += [6666.0 for i in range(max_len-len(y_val))]
                                    y_val_preds += [6666.0 for i in range(max_len-len(y_val_preds))]
                                    plot_best_text_y += [6666.0 for i in range(max_len-len(plot_best_text_y))]
                                    plot_best_text_preds += [6666.0 for i in range(max_len-len(plot_best_text_preds))]
                                    plot_best_exp_y += [6666.0 for i in range(max_len-len(plot_best_exp_y))]
                                    plot_best_exp_preds += [6666.0 for i in range(max_len-len(plot_best_exp_preds))]

                                    plot_result = pd.DataFrame({
                                        'y_train': y_train,
                                        'y_train_preds': y_train_preds,
                                        'y_val': y_val,
                                        'y_val_preds': y_val_preds,
                                        'text_y': plot_best_text_y,
                                        'text_preds': plot_best_text_preds,
                                        'exp_y': plot_best_exp_y,
                                        'exp_preds': plot_best_exp_preds
                                    })
                                    plot_result.to_excel(f"./outputs/preds/preds_{paras_string}.xlsx", index=None)

                                    ######################## save result  ###################
                                    with open('./outputs/reg_model.csv', 'a+') as csvfile:  
                                        csvwriter = csv.writer(csvfile)
                                        csvwriter.writerow([prop, train_result['r2'], val_result['r2'], best_new_text_result['r2'],
                                            best_exp_result['r2'], seed, split_ratio, perplexity, train_batch, epoch, lr, step, gamma_ratio])
