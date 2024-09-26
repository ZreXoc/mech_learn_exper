import numpy as np
import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, dataset
from torch.utils.data.sampler import T_co
from src.dataset import CommentDataset
from src.tokenizer import tokenize_and_align_labels, tokenizer
from src.constants import ids_to_labels

from transformers import BertTokenizerFast

MAX_LEN = 200

SPLITS = {
    'train': os.path.join('data', 'train_data_public.csv'),
    'test': os.path.join('data', 'test_public.csv'),
}

LABELS =  ["B-BANK", "I-BANK", "B-PRODUCT", "I-PRODUCT", 'B-COMMENTS_N', 'I-COMMENTS_N', 'B-COMMENTS_ADJ', 'I-COMMENTS_ADJ', 'O']

df = pd.read_csv(SPLITS['train']).head()

def max_length(df):

    maxi = -1
    maxlen=-1
    for idx, row in df.iterrows():
        l = (len(row['text']))
        if(l>maxlen):
            maxlen = l
            maxi = idx

    print(maxlen,maxi)

# dataset = CommentDataset(SPLITS['train'])

# data, label = dataset.__getitem__(1)

# print(data)
# input_ids = data['input_ids']
# mask = data['attention_mask']

# print(input_ids)
# print(mask)
# print(label)

# ids_clean = input_ids[mask==1]
# label_clean = label[mask==1]
# print(ids_clean,ids_clean.size())
# print(label_clean,label_clean.size())
# print([ids_to_labels[i.item()] for i in label_clean])
# print(tokenizer.convert_ids_to_tokens(ids_clean))
# print(dataset.labels[1])

# label_all_tokens = False
# df['text'] = df['text'].map(lambda x: tokenizer.encode(x, padding='max_length', max_length=MAX_LEN, truncation=True, return_tensors='pt'))
# print(df['text'])
# print(len(df['text'][1]))
# print(len(df['BIO_anno'][1]))
# df['text'] = df['text'].map(lambda x: tokenizer.encode(x, padding='max_length', max_length=MAX_LEN, truncation=True, return_tensors='pt'))
# print(df['text'][2])
# df['text'] = df['text'].map(lambda x: tokenizer.decode(x[0]))
# print(df['text'][2])
# print(len(df['text'][3]))
# label_all_tokens = False
# code = tokenizer('你好再见！！！。Geir good bye!!!hi. hello.','123再见', return_length=True, return_tensors='pt')
# print(code)
# print(tokenizer.convert_ids_to_tokens(code['input_ids']))
# code = tokenizer('你好再见！！！。Geir good bye!!!hi. hello.',is_split_into_words=False)
# print(code)
# print(tokenizer.convert_ids_to_tokens(code['input_ids']))

