import logging
import torch
import pandas as pd
import re
from src.config import config
from src.constants import LABELS, NUM_LABLES,ids_to_labels,MAX_SEQ_LENGTH, SOS_IDS, EOS_IDS, PAD_IDS,labels_to_ids
from transformers import BertModel, BertForTokenClassification, BertTokenizerFast

from src.utils import BIO_to_inner

tokenizer = BertTokenizerFast.from_pretrained(config.pretrained, cache_dir='./cache')

with open('vocab.txt', 'r') as file:
    vocab = [line.strip() for line in file.readlines()]
tokenizer.add_tokens(vocab)


def align_label(tokens, labels, origin_text):
    # tokens = tokenizer.convert_tokens_to_string(tokens).lower().split()
    # labels = labels[origin_text != ' ']
    labels = [label for label, text in zip(labels, origin_text) if text != ' ']

    p = 0
    out = []
    
    for token in tokens:
        token = token.strip('##')
        # q = origin_text.find(token, q)
        # print(token,out, p, labels[p], origin_text[p])
        # print(len(token))

        out.append(labels[p])
        p += 1 if token == "[UNK]" else len(token)


    return out

def realign(labels, tokens, origin_text):
    text = ''.join(tokenizer.convert_tokens_to_string(tokens).split())
    text = text.replace('[UNK]', 'ø').lower()
    origin_text = origin_text.lower()
    
    aligned_labels = []
    # print(111,origin_text, text, tokens)
    
    for token in tokens:
        label = labels.pop(0)
        if token.startswith('##'):
            label = BIO_to_inner(aligned_labels[-1])
            aligned_labels.extend([label]*len(token.strip("##")))
        else:
            aligned_labels.append(label)
            aligned_labels.extend([BIO_to_inner(label)]*(len(token)-1))
        # print(label, token)

    return_labels = []

    i=0
    j=0
    while i<len(text):
        t = text[i]
        ot = origin_text[j]
        if(t =='ø'):
            i+=1
            return_labels.append('O')
        elif(ot == t):
            return_labels.append( aligned_labels[i])
            i+=1
        else:
            return_labels.append('O')
        j+=1

    return return_labels

def tokenize_sentence(sentence, labels=None, align=True,tokenizer=tokenizer, max_seq_len=MAX_SEQ_LENGTH):
    tokenized_data = tokenizer.encode_plus(
        sentence,
        return_length=True,
        truncation=True,
        padding='max_length',
        max_length=max_seq_len,
        return_tensors='pt',
        )

    ids = tokenized_data['input_ids']
    ids_clean = ids[ids!=0][1:-1]

    tokens = tokenizer.convert_ids_to_tokens(ids_clean.tolist())
    # print(111,tokens)

    if(not labels): return tokenized_data

    # -2 for [CLS] and [SEP]
    tokenized_labels = labels
    if(align):
        tokenized_labels = align_label(tokens, labels, sentence)

    # print(11,tokenized_labels)

    if(type(tokenized_labels[0])=='str'): tokenized_labels = [labels_to_ids[i] for i in tokenized_labels]

    tokenized_labels = [SOS_IDS] + tokenized_labels + [EOS_IDS]
    tokenized_labels += [PAD_IDS] * (
        tokenized_data['input_ids'].numel() - len(tokenized_labels))

    # print(tokenized_labels)
    tokenized_labels = torch.tensor(tokenized_labels)

    return tokenized_data, tokenized_labels
