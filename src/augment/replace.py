import random
import jieba
import pandas as pd
from src.constants import labels_to_ids

dfb = pd.read_csv('./data/bank.csv')
dfb = dfb.stack()

def rep_bank(w, labels, p):
    # print(w,labels)
    if(labels[0] == labels_to_ids['B-BANK']):
        out =  (dfb.sample(n=1).values[0]) if random.random() < p else w
        print(w,labels)
        print(f'replace {w} to {out}')
        return out

    return w

def replace(text, labels,p=0.1, replaceFn=None):
    nt=[]
    nl=[]
    out = jieba.tokenize(text)
    cut = []

    for o in out:
        word, s, e = o
        cut += [o]
        append =  replaceFn(word,labels[s:e],p) if (replaceFn) else f"<{word}>" if random.random() < p else word
        nt += append
        nl += labels[s:e]
        nl += labels[e-1:e]*(len(append)-len(word))

    
    # print(text,list(zip(nt,nl)),len(nt), len(nl))

    return nt, nl, cut

