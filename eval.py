import logging
import torch
import os
from transformers import BertForSequenceClassification, BertForTokenClassification
import pandas as pd
from src.constants import MAX_SEQ_LENGTH,  NUM_LABLES, SPLITS, ids_to_labels
from src.config import config, out_dir
from tqdm import tqdm

from src.model import NER_Model
from src.tokenizer2 import tokenizer, tokenize_sentence, realign

use_cuda = config.cuda
device = torch.device("cuda" if use_cuda else "cpu")

tqdm.pandas()

df = pd.read_csv(SPLITS['test'])
# df = df.head()

tokenized_data = []

# model_path1='./out/remote/10212228-train/t1-e16.pt'
model_path1='./out/remote/1101609-train/t1-best.pth'
model_path2='./out/remote/11081410/t2-e12.pt'

state_dict_task1 = torch.load(model_path1, weights_only=True)
state_dict_task2 = torch.load(model_path2, weights_only=True)
def eval_token(model, sentence, with_mask=True, no_sep=True):
    model.eval()
    data = tokenize_sentence(sentence)
    tokens = tokenizer.convert_ids_to_tokens(data['input_ids'].tolist()[0])
    
    mask = data['attention_mask'].to(device)
    input_ids = data['input_ids'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    # label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)
    # print(input_ids)
    
    pred = model(input_ids=input_ids, mask=mask, labels=None, token_type_ids=token_type_ids)

    # pred_clean  = []
    # if(with_mask):
        # pred_clean = pred[mask != 0]
    # else:
        # pred_clean = pred

    
    # predictions = pred_clean.tolist()    # print(sentence)
    # print(pred)
    prediction_label = [ids_to_labels[i] for i in pred[0]]

    if(no_sep):
        prediction_label = prediction_label[1:-1]
        tokens = tokens[1:len(prediction_label)+1]
    return prediction_label, tokens

def eval_class(model, sentence):
    model.eval()
    data = tokenize_sentence(sentence)
    # print(data)
    mask = data['attention_mask'].to(device)
    input_ids = data['input_ids'].to(device)
    # label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)
    
    out = model(input_ids, mask, None)
    logits = out.logits[0]

    predictions = logits.argmax(dim=0).tolist()    # print(sentence)
    # print(logits_clean)
    return predictions

stamp = config.id
torch.no_grad()
df_out = df.copy()
# print(label)

# with open(f'./out/{stamp}.out.log','w') as f:
    # f.write(f"model1:{model_path1}\nmodel2:{model_path2}")

logging.info(f"with model: 1: {model_path1}; 2: {model_path2}")

# model = BertForTokenClassification.from_pretrained(config.pretrained, num_labels=NUM_LABLES, cache_dir='./cache')
model = NER_Model()
model.load_state_dict(state_dict=state_dict_task1)

if use_cuda:
    model = model.cuda()

df_out['BIO_anno'] = df['text'].progress_map(lambda s: ' '.join(realign(* eval_token(model,s), s)))

model = BertForSequenceClassification.from_pretrained(config.pretrained, num_labels=3, cache_dir='./cache')

if use_cuda:
    model = model.cuda()

model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(state_dict=state_dict_task2)

df_out['class'] = df['text'].progress_map(lambda x: eval_class(model,x))


# print(df_out)
df_out.to_csv(os.path.join(out_dir, 'submission.csv'), sep=',',index=False, columns=['id','BIO_anno','class'])
# df_out.to_csv(f'/tmp/19-{stamp}-1.csv', sep=',',index=False, columns=['id','text','class'])
