import torch
import time
from transformers import BertForSequenceClassification, BertForTokenClassification
from src.model import BertModel
import pandas as pd
from src.constants import MAX_SEQ_LENGTH, MODEL_NAME, NUM_LABLES, SPEC_LABEL, SPLITS, ids_to_labels
from src.tokenizer import tokenize_and_align_labels, tokenizer

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

df = pd.read_csv(SPLITS['test'])
# df = df.head()

tokenized_data = []

model_path1='./model/large/261135-e19.pth'
model_path2='./model/large/261435/261435-t2-e9.pth'

state_dict_task1 = torch.load(model_path1, weights_only=False)
state_dict_task2 = torch.load(model_path2, weights_only=False)

def eval_token(model, sentence, with_mask=True, no_sep=True):
    model.eval()
    data =  tokenizer.encode_plus(
        list(sentence),
        return_length=True,
        truncation=True,
        is_split_into_words=True, # TODO split English
        padding='max_length',
        max_length=MAX_SEQ_LENGTH,
        return_tensors='pt'
        )
    # print(data)
    mask = data['attention_mask'].to(device)
    input_ids = data['input_ids'].to(device)
    # label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)
    
    logits = model(input_ids, mask, None)

    logits_clean  = []
    if(with_mask):
        logits_clean = logits[0][mask != 0]
    else:
        logits_clean = logits[0]

    
    predictions = logits_clean.argmax(dim=1).tolist()    # print(sentence)
    # print(logits_clean)
    prediction_label = [ids_to_labels[i] for i in predictions]
    if(no_sep): prediction_label = prediction_label[1:len(prediction_label)-1]
    return prediction_label

def eval_class(model, sentence):
    model.eval()
    data =  tokenizer.encode_plus(
        list(sentence),
        return_length=True,
        truncation=True,
        is_split_into_words=True, # TODO split English
        padding='max_length',
        max_length=MAX_SEQ_LENGTH,
        return_tensors='pt'
        )
    # print(data)
    mask = data['attention_mask'].to(device)
    input_ids = data['input_ids'].to(device)
    # label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)
    
    out = model(input_ids, mask, None)
    logits = out.logits[0]

    predictions = logits.argmax(dim=0).tolist()    # print(sentence)
    # print(logits_clean)
    return predictions
    ...

# label = evaluate_single_text(model, '哦，我搞错了?我的贷款是汇丰银行的国债。他们的货款的服务令人伤心')

stamp = time.strftime("%d%H%M")
torch.no_grad()
df_out = df.copy()
# print(label)

with open(f'./out/{stamp}.out.log','w') as f:
    f.write(f"model1:{model_path1}\nmodel2:{model_path2}")

model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABLES)
model.load_state_dict(state_dict=state_dict_task1)

if use_cuda:
    model = model.cuda()
df_out['BIO_anno'] = df['text'].map(lambda x: ' '.join(eval_token(model,x)))

model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
if use_cuda:
    model = model.cuda()
model.load_state_dict(state_dict=state_dict_task2)

df_out['class'] = df['text'].map(lambda x: eval_class(model,x))

# print(df_out)
df_out.to_csv(f'./out/{stamp}.csv', sep=',',index=False, columns=['id','BIO_anno','class'])
# df_out.to_csv(f'/tmp/19-{stamp}-1.csv', sep=',',index=False, columns=['id','text','class'])
