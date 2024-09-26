import torch
from transformers import BertForTokenClassification
from src.model import BertModel
import pandas as pd
from src.constants import MAX_SEQ_LENGTH, MODEL_NAME, NUM_LABLES, SPEC_LABEL, SPLITS, ids_to_labels
from src.tokenizer import tokenize_and_align_labels, tokenizer

batch_size = 50 # Too small 
LEARNING_RATE = 0.1
EPOCHS = 20
num_workers = 4
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

df = pd.read_csv(SPLITS['test'])
# df = df.head()

tokenized_data = []

# for i, row in df.iterrows():
    # data =  tokenizer.encode_plus(
        # row['text'].split(),
        # return_length=True,
        # is_split_into_words=True,
        # truncation=True,
        # padding='max_length',
        # max_length=MAX_SEQ_LENGTH,
        # return_tensors='pt'
        # )
    # tokenized_data.append(data)



# print(tokenized_data)
state_dict = torch.load('./model/large/261135-e19.pth', weights_only=False)
model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABLES)
model.load_state_dict(state_dict=state_dict)

if use_cuda:
    model = model.cuda()

def evaluate_single_text(model, sentence, with_mask=True, no_sep=True):
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

label = evaluate_single_text(model, '哦，我搞错了?我的贷款是汇丰银行的国债。他们的货款的服务令人伤心')

print(label)
# df['BIO_anno'] = df['text'].map(lambda x: ' '.join(evaluate_single_text(model,x)))

# df.to_csv('./temp.csv', sep=',',index=False, columns=['id','BIO_anno'])

