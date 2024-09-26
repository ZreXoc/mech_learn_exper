# -*- coding: utf-8 -*-
import time
import torch

import os
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
from transformers import BertForTokenClassification, BertForSequenceClassification

from src.constants import LABELS, MAX_SEQ_LENGTH, MODEL_NAME, NUM_LABLES, PAD_LABEL, SPEC_LABEL, SPLITS
from src.dataset import CommentDataset
from src.model import BertModel
from src.tokenizer import tokenize_and_align_labels
# from transformers import BertModel, BertConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = 'true'

batch_size = 40 # Too small 
LEARNING_RATE = 3e-5
EPOCHS = 20
num_workers = 4
use_cuda = torch.cuda.is_available()
# use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

stamp = time.strftime("%d%H%M")

def train_loop1(model:torch.nn.Module):
    
    dataset = CommentDataset(SPLITS['train'])
    len_data = len(dataset)
    train_dataset, val_dataset = random_split(dataset, [int(len_data*0.9), len_data-int(len_data*0.9)])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers,drop_last=True)

    if use_cuda:
        model = model.cuda()


    # FULL_FINETUNING = True
    # if FULL_FINETUNING:
        # param_optimizer = list(model.named_parameters())
        # no_decay = ['bias', 'gamma', 'beta']
        # optimizer_grouped_parameters = [
            # {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             # 'weight_decay_rate': 0.01},
            # {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             # 'weight_decay_rate': 0.0}
        # ]
    # else:
        # param_optimizer = list(model.classifier.named_parameters())
        # optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    best_acc = 0
    best_loss = 1000
    model.zero_grad()
    for epoch_num in range(EPOCHS):
        total_acc_train = 0
        total_loss_train = 0


        train_idx = 0
        model.train()
        for train_data, train_label in tqdm(train_loader):
            train_idx += 1
            train_label = train_label.to(device)
            mask = train_data['attention_mask'].to(device)
            # mask = torch.ones_like(train_data['attention_mask']).to(device)
            input_id = train_data['input_ids'].to(device)
            token_type_ids = train_data['token_type_ids'].to(device)
            # print(train_label[0])
            # print(mask[0])
            # print(input_id[0])
            # print(input_id.size(), mask.size(),train_label.size(), token_type_ids.size())
            # 输入模型训练结果：损失及分类概率
            optimizer.zero_grad()
            loss, logits = model(input_ids=input_id,attention_mask=mask,labels=train_label,
                           token_type_ids=token_type_ids, return_dict=False, output_hidden_states=False, output_attentions=False)
            # print(logits.size(),train_label.size())
            # 过滤掉特殊token及padding的token
            logits_clean = []
            label_clean = []
            logits_clean = logits[train_label != PAD_LABEL]
            label_clean = train_label[train_label != PAD_LABEL]
            # print('#tr', train_label)
            # print(train_label.size(),logits_clean.size(),label_clean.size())
            # print(SPEC_LABEL, train_label[0])
            # print(SPEC_LABEL, train_label[1])
            # print(label_clean[:MAX_SEQ_LENGTH])
            # 获取最大概率值
            predictions = logits_clean.argmax(dim=1)
          # 计算准确率
            acc = (predictions == label_clean).float().mean() # TODO
            # print(acc, len(train_dataset))
            # print('logit',logits_clean)
            # print('pred',predictions, predictions.size())
            # print('pred',[torch.sum(predictions.eq(i)).item() for i in range(NUM_LABLES)])
            # print('label',[torch.sum(label_clean.eq(i)).item() for i in range(NUM_LABLES)])
            # print('id', train_idx)
            # print('acc',acc.item())
            # print('loss',loss.item())
            total_acc_train += acc.item()
            total_loss_train += loss.item()
      # 反向传递
            loss.backward()
            # 参数更新
            optimizer.step()

        total_acc_val = 0
        val_idx = 0
        model.eval()
        with torch.no_grad():
            for val_data, val_label in val_loader:
                val_idx+=1
                val_label = val_label.to(device)
                mask = val_data['attention_mask'].to(device)
                # mask = torch.ones_like(val_data['attention_mask']).to(device)
                input_id = val_data['input_ids'].to(device)
                token_type_ids = val_data['token_type_ids'].to(device)
                logits, = model(input_ids=input_id,attention_mask=mask,labels=None,
                               token_type_ids=token_type_ids, return_dict=False, output_hidden_states=False, output_attentions=False)
                logits_clean = []
                label_clean = []
                label_clean = val_label[val_label != PAD_LABEL]
                logits_clean = logits[val_label != PAD_LABEL]
                predictions = logits_clean.argmax(dim=1)
              # 计算准确率
                acc = (predictions == label_clean).float().mean() # TODO
                print('val acc',acc.item())
                total_acc_val += acc.item()

        print(
            f'''Epochs: {epoch_num + 1} | 
                Loss: {total_loss_train / train_idx: .8f} | 
                Accuracy: {total_acc_train / train_idx: .8f} |
                val_Accuracy: {total_acc_val / val_idx: .8f} |
               ''')

        if(not epoch_num % 2): torch.save(model.state_dict(), f'./model/{stamp}-e{epoch_num+1}.pth')

def train2(model:torch.nn.Module):
    dataset = CommentDataset(SPLITS['train'])
    len_data = len(dataset)
    train_dataset, val_dataset = random_split(dataset, [int(len_data*0.9), len_data-int(len_data*0.9)])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers,drop_last=True)

    if(use_cuda):
        model = model.cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    for epoch_num in range(EPOCHS):
        total_acc_train = 0
        total_loss_train = 0

        train_idx = 0
        model.train()
        for train_data, _train_label, mood_label in tqdm(train_loader):
            train_idx += 1
            mask = train_data['attention_mask'].to(device)
            # mask = torch.ones_like(train_data['attention_mask']).to(device)
            input_id = train_data['input_ids'].to(device)
            token_type_ids = train_data['token_type_ids'].to(device)
            mood_label = mood_label.to(device)
            # 输入模型训练结果：损失及分类概率
            optimizer.zero_grad()
            loss, logits = model(input_ids=input_id,attention_mask=mask,labels=mood_label,
                           token_type_ids=token_type_ids, return_dict=False, output_hidden_states=False, output_attentions=False)
            # 获取最大概率值
            predictions = logits.argmax(dim=1)
            # print(predictions, mood_label)
          # 计算准确率
            acc = (predictions == mood_label).float().mean() # TODO
            total_acc_train += acc.item()
            total_loss_train += loss.item()
      # 反向传递
            loss.backward()
            # 参数更新
            optimizer.step()

        total_acc_val = 0
        val_idx = 0
        model.eval()
        with torch.no_grad():
            for val_data, _val_label, mood_label in val_loader:
                val_idx+=1
                mood_label= mood_label.to(device)
                mask = val_data['attention_mask'].to(device)
                # mask = torch.ones_like(val_data['attention_mask']).to(device)
                input_id = val_data['input_ids'].to(device)
                token_type_ids = val_data['token_type_ids'].to(device)
                logits, = model(input_ids=input_id,attention_mask=mask,labels=None,
                               token_type_ids=token_type_ids, return_dict=False, output_hidden_states=False, output_attentions=False)
                predictions = logits.argmax(dim=1)
              # 计算准确率
                acc = (predictions == mood_label).float().mean()
                print('val acc',acc.item())
                total_acc_val += acc.item()

        print(
            f'''Epochs: {epoch_num + 1} | 
                Loss: {total_loss_train / train_idx: .8f} | 
                Accuracy: {total_acc_train / train_idx: .8f} |
                val_Accuracy: {total_acc_val / val_idx: .8f} |
               ''')
        
        os.mkdir('./model/{stamp}')
        if(not epoch_num % 1): torch.save(model.state_dict(), f'./model/{stamp}-t2-e{epoch_num+1}.pth')

if __name__ == '__main__':
    # model = BertModel.from_pretrained('bert-base-chinese')
    # model = BertModel(NUM_LABLES)
    # model1 = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABLES)
    # train_loop1(model1)

    model2 = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    train2(model2)
