# -*- coding: utf-8 -*-
import logging
import torch

import os
from torch.utils.data import ConcatDataset, DataLoader, random_split

from tqdm import tqdm
from transformers import BertForTokenClassification, BertForSequenceClassification

from src.constants import LABELS, MAX_SEQ_LENGTH, NUM_LABLES, PAD_LABEL, SPEC_LABEL, SPLITS
from src.dataset import CommentDataset, MoodDataset
# from src.model import BertModel
from src.tokenizer2 import tokenizer
from transformers import BertModel, BertConfig
from src.config import config, out_dir

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = 'true'

SAVE_FREQUENCE = config.save_freq

task_id = config.id

BATCH_SIZE = config.batch_size  # Too small
LEARNING_RATE = config.lr
EPOCHS = config.niter
NUM_WORKERS = config.workers
use_cuda = config.cuda
# use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")


def train_loop1(model: torch.nn.Module):
    dataset = CommentDataset(SPLITS['train'])

    train_dataset, val_dataset = random_split(
        dataset, [0.9, 0.1])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    best_acc = 0
    best_loss = 1000
    logging.info(f'Start training task1')  
    model.zero_grad()
    for epoch_num in range(EPOCHS):
        total_acc_train = 0
        total_loss_train = 0

        train_idx = 0
        model.train()
        for train_data, train_label, _mood_label in tqdm(train_loader):
            train_idx += 1
            train_label = train_label.to(device)
            mask = train_data['attention_mask'].to(device)
            # mask = torch.ones_like(train_data['attention_mask']).to(device)
            input_id = train_data['input_ids'].to(device)
            token_type_ids = train_data['token_type_ids'].to(device)
            # print(input_id.size(), mask.size(),train_label.size(), token_type_ids.size())
            # 输入模型训练结果：损失及分类概率
            optimizer.zero_grad()
            loss, logits = model(input_ids=input_id, attention_mask=mask, labels=train_label,
                                 token_type_ids=token_type_ids, return_dict=False, output_hidden_states=False, output_attentions=False)
            # print(logits.size(),train_label.size())
            # 过滤掉特殊token及padding的token
            logits_clean = []
            label_clean = []
            logits_clean = logits[train_label != PAD_LABEL]
            label_clean = train_label[train_label != PAD_LABEL]
            # 获取最大概率值
            predictions = logits_clean.argmax(dim=1)



          # 计算准确率
            acc = (predictions == label_clean).float().mean()  # TODO
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
            for val_data, val_label, _mood_label in val_loader:
                val_idx += 1
                # print(val_idx)
                val_label = val_label.to(device)
                mask = val_data['attention_mask'].to(device)
                # mask = torch.ones_like(val_data['attention_mask']).to(device)
                input_id = val_data['input_ids'].to(device)
                token_type_ids = val_data['token_type_ids'].to(device)
                logits, = model(input_ids=input_id, attention_mask=mask, labels=None,
                                token_type_ids=token_type_ids, return_dict=False, output_hidden_states=False, output_attentions=False)
                logits_clean = []
                label_clean = []
                label_clean = val_label[val_label != PAD_LABEL]
                logits_clean = logits[val_label != PAD_LABEL]
                predictions = logits_clean.argmax(dim=1)
              # 计算准确率
                acc = (predictions == label_clean).float().mean()  # TODO
                # print('val acc', acc.item())
                total_acc_val += acc.item()

        logging.info(
            f'''Epochs: {epoch_num + 1} | 
                Loss: {total_loss_train / train_idx: .8f} | 
                Accuracy: {total_acc_train / train_idx: .8f} |
                val_Accuracy: {total_acc_val / val_idx: .8f} |
               ''')

        if (SAVE_FREQUENCE and not epoch_num % SAVE_FREQUENCE):
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f't1-e{epoch_num+1}.pt'))


def train2(model: torch.nn.Module):
    dataset = MoodDataset()
    train_dataset, val_dataset = random_split(
        dataset, [0.9,0.1])
    train_dataset = ConcatDataset([train_dataset, MoodDataset(fromRetrans=True)])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

    if (use_cuda):
        model = model.cuda()

    logging.info(f'Start training task2')  
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    for epoch_num in range(EPOCHS):
        total_acc_train = 0
        total_loss_train = 0
        train_idx = 0
        model.train()
        for train_data, mood_label in tqdm(train_loader):
            train_idx += 1
            mask = train_data['attention_mask'].to(device)
            # mask = torch.ones_like(train_data['attention_mask']).to(device)
            input_id = train_data['input_ids'].to(device)
            token_type_ids = train_data['token_type_ids'].to(device)
            mood_label = mood_label.to(device)
            # 输入模型训练结果：损失及分类概率
            optimizer.zero_grad()
            loss, logits = model(input_ids=input_id, attention_mask=mask, labels=mood_label,
                                 token_type_ids=token_type_ids, return_dict=False, output_hidden_states=False, output_attentions=False)
            # 获取最大概率值
            predictions = logits.argmax(dim=1)
            # print(predictions, mood_label)
          # 计算准确率
            acc = (predictions == mood_label).float().mean()  # TODO
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
            for val_data, mood_label in val_loader:
                val_idx += 1
                mood_label = mood_label.to(device)
                mask = val_data['attention_mask'].to(device)
                # mask = torch.ones_like(val_data['attention_mask']).to(device)
                input_id = val_data['input_ids'].to(device)
                token_type_ids = val_data['token_type_ids'].to(device)
                logits, = model(input_ids=input_id, attention_mask=mask, labels=None,
                                token_type_ids=token_type_ids, return_dict=False, output_hidden_states=False, output_attentions=False)
                predictions = logits.argmax(dim=1)
              # 计算准确率
                acc = (predictions == mood_label).float().mean()
                # print('val acc',acc.item())
                total_acc_val += acc.item()

        logging.info(
            f'''Epochs: {epoch_num + 1} | 
                Loss: {total_loss_train / train_idx: .8f} | 
                Accuracy: {total_acc_train / train_idx: .8f} |
                val_Accuracy: {total_acc_val / val_idx: .8f} |
               ''')

        if (SAVE_FREQUENCE and not epoch_num % SAVE_FREQUENCE):
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f't2-e{epoch_num+1}.pt'))

if __name__ == '__main__':
    # logging.info('''Task 1
# ############################################
# ''')
    # model1 = BertForTokenClassification.from_pretrained(config.pretrained, num_labels=NUM_LABLES,cache_dir='./cache')
    # model1.resize_token_embeddings(len(tokenizer))
    # train_loop1(model1)

    logging.info('''Task 2
############################################
''')
    model2 = BertForSequenceClassification.from_pretrained(
        config.pretrained, num_labels=3, cache_dir='./cache')
    model2.resize_token_embeddings(len(tokenizer))
    train2(model2)
