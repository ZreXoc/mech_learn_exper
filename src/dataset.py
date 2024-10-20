
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.constants import LABELS, SPEC_LABEL, SPLITS,  labels_to_ids, ids_to_labels, MAX_SEQ_LENGTH
# from src.tokenizer import tokenize_and_align_labels, tokenizer
from src.tokenizer2 import tokenize_sentence

class CommentDataset(Dataset):
    def __init__(self, path) -> None:
        self.path = path
        df = pd.read_csv(path, index_col=0)
        # df = df.head(1)

        # df['text'] = df['text'].map(lambda x: list(x))
        df['BIO_anno'] = df['BIO_anno'].map(lambda x: [labels_to_ids[i] for i in x.split()])
        tokens = df['text'].values
        labels = df['BIO_anno'].values

        tokenized_data = []
        tokenized_label= []
        for i, row in df.iterrows():
            data, label =  tokenize_sentence(
                row['text'], row['BIO_anno'], max_seq_len=MAX_SEQ_LENGTH)
            data['input_ids'] = data['input_ids'][0]
            data['attention_mask'] = data['attention_mask'][0]
            data['token_type_ids'] = data['token_type_ids'][0]

            tokenized_data.append(data)
            tokenized_label.append(label)

        self.df = df
        self.columns = self.df.columns
        self.tokens = tokens
        self.labels = labels
        self.tokenized_data = tokenized_data
        self.tokenized_label = tokenized_label
        self.mood_label = torch.tensor(df['class'].values)

    def __getitem__(self, index):
        return self.tokenized_data[index],self.tokenized_label[index],self.mood_label[index]

    def get_labels(self):
        pass

    def __len__(self):
        if len(self.df) == 0:
            raise Exception("\n no data found in {}".format(
                self.path))  # 代码具有友好的提示功能，便于debug
        return len(self.df)

class MoodDataset(Dataset):
    def __init__(self, fromRetrans = False) -> None:
        self.isRetrans = fromRetrans

        data = None


        if(fromRetrans):
            data = pd.read_csv('./data/trans.csv').rename(columns={"cn":"text"})
            data.drop(columns=['en'],inplace=True)
            data['isRetrans'] = 1
        else:
            data = pd.read_csv('./data/train_data_public.csv').drop(columns=['BIO_anno'])
            data['isRetrans']=0

        # data = data.head(1000)

        tokenized_data=[]
        for i, row in data.iterrows():
            token =  tokenize_sentence(
                row['text'], max_seq_len=MAX_SEQ_LENGTH)
            token['input_ids'] = token['input_ids'][0]
            token['attention_mask'] = token['attention_mask'][0]
            token['token_type_ids'] = token['token_type_ids'][0]

            tokenized_data.append(token)

        self.tokenized_data = tokenized_data
        self.mood = data['class'].values
        self.data = data

    def __getitem__(self, index):
        return self.tokenized_data[index], self.mood[index]

    def __len__(self):
        return len(self.data)
