
import pandas as pd
from pandas.core.dtypes.dtypes import np
import torch
from torch.utils.data import Dataset
from src.augment.replace import rep_bank, replace
from src.constants import LABELS, SPEC_LABEL, SPLITS,  labels_to_ids, ids_to_labels, MAX_SEQ_LENGTH
# from src.tokenizer import tokenize_and_align_labels, tokenizer
from src.tokenizer2 import tokenize_sentence

class CommentDataset(Dataset):
    def __init__(self, path) -> None:
        self.path = path
        df = pd.read_csv(path, index_col=0)
        # df = df.head(100)

        df = self.preprocess_data(df)

        # df['text'] = df['text'].map(lambda x: list(x))
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

        self.df:pd.DataFrame = df
        self.columns = self.df.columns
        self.tokens = tokens
        self.labels = labels
        self.tokenized_data = tokenized_data
        self.tokenized_label = tokenized_label
        self.mood_label = torch.tensor(df['class'].values)

    def preprocess_data(self, df:pd.DataFrame):
        df['BIO_anno'] = df['BIO_anno'].map(lambda x: [labels_to_ids[i] for i in x.split()])

        # 同类词替换
        # nd = []
        # random_indices = np.random.choice(df.index, size=len(df)*2, replace=True)

        # random_samples = df.loc[random_indices]

        # for i, row in random_samples.iterrows():

            # text = row['text']
            # label = row['BIO_anno']
            # mood = row['class']

            # nt,nl,cu = replace(text,label, p=1, replaceFn=rep_bank)
            # nd += [(''.join(nt),nl,mood)]


        # ndf = pd.DataFrame(nd, columns=df.columns)

        # df = pd.concat([df,ndf])
        # print(df)
        # print(len(df))

        return df

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
