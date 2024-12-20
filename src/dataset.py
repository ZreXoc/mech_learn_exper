
import pandas as pd
from pandas.core.dtypes.dtypes import np
import torch
from torch.utils.data import Dataset
from src.augment.replace import rep_bank, replace
from src.constants import LABELS, SPLITS,  labels_to_ids, ids_to_labels, MAX_SEQ_LENGTH
# from src.tokenizer import tokenize_and_align_labels, tokenizer
from src.tokenizer2 import tokenize_sentence
from src.config import config
from tqdm import tqdm

tqdm.pandas()

class CommentDataset(Dataset):
    def __init__(self, path, withReplace=False) -> None:
        self.path = path
        df = pd.read_csv(path, index_col=0)
        # df.drop(columns=['bank_topic'], inplace=True)

        df = self.preprocess_data(df, withReplace)
        if(config.debug): df = df.head(500)


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

    def preprocess_data(self, df:pd.DataFrame, withReplace=False):
        df['BIO_anno'] = df['BIO_anno'].map(lambda x: [labels_to_ids[i] for i in x.split()])

        if(withReplace):
            # 同类词替换
            nd = []
            random_indices = np.random.choice(df.index, size=len(df)*2, replace=True)

            random_samples = df.loc[random_indices]

            for i, row in random_samples.iterrows():

                text = row['text']
                label = row['BIO_anno']
                mood = row['class']

                nt,nl,cu = replace(text,label, p=1, replaceFn=rep_bank)
                nd += [(''.join(nt),nl,mood)]


            ndf = pd.DataFrame(nd, columns=df.columns)

            df = pd.concat([df,ndf])
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
    def __init__(self, withOrigin = True,withGPT=False, mark = False ,fromRetrans = False, withExt = False) -> None:
        self.isRetrans = fromRetrans

        data = None

        if(withOrigin):
            data = pd.read_csv(SPLITS['train'])
            if(mark):
                rpO = lambda t: '' if t =='O' else t;
                data['text'] = data.apply(lambda row: ''.join([rpO(a) + b for a, b in zip(row['BIO_anno'].split(), row['text'])]), axis=1)
            data.drop(columns=['BIO_anno'],inplace=True)
        else:
            data = pd.read_csv(SPLITS['train'],nrows=0).drop(columns=['BIO_anno'])

        # data.set_index('id', inplace=True)
        data['type'] = 'origin';


        if(withGPT):
            df = pd.read_csv('./data/gpt.csv')
            df['type'] = 'gpt'
            # df.set_index('id', inplace=True)
            data = pd.concat([data,df])

        if(withExt):
            df = pd.read_csv('./data/online_shopping_10_cats.csv')
            df = df[df.iloc[:, 0] == '计算机']
            df = df.rename(columns={'review': 'text', 'label': 'class'})
            # df.drop(columns=['cat'])
            df['type'] = 'ext'
            # print(df.sample(5))

            data = pd.concat([data,df[['text', 'class','type']]])

        if(fromRetrans):
            df = pd.read_csv('./data/trans.csv').rename(columns={"cn":"text"})
            df.drop(columns=['en'],inplace=True)
            df['type'] = 'trans';
            # print(df.sample(5))
            data = pd.concat([data,df])

        # print(data.tail())

        # data = data.head(10)

        # rows_with_nan = data[data.isnull().any(axis=1)]
        # print(rows_with_nan)

        data['class'] = data['class'].astype(int)

        tokenized_data=[]
        for i, row in data.iterrows():
            token =  tokenize_sentence(
                row['text'], max_seq_len=MAX_SEQ_LENGTH)
            token['input_ids'] = token['input_ids'][0]
            token['attention_mask'] = token['attention_mask'][0]
            token['token_type_ids'] = token['token_type_ids'][0]

            tokenized_data.append(token)

        # data.reset_index(inplace=True)
        data.reset_index(drop=True, inplace=True)

        self.tokenized_data = tokenized_data
        self.mood = data['class'].values
        self.data = data

    def __getitem__(self, index):
        # print(self.data.iloc[index], self.mood[index])
        return self.tokenized_data[index], self.mood[index]

    def __len__(self):
        return len(self.data)
