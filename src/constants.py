import os

# MODEL_NAME = 'hfl/chinese-bert-wwm'
# MODEL_NAME = '/home/xic/.cache/huggingface/hub/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f'

MAX_SEQ_LENGTH = 200 # 最长长度约为一百九十

ROOT_DIR = os.path.curdir

SPLITS = {
    'train': os.path.join(ROOT_DIR,'data/train_data_public_2.csv'),
    'test': os.path.join(ROOT_DIR, 'data/test_public.csv'),
}

MOOD_TYPE = {
    'NEG': 0,
    'POS': 1,
    'NEU': 2
}

LABELS =  ["B-BANK", "I-BANK", "B-PRODUCT", "I-PRODUCT", 'B-COMMENTS_N', 'I-COMMENTS_N', 'B-COMMENTS_ADJ', 'I-COMMENTS_ADJ', 'O']

SPEC_LABEL = len(LABELS) 
PAD_LABEL = len(LABELS) + 1
# SPEC_LABEL = -1

labels_to_ids = {k: v for v, k in enumerate((LABELS))}
ids_to_labels = {v: k for v, k in enumerate((LABELS))}
ids_to_labels[SPEC_LABEL] = '[UNK]'
ids_to_labels[PAD_LABEL] = '[PAD]'
print(ids_to_labels)

NUM_LABLES = len(LABELS)+ 2



COMMENT_COLUMNS = {
    'text': 'text',
    'BIO_anno': "BIO_anno",
    'class': 'class'
}
