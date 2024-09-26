import torch
from transformers import BertForTokenClassification, BertTokenizerFast

from src.constants import MODEL_NAME, PAD_LABEL, SPEC_LABEL

# tokenizer = BertTokenizerFast.from_pretrained('/home/xic/.cache/huggingface/hub/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f')
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(tokens,labels=None, tokenizer=tokenizer,
                              max_seq_len=512):
    # print(tokens)
    tokenized_input = tokenizer.encode_plus(
        tokens,
        return_length=True,
        # is_split_into_words=True, # TODO split English
        truncation=True,
        padding='max_length',
        max_length=max_seq_len,
        return_tensors='pt',
        )

    ids = tokenized_input['input_ids']
    ids_clean = ids[ids!=0]

    if(not labels): return tokenized_input

    # -2 for [CLS] and [SEP]
    if len(ids_clean) - 2 < len(labels):
        labels = labels[:len(ids_clean) - 1]
    tokenized_label = [SPEC_LABEL] + labels + [SPEC_LABEL]
    tokenized_label += [PAD_LABEL] * (
        tokenized_input['input_ids'].numel() - len(tokenized_label))

    tokenized_label = torch.tensor(tokenized_label)
    return tokenized_input, tokenized_label
